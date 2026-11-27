"""Exact zero-drift HuggingFace <-> safetensors checkpoint exporter and allowlist guard."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import shutil
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORBIDDEN_SUBSTRINGS = (
    "experts",
    "visual",
    "embed_tokens",
    "lm_head",
    "scale",
)


def normalize_param_name(name: str) -> str:
    """Normalize shorthand layer names (e.g. 'layers.6...') to canonical HF 'model.layers.6...'."""
    if name.startswith("layers.") or name in ("message_input.query", "final_input.query"):
        return "model." + name
    return name


def is_editable_param(name: str) -> bool:
    """Return True iff name is one of the unquantized BF16/FP32 control/attention/router tensors."""
    n = normalize_param_name(name)
    if any(bad in n for bad in FORBIDDEN_SUBSTRINGS):
        return False
    if n in ("model.message_input.query", "model.final_input.query"):
        return True
    if not n.startswith("model.layers."):
        return False
    allowed_suffixes = (
        ".attention_input.query",
        ".mlp_input.query",
        ".self_attn.o_proj.linear.weight",
        ".self_attn.o_proj.gate_proj.weight",
        ".self_attn.q_proj.linear.weight",
        ".self_attn.k_proj.linear.weight",
        ".self_attn.v_proj.linear.weight",
        ".post_attention_convolution.conv1d.weight",
        ".post_mlp_convolution.conv1d.weight",
        ".mlp.gate.weight",
        ".mlp.gate.e_score_correction_bias",
    )
    return ("_message_gate." in n and n.endswith(".weight")) or n.endswith(allowed_suffixes)


def _legacy_transformers_name_to_ckpt_deltas(
    tf_name: str, delta: torch.Tensor
) -> List[Tuple[str, Optional[slice], torch.Tensor]]:
    """Map a HuggingFace Inkling2LoopForCausalLM parameter delta back to checkpoint safetensors keys.

    Returns a list of (checkpoint_key, row_slice_or_None, delta_to_add).
    """
    tf_name = normalize_param_name(tf_name)
    if not is_editable_param(tf_name):
        raise ValueError(f"Refusing to map non-editable or forbidden parameter: {tf_name}")

    d = delta.detach().float().cpu()
    if tf_name == "model.message_input.query":
        return [("model.llm.middle_cycle_cross_pass_attn_res.query", None, d)]
    if tf_name == "model.final_input.query":
        return [("model.llm.attn_res.layers.24.attention_input.query", None, d)]

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", tf_name)
    if not m:
        raise ValueError(f"Unsupported parameter name: {tf_name}")
    lidx, rest = int(m.group(1)), m.group(2)

    if rest == "attention_input.query":
        if lidx == 0:
            raise ValueError("Layer 0 has no attention_input.query in checkpoint.")
        return [(f"model.llm.attn_res.layers.{lidx}.attention_input.query", None, d)]
    if rest == "mlp_input.query":
        return [(f"model.llm.attn_res.layers.{lidx}.mlp_input.query", None, d)]
    if rest == "self_attn.o_proj.gate_proj.weight":
        return [(f"model.llm.layers.{lidx}.attn.wo_ud.weight", slice(0, 3072), d)]
    if rest == "self_attn.o_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.wo_ud.weight", slice(3072, 6144), d.T.contiguous())]
    if rest == "self_attn.q_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(0, 3072), d)]
    if rest == "self_attn.k_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(3072, 4096), d)]
    if rest == "self_attn.v_proj.linear.weight":
        return [(f"model.llm.layers.{lidx}.attn.qkvr.weight", slice(4096, 5120), d)]
    if rest == "post_attention_convolution.conv1d.weight":
        return [(f"model.llm.layers.{lidx}.attn_sconv.conv.weight", None, d)]
    if rest == "post_mlp_convolution.conv1d.weight":
        return [(f"model.llm.layers.{lidx}.mlp_sconv.conv.weight", None, d)]
    if rest == "mlp.gate.weight":
        return [(f"model.llm.layers.{lidx}.moe.router.weight", None, d)]
    if rest == "mlp.gate.e_score_correction_bias":
        return [(f"model.llm.layers.{lidx}.moe.router_bias", None, d)]

    mg = re.match(r"^(q|k|v|mlp)_message_gate\.(down_proj|forget_gate_proj|message_gate_proj)\.weight$", rest)
    if mg:
        if not (6 <= lidx < 18):
            raise ValueError(f"MessageGate only exists on middle loop layers 6..17, got L{lidx}")
        kind, proj = mg.group(1), mg.group(2)
        loop_off = lidx - 6
        if proj == "down_proj":
            return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.down_proj.weight", None, d)]
        if proj == "forget_gate_proj":
            return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.up_proj.weight", slice(0, 2304), d)]
        return [(f"model.llm.middle_cycle_cross_pass_gates.{loop_off}.{kind}_gate.up_proj.weight", slice(2304, 4608), d)]

    raise ValueError(f"Unmapped editable parameter: {tf_name}")


def _upstream_inkling_deltas(name: str, delta: torch.Tensor):
    """Inverse of the supported upstream Inkling2Loop loader's original format.

    Unlike the older packed layout, these controllers/projections are separate
    tensors. Message queries alone need a split, reversing upstream's
    Inkling2LoopMergeMessageQueries (query, key, value, MLP order).
    """
    if name == "model.message_input.query":
        if delta.ndim != 3 or delta.shape[1] != 4:
            raise ValueError("message_input.query requires shape (loop_layers, 4, hidden_size)")
        return [
            ("model.llm.middle_cycle_cross_pass_queries", None, delta[:, [0, 3]].flatten(0, 1).contiguous()),
            ("model.llm.middle_cycle_cross_pass_kv_queries", None, delta[:, 1:3].flatten(0, 1).contiguous()),
        ]
    if name == "model.final_input.query":
        return [("model.llm.attn_res.output_query", None, delta)]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if not match:
        return []
    layer, rest = int(match[1]), match[2]
    if rest == "attention_input.query":
        return [(f"model.llm.attn_res.attn_queries.{layer - 1}", None, delta)]
    if rest == "mlp_input.query":
        return [(f"model.llm.attn_res.mlp_queries.{layer}", None, delta)]
    prefix = f"model.llm.layers.{layer}."
    names = {
        "self_attn.q_proj.linear.weight": "attn.wq_du.weight",
        "self_attn.k_proj.linear.weight": "attn.wk_dv.weight",
        "self_attn.v_proj.linear.weight": "attn.wv_dv.weight",
        "self_attn.o_proj.linear.weight": "attn.wo_ud.weight",
        "self_attn.o_proj.gate_proj.weight": "attn.gate_attn.weight",
        "post_attention_convolution.conv1d.weight": "attn_sconv.weight",
        "post_mlp_convolution.conv1d.weight": "mlp_sconv.weight",
        "mlp.gate.weight": "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
    }
    if rest in names:
        return [(prefix + names[rest], None, delta)]
    gate = re.fullmatch(r"(q|k|v|mlp)_message_gate\.(down_proj|forget_gate_proj|message_gate_proj)\.weight", rest)
    if gate:
        kind = "attn" if gate[1] == "q" else gate[1]
        projection = {"down_proj": "down_weight", "forget_gate_proj": "forget_weight",
                      "message_gate_proj": "gain_weight"}[gate[2]]
        return [(prefix + f"controllers.{kind}.{projection}", None, delta)]
    return []


def _mapping_matches(mapping, shapes):
    if not mapping:
        return False
    for key, rows, delta in mapping:
        if key not in shapes:
            return False
        shape = tuple(shapes[key])
        if rows is not None:
            if not shape:
                return False
            shape = (len(range(*rows.indices(shape[0]))), *shape[1:])
        if tuple(delta.shape) != shape:
            return False
    return True


def transformers_name_to_ckpt_deltas(
    tf_name: str, delta: torch.Tensor, *, checkpoint_shapes: Mapping[str, Sequence[int]] | None = None,
) -> List[Tuple[str, Optional[slice], torch.Tensor]]:
    """Map edits using observed keys AND shapes; retain the old two-argument API.

    Exports always supply the actual checkpoint layout. This distinguishes the
    upstream separate wo_ud matrix from the legacy packed tensor sharing its
    name, avoiding an incorrect transpose or slice. Without a layout, return
    the historical mapping for backward compatibility only.
    """
    if checkpoint_shapes is None:
        return _legacy_transformers_name_to_ckpt_deltas(tf_name, delta)
    name = normalize_param_name(tf_name)
    if not is_editable_param(name):
        raise ValueError(f"Refusing to map non-editable or forbidden parameter: {name}")
    delta = delta.detach().float().cpu()
    if name in checkpoint_shapes:
        return [(name, None, delta)]
    candidates = [_upstream_inkling_deltas(name, delta)]
    if _mapping_matches(candidates[0], checkpoint_shapes):
        return candidates[0]
    try:
        candidates.append(_legacy_transformers_name_to_ckpt_deltas(name, delta))
    except ValueError:
        pass
    if len(candidates) > 1 and _mapping_matches(candidates[1], checkpoint_shapes):
        return candidates[1]
    if any(mapping and all(key in checkpoint_shapes for key, _, _ in mapping) for mapping in candidates):
        raise ValueError(f"Checkpoint tensor shapes do not match a supported mapping for '{name}'")
    raise KeyError(f"No supported checkpoint keys for '{name}'")


def apply_weight_deltas(
    model: torch.nn.Module, deltas: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """In-place apply FP32 deltas to model parameters/buffers, returning exact CPU backups."""
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    backups: Dict[str, torch.Tensor] = {}
    prepared = {}
    # Validate every edit first: broadcasting or a late error must not partially
    # mutate an otherwise valid checkpoint.
    for raw_k, d in deltas.items():
        k = normalize_param_name(raw_k)
        if k in prepared:
            raise ValueError(f"Duplicate normalized parameter '{k}'")
        if k not in named:
            raise KeyError(f"Parameter '{k}' not present in model.")
        t = named[k]
        if tuple(t.shape) != tuple(d.shape):
            raise ValueError(f"Delta shape mismatch for '{k}': {tuple(d.shape)} vs {tuple(t.shape)}")
        if not t.is_floating_point() or not torch.isfinite(d).all():
            raise ValueError(f"Delta for '{k}' must be finite and target floating-point weights")
        updated = (t.detach().float() + d.detach().to(t.device, dtype=torch.float32)).to(t.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError(f"Delta for '{k}' produces non-finite weights")
        prepared[k] = updated
    with torch.no_grad():
        for k, updated in prepared.items():
            t = named[k]
            backups[k] = t.detach().cpu().clone()
            t.copy_(updated)
    return backups


def restore_weights(model: torch.nn.Module, backups: Mapping[str, torch.Tensor]) -> None:
    """Restore model parameters/buffers from backups returned by apply_weight_deltas."""
    named = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with torch.no_grad():
        for k, orig in backups.items():
            named[k].data.copy_(orig.to(named[k].device, dtype=named[k].dtype))


def resolve_checkpoint_dir(
    model_or_path: str | Path, revision: str | None = None, *, local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or Hub ID to a concrete local checkpoint snapshot."""
    path = Path(model_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or str(model_or_path).startswith(("./", "../", "~")):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_or_path}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=str(model_or_path), revision=revision, local_files_only=local_files_only,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.py", "*.jinja"],
    )).resolve()


def _checkpoint_layout(base: Path, requested_filename: str):
    index_path = base / "model.safetensors.index.json"
    declared = None
    if index_path.exists():
        declared = json.loads(index_path.read_text())["weight_map"]
        files = sorted(set(declared.values()))
    else:
        candidates = sorted(p.name for p in base.glob("*.safetensors"))
        if len(candidates) != 1:
            raise ValueError("Expected one safetensors file or a model.safetensors.index.json shard index")
        files = candidates
        if requested_filename != "model-00001-of-00001.safetensors" and requested_filename not in files:
            raise FileNotFoundError(f"Requested safetensors file not found: {requested_filename}")
    weight_map, shapes, elements = {}, {}, 0
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid checkpoint shard path: {name}")
        with safe_open(str(base / name), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                weight_map[key] = name
                shapes[key] = tuple(reader.get_slice(key).get_shape())
                elements += math.prod(shapes[key])
    if declared is not None and declared != weight_map:
        raise ValueError("Shard index does not match the actual tensor keys")
    return files, weight_map, shapes, elements


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)))


def export_edited_checkpoint(
    base_dir: str,
    out_dir: str,
    deltas_dict: Mapping[str, torch.Tensor],
    safetensors_filename: str = "model-00001-of-00001.safetensors",
) -> List[Dict[str, object]]:
    """Export local/Hub, single-file or sharded checkpoints and verify saved tensor integrity.

    Native HF keys are edited directly. Original Inkling checkpoint keys use the
    explicit inverse mapping above. Quantized and other untargeted tensors must
    remain bit-for-bit identical. The audit counts stored elements, not unpacked
    logical parameters of quantized tensors.
    """
    base = resolve_checkpoint_dir(base_dir)
    out = Path(out_dir).expanduser().resolve()
    if out == base or base in out.parents or out in base.parents:
        raise ValueError("Output checkpoint must be separate from the source directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Output checkpoint directory must be new or empty; refusing to mix checkpoint files")
    files, weight_map, shapes, elements = _checkpoint_layout(base, safetensors_filename)
    ckpt_edits: Dict[str, List[Tuple[Optional[slice], torch.Tensor, str]]] = {}
    seen = set()
    for tf_name, delta in deltas_dict.items():
        name = normalize_param_name(tf_name)
        if name in seen:
            raise ValueError(f"Duplicate normalized parameter: {name}")
        seen.add(name)
        if not is_editable_param(name):
            raise ValueError(f"Refusing to edit forbidden parameter: {name}")
        if not torch.isfinite(delta).all():
            raise ValueError(f"Delta for '{name}' must be finite")
        mapped = transformers_name_to_ckpt_deltas(name, delta, checkpoint_shapes=shapes)
        for ck, sl, d in mapped:
            if ck not in weight_map:
                raise KeyError(f"Checkpoint key '{ck}' missing from {base}")
            ckpt_edits.setdefault(ck, []).append((sl, d, name))

    # Validate and prepare all targeted tensors before creating any output files.
    report: List[Dict[str, object]] = []
    replacements = {}
    for ck, edit_list in ckpt_edits.items():
        with safe_open(str(base / weight_map[ck]), framework="pt", device="cpu") as reader:
            orig = reader.get_tensor(ck)
        if orig.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise TypeError(f"Refusing to modify quantized tensor '{ck}' (dtype={orig.dtype}).")
        updated = orig.float().clone()
        for sl, d, _ in edit_list:
            if sl is None:
                if tuple(updated.shape) != tuple(d.shape):
                    raise ValueError(f"Shape mismatch on '{ck}': {tuple(updated.shape)} vs {tuple(d.shape)}")
                updated += d
            else:
                if tuple(updated[sl].shape) != tuple(d.shape):
                    raise ValueError(f"Slice shape mismatch on '{ck}'[{sl}]: {tuple(updated[sl].shape)} vs {tuple(d.shape)}")
                updated[sl] += d
        updated = updated.to(orig.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError(f"Edit produces non-finite values in '{ck}'")
        diff = updated.float() - orig.float()
        rel = diff.norm().item() / (orig.float().norm().item() + 1e-12)
        replacements[ck] = updated
        report.append({
            "ckpt_key": ck,
            "dtype": str(orig.dtype),
            "shape": list(orig.shape),
            "rel_change": round(rel, 6),
            "sources": [src_name for _, _, src_name in edit_list],
        })

    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.name in set(files) | {".cache", ".git", "weight_edit_report.json", "weight_integrity_audit.json"}:
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out / item.name)

    changed_keys = []
    for name in files:
        src, dst = base / name, out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        edited = {k: t for k, t in replacements.items() if weight_map[k] == name}
        if not edited:
            shutil.copy2(src, dst)
            if _file_sha256(src) != _file_sha256(dst):
                raise RuntimeError(f"Unedited shard changed during export: {name}")
            continue
        with safe_open(str(src), framework="pt", device="cpu") as reader:
            tensors = {k: reader.get_tensor(k) for k in reader.keys()}
            metadata = reader.metadata()
        originals = dict(tensors)
        tensors.update(edited)
        save_file(tensors, str(dst), metadata=metadata)
        with safe_open(str(dst), framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(originals):
                raise RuntimeError(f"Export changed tensor keys in {name}")
            for key, original in originals.items():
                actual = saved.get_tensor(key)
                if actual.shape != original.shape or actual.dtype != original.dtype:
                    raise RuntimeError(f"Export changed shape or dtype: {key}")
                if not _same_bits(actual, tensors[key]):
                    raise RuntimeError(f"Saved tensor differs from prepared value: {key}")
                if not _same_bits(actual, original):
                    if key not in edited:
                        raise RuntimeError(f"Untargeted tensor changed: {key}")
                    changed_keys.append(key)
    audit = {
        "source": str(base), "tensor_keys": len(weight_map), "stored_elements": elements,
        "changed_keys": sorted(changed_keys), "unchanged_keys_verified": len(weight_map) - len(changed_keys),
        "shapes_dtypes_preserved": True,
    }
    (out / "weight_edit_report.json").write_text(json.dumps(report, indent=2))
    (out / "weight_integrity_audit.json").write_text(json.dumps(audit, indent=2))
    return report
