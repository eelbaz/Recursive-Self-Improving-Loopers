"""Unit & round-trip tests for rse_looper.checkpoint."""

import json
import os
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rse_looper.checkpoint import (
    apply_weight_deltas,
    export_edited_checkpoint,
    is_editable_param,
    normalize_param_name,
    restore_weights,
    transformers_name_to_ckpt_deltas,
)


def test_normalize_and_allowlist_blocks_experts_vision_embeds():
    assert normalize_param_name("layers.6.self_attn.o_proj.linear.weight") == "model.layers.6.self_attn.o_proj.linear.weight"
    assert normalize_param_name("final_input.query") == "model.final_input.query"
    assert is_editable_param("model.layers.17.self_attn.o_proj.linear.weight")
    assert is_editable_param("layers.11.mlp.gate.e_score_correction_bias")
    assert is_editable_param("model.layers.6.mlp_message_gate.down_proj.weight")
    assert is_editable_param("model.message_input.query")
    assert is_editable_param("model.final_input.query")

    # Strictly forbidden tensors (quantized experts, vision tower, embeddings, lm_head, scales)
    for forbidden in [
        "model.layers.6.mlp.experts.0.down_proj.weight",
        "model.visual.backbone.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.6.mlp.experts.weight_scale",
        "model.layers.0.input_layernorm.weight",
    ]:
        assert not is_editable_param(forbidden)
        with pytest.raises(ValueError):
            transformers_name_to_ckpt_deltas(forbidden, torch.zeros(2, 2))


def test_inverse_mapping_offsets_and_transposes():
    # o_proj.linear.weight transposes into wo_ud[3072:6144]
    d_o = torch.randn(2304, 3072)
    [(ck, sl, d)] = transformers_name_to_ckpt_deltas("model.layers.17.self_attn.o_proj.linear.weight", d_o)
    assert ck == "model.llm.layers.17.attn.wo_ud.weight"
    assert sl == slice(3072, 6144)
    assert torch.allclose(d, d_o.T)

    # MessageGate uses offset lidx - 6
    d_mg = torch.randn(2304, 64)
    [(ck, sl, d)] = transformers_name_to_ckpt_deltas("model.layers.10.mlp_message_gate.message_gate_proj.weight", d_mg)
    assert ck == "model.llm.middle_cycle_cross_pass_gates.4.mlp_gate.up_proj.weight"
    assert sl == slice(2304, 4608)

    # MessageGate rejected outside middle loop [6..17]
    with pytest.raises(ValueError):
        transformers_name_to_ckpt_deltas("model.layers.3.mlp_message_gate.down_proj.weight", torch.zeros(64, 4608))

    # final_input.query maps to attn_res layer 24
    [(ck, sl, _)] = transformers_name_to_ckpt_deltas("model.final_input.query", torch.zeros(2304))
    assert ck == "model.llm.attn_res.layers.24.attention_input.query"
    assert sl is None


def test_export_edited_checkpoint_zero_drift_roundtrip(tmp_path):
    base_dir = tmp_path / "base"
    out_dir = tmp_path / "edited"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))

    # Build minimal synthetic safetensors containing BF16 editable + uint8 quantized expert + vision tensors
    orig_tensors = {
        "model.llm.layers.17.attn_sconv.conv.weight": torch.ones(16, 1, 4, dtype=torch.bfloat16),
        "model.llm.attn_res.layers.24.attention_input.query": torch.ones(16, dtype=torch.float32),
        "model.llm.layers.17.moe.experts.0.w1_nvfp4": torch.randint(0, 255, (32, 32), dtype=torch.uint8),
        "model.visual.patch_embed.weight": torch.randn(8, 8, dtype=torch.bfloat16),
    }
    save_file(orig_tensors, str(base_dir / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})

    deltas = {
        "model.layers.17.post_attention_convolution.conv1d.weight": torch.full((16, 1, 4), 0.05),
        "model.final_input.query": torch.full((16,), -0.02),
    }
    report = export_edited_checkpoint(str(base_dir), str(out_dir), deltas)
    assert len(report) == 2

    with safe_open(str(out_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as f:
        new_keys = set(f.keys())
        assert new_keys == set(orig_tensors.keys())
        # Untouched quantized expert & vision tensors MUST be bit-for-bit identical!
        assert torch.equal(f.get_tensor("model.llm.layers.17.moe.experts.0.w1_nvfp4"), orig_tensors["model.llm.layers.17.moe.experts.0.w1_nvfp4"])
        assert torch.equal(f.get_tensor("model.visual.patch_embed.weight"), orig_tensors["model.visual.patch_embed.weight"])
        # Edited tensors match expected values & preserve dtypes
        ed_q = f.get_tensor("model.llm.attn_res.layers.24.attention_input.query")
        assert ed_q.dtype == torch.float32
        assert torch.allclose(ed_q, torch.full((16,), 0.98), atol=1e-6)
        # Parameter count strictly identical
        orig_params = sum(t.numel() for t in orig_tensors.values())
        new_params = sum(f.get_tensor(k).numel() for k in new_keys)
        assert orig_params == new_params


def test_apply_and_restore_weights_exact():
    m = torch.nn.Linear(8, 4, bias=False)
    orig = m.weight.detach().clone()
    backups = apply_weight_deltas(m, {"weight": torch.full((4, 8), 0.1)})
    assert torch.allclose(m.weight, orig + 0.1, atol=1e-6)
    restore_weights(m, backups)
    assert torch.equal(m.weight, orig)


def test_given_invalid_second_delta_when_applying_then_no_parameter_changes():
    """Given one valid and one malformed edit, When applying, Then preflight all edits atomically."""
    model = torch.nn.Linear(2, 2)
    original = {k: t.detach().clone() for k, t in model.named_parameters()}
    with pytest.raises(ValueError, match="shape"):
        apply_weight_deltas(model, {"weight": torch.ones(2, 2), "bias": torch.ones(1)})
    for key, value in model.named_parameters():
        assert torch.equal(value, original[key])


def test_given_native_single_file_checkpoint_when_exporting_then_preserve_actual_tensors(tmp_path):
    """Given a native HF single-file checkpoint, When edited, Then verify saved keys, dtypes and unedited bytes."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    key = "model.layers.6.mlp.gate.weight"
    tensors = {key: torch.ones(2, 3), "model.layers.6.mlp.experts.weight": torch.arange(8, dtype=torch.uint8)}
    save_file(tensors, str(base / "model.safetensors"), metadata={"format": "pt"})
    report = export_edited_checkpoint(str(base), str(out), {key: torch.full((2, 3), 0.5)})
    with safe_open(str(out / "model.safetensors"), framework="pt") as saved:
        assert set(saved.keys()) == set(tensors)
        assert torch.equal(saved.get_tensor(key), torch.full((2, 3), 1.5))
        assert torch.equal(saved.get_tensor("model.layers.6.mlp.experts.weight"), tensors["model.layers.6.mlp.experts.weight"])
        assert saved.metadata() == {"format": "pt"}
    assert report[0]["ckpt_key"] == key
    audit = json.loads((out / "weight_integrity_audit.json").read_text())
    assert audit["tensor_keys"] == 2
    assert audit["changed_keys"] == [key]
    assert audit["unchanged_keys_verified"] == 1


def test_given_sharded_checkpoint_when_exporting_then_route_edits_and_preserve_index(tmp_path):
    """Given two shards, When a tensor in shard two changes, Then the other shard and index remain exact."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    router = "model.layers.6.mlp.gate.weight"
    expert = "model.layers.6.mlp.experts.weight"
    shard1, shard2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file({expert: torch.arange(8, dtype=torch.uint8)}, str(base / shard1))
    save_file({router: torch.ones(2, 3)}, str(base / shard2))
    index = {"metadata": {"total_size": 32}, "weight_map": {expert: shard1, router: shard2}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    export_edited_checkpoint(str(base), str(out), {router: torch.full((2, 3), 0.25)})
    assert (out / shard1).read_bytes() == (base / shard1).read_bytes()
    assert (out / "model.safetensors.index.json").read_bytes() == (base / "model.safetensors.index.json").read_bytes()
    with safe_open(str(out / shard2), framework="pt") as saved:
        assert torch.equal(saved.get_tensor(router), torch.full((2, 3), 1.25))


def test_given_nonfinite_export_delta_when_exporting_then_leave_no_output(tmp_path):
    """Given a nonfinite edit, When exporting, Then reject it before making a checkpoint."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    key = "model.llm.attn_res.layers.24.attention_input.query"
    save_file({key: torch.ones(2)}, str(base / "model-00001-of-00001.safetensors"))
    with pytest.raises(ValueError, match="finite"):
        export_edited_checkpoint(str(base), str(out), {"model.final_input.query": torch.tensor([float("nan"), 0.])})
    assert not out.exists()


def test_given_hub_id_when_exporting_then_resolve_snapshot_without_treating_id_as_directory(tmp_path, monkeypatch):
    """Given a Hub ID, When exporting, Then resolve it through the Hub client (mocked, no downloads)."""
    import sys
    from types import SimpleNamespace
    base, out = tmp_path / "snapshot", tmp_path / "out"
    base.mkdir()
    key = "model.final_input.query"
    save_file({key: torch.ones(2)}, str(base / "model.safetensors"))
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append(repo_id)
        return str(base)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    export_edited_checkpoint("owner/model", str(out), {key: torch.ones(2)})
    assert calls == ["owner/model"]
    assert (out / "model.safetensors").exists()


def test_given_stale_output_shards_when_exporting_then_reject_without_modifying_output(tmp_path):
    """Given an existing output index, When exporting, Then never leave a mix of old and new weights."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    key = "model.final_input.query"
    save_file({key: torch.ones(2)}, str(base / "model.safetensors"))
    old_index = '{"weight_map": {"old": "stale.safetensors"}}'
    (out / "model.safetensors.index.json").write_text(old_index)
    with pytest.raises(FileExistsError, match="empty"):
        export_edited_checkpoint(str(base), str(out), {key: torch.ones(2)})
    assert (out / "model.safetensors.index.json").read_text() == old_index
    assert not (out / "model.safetensors").exists()


def test_given_scalar_quantization_metadata_when_exporting_then_compare_its_exact_bytes(tmp_path):
    """Given scalar scale metadata beside an edit, When verifying saved tensors, Then preserve it exactly."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    key = "model.final_input.query"
    scale = "model.layers.6.mlp.experts.weight_scale"
    save_file({key: torch.ones(2), scale: torch.tensor(0.5)}, str(base / "model.safetensors"))
    export_edited_checkpoint(str(base), str(out), {key: torch.ones(2)})
    with safe_open(str(out / "model.safetensors"), framework="pt") as saved:
        assert saved.get_tensor(scale).shape == ()
        assert saved.get_tensor(scale).item() == 0.5


def test_given_standalone_chat_template_when_resolving_hub_then_preserve_prompt_definition(tmp_path, monkeypatch):
    """Given a Hub tokenizer's Jinja file, When filtering its snapshot, Then retain the actual chat template."""
    import fnmatch
    import sys
    from types import SimpleNamespace
    from rse_looper.checkpoint import resolve_checkpoint_dir

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    remote_files = {"config.json": "{}", "chat_template.jinja": "{{ messages[0]['content'] }}"}

    def snapshot_download(repo_id, *, allow_patterns, **kwargs):
        for name, content in remote_files.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                (snapshot / name).write_text(content)
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    resolved = resolve_checkpoint_dir("owner/model")
    assert (resolved / "chat_template.jinja").read_text() == remote_files["chat_template.jinja"]


@pytest.mark.parametrize("raw_key,shape", [
    *[(f"model.llm.layers.6.controllers.{kind}.{projection}", shape)
      for kind in ("attn", "k", "v", "mlp")
      for projection, shape in (("down_weight", (3, 8)), ("gain_weight", (4, 3)), ("forget_weight", (4, 3)))],
    *[(f"model.llm.layers.6.attn.{projection}.weight", (4, 3))
      for projection in ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn")],
    ("model.llm.layers.6.attn_sconv.weight", (4, 1, 3)),
    ("model.llm.layers.6.mlp_sconv.weight", (4, 1, 3)),
    ("model.llm.layers.6.mlp.gate.weight", (4, 3)),
    ("model.llm.layers.6.mlp.gate.bias", (4,)),
    ("model.llm.attn_res.attn_queries.5", (4,)),
    ("model.llm.attn_res.mlp_queries.6", (4,)),
    ("model.llm.attn_res.output_query", (4,)),
])
def test_given_upstream_original_format_when_exporting_then_invert_its_actual_renaming(tmp_path, raw_key, shape):
    """Given upstream's original-format key, When applying its native delta, Then preserve its layout exactly."""
    pytest.importorskip("transformers")
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    # Independent oracle: obtain the native name through upstream's own loader,
    # rather than duplicating our inverse-name mapping in the expected result.
    native = raw_key
    for transform in get_checkpoint_conversion_mapping("inkling2_loop_text"):
        if isinstance(transform, WeightRenaming):
            native, _ = transform.rename_source_key(native)
    assert is_editable_param(native), native
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    original = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    expert_key = "model.llm.layers.6.mlp.experts.w13_weight"
    experts = torch.arange(8, dtype=torch.uint8)
    save_file({raw_key: original, expert_key: experts}, str(base / "model.safetensors"))
    delta = torch.full(shape, 0.25)
    export_edited_checkpoint(str(base), str(out), {native: delta})
    with safe_open(str(out / "model.safetensors"), framework="pt") as saved:
        assert torch.equal(saved.get_tensor(raw_key), original + delta)
        assert torch.equal(saved.get_tensor(expert_key), experts)


def test_given_upstream_split_message_queries_when_exporting_then_roundtrip_through_official_merge(tmp_path):
    """Given split query/MLP and key/value tensors, When editing merged queries, Then reload the exact delta."""
    pytest.importorskip("transformers")
    from transformers.core_model_loading import Inkling2LoopMergeMessageQueries

    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    q_key = "model.llm.middle_cycle_cross_pass_queries"
    kv_key = "model.llm.middle_cycle_cross_pass_kv_queries"
    q = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kv = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)
    save_file({q_key: q, kv_key: kv}, str(base / "model.safetensors"))

    def upstream_merge(a, b):
        return Inkling2LoopMergeMessageQueries().convert(
            {q_key: a, kv_key: b}, [q_key, kv_key], ["message_input.query"],
        )["message_input.query"]

    original_native = upstream_merge(q, kv)
    delta = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 16
    export_edited_checkpoint(str(base), str(out), {"model.message_input.query": delta})
    with safe_open(str(out / "model.safetensors"), framework="pt") as saved:
        loaded = upstream_merge(saved.get_tensor(q_key), saved.get_tensor(kv_key))
    assert torch.equal(loaded, original_native + delta)
