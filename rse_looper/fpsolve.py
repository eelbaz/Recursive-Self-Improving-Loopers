"""Backprop-free activation matching with a soft preservation penalty (FP-Solve).

No optimizer states or added parameters. For each selected matrix, solve:
    min_{dW}  (1/N_e) ||X_edit dW^T - dY||_F^2
            + (lam_keep / N_k) ||X_keep dW^T||_F^2
            + lam_2 ||dW||_F^2
whose unconstrained Cholesky solution is:
    dW^T = (G_edit + lam_keep * G_keep + lam_2 * I)^{-1} (X_edit^T dY / N_e).

The optional norm cap rescales that solution. The preservation term is a soft
penalty, not an exact null-space projection or a proof of recurrent contraction.
The model pass fits pass-4 activations to recorded pass-3 targets layer by layer.
"""

from __future__ import annotations

from typing import Tuple
import math
import torch


@torch.no_grad()
def solve_nullspace_ridge(
    X_edit: torch.Tensor,
    dY: torch.Tensor,
    X_keep: torch.Tensor | None = None,
    lam_keep: float = 6.0,
    lam_ridge_frac: float = 0.08,
    max_rel: float = 0.18,
    W_orig: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float]:
    """Cholesky ridge solve with soft output preservation and optional norm capping.

    Args:
        X_edit: (N_e, D_in) activations where the edit should match target residual dY.
        dY: (N_e, D_out) desired output residual Y_target - Y_current.
        X_keep: optional (N_k, D_in) activations whose output changes are penalized.
        lam_keep: relative weight on preserving X_keep outputs.
        lam_ridge_frac: scale-invariant ridge regularization as a fraction of mean diagonal of G_edit.
        max_rel: relative Frobenius norm cap ||dW||_F / ||W_orig||_F when W_orig is provided.
        W_orig: optional original weight matrix (D_out, D_in) for norm capping.

    Returns:
        (dW, rel_change) where dW has shape (D_out, D_in).
    """
    for name, value in (("lam_keep", lam_keep), ("lam_ridge_frac", lam_ridge_frac), ("max_rel", max_rel)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    Xe = X_edit.detach().float()
    dYf = dY.detach().float()
    if Xe.ndim != 2 or dYf.ndim != 2 or Xe.shape[0] != dYf.shape[0]:
        raise ValueError(f"Incompatible shapes X_edit={tuple(Xe.shape)}, dY={tuple(dYf.shape)}")
    if not Xe.numel() or not dYf.numel():
        raise ValueError("Calibration arrays must not be empty")
    if Xe.device != dYf.device:
        raise ValueError("X_edit and dY must be on the same device")
    if not (torch.isfinite(Xe).all() and torch.isfinite(dYf).all()):
        raise ValueError("X_edit and dY must be finite")

    D_in = Xe.shape[1]
    n_e = max(1, Xe.shape[0])
    Ge = (Xe.T @ Xe) / float(n_e)
    tr_scale = float(torch.trace(Ge).item()) / max(1, D_in)
    lam2 = max(float(lam_ridge_frac) * tr_scale, 1e-4)

    A = Ge + lam2 * torch.eye(D_in, dtype=torch.float32, device=Xe.device)
    if X_keep is not None:
        Xk = X_keep.detach().float()
        if Xk.ndim != 2 or Xk.shape[1] != D_in:
            raise ValueError(f"X_keep shape {tuple(Xk.shape)} incompatible with D_in={D_in}")
        if Xk.device != Xe.device or not torch.isfinite(Xk).all():
            raise ValueError("X_keep must be finite and on the same device as X_edit")
        Gk = (Xk.T @ Xk) / float(max(1, Xk.shape[0]))
        A = A + float(lam_keep) * Gk

    B = (Xe.T @ dYf) / float(n_e)
    if W_orig is not None:
        if tuple(W_orig.shape) != (dYf.shape[1], D_in) or not torch.isfinite(W_orig).all():
            raise ValueError("W_orig must be finite with shape (D_out, D_in)")
    if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
        raise ValueError("Non-finite ridge system; rescale calibration activations")
    factor = torch.linalg.cholesky(A)
    dW = torch.cholesky_solve(B, factor).T  # (D_out, D_in)
    if not torch.isfinite(dW).all():
        raise ValueError("Non-finite closed-form FP-Solve solution")

    rel = 0.0
    if W_orig is not None:
        w_norm = float(W_orig.detach().float().norm().item()) + 1e-12
        rel = float(dW.norm().item()) / w_norm
        if rel > max_rel:
            dW = dW * (float(max_rel) / rel)
            rel = float(max_rel)
    return dW, rel


DEFAULT_FPSOLVE_SUBMODULES = (
    "q_message_gate.down_proj",
    "q_message_gate.forget_gate_proj",
    "q_message_gate.message_gate_proj",
    "k_message_gate.down_proj",
    "k_message_gate.forget_gate_proj",
    "k_message_gate.message_gate_proj",
    "v_message_gate.down_proj",
    "v_message_gate.forget_gate_proj",
    "v_message_gate.message_gate_proj",
    "mlp_message_gate.down_proj",
    "mlp_message_gate.forget_gate_proj",
    "mlp_message_gate.message_gate_proj",
    "self_attn.o_proj.linear",
    "mlp.gate",
)


@torch.no_grad()
def run_fpsolve_on_model(
    model: torch.nn.Module,
    cal_batches: list,
    lam_keep: float = 4.0,
    ridge_frac: float = 0.06,
    max_rel: float = 0.25,
    subnames: Tuple[str, ...] = DEFAULT_FPSOLVE_SUBMODULES,
    layer_range: Tuple[int, int] = (6, 18),
) -> dict:
    """Fit pass-4 outputs to pass-3 targets, restoring mode and rolling back failed fits."""
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward

    if not cal_batches:
        raise ValueError("At least one calibration batch is required")
    tm = model.model
    lo, hi = layer_range
    if not (tm.config.loop_layer_index_start <= lo < hi <= tm.config.loop_layer_index_end):
        raise ValueError("layer_range must lie within the model's looped layers")
    modules = [tm.layers[i].get_submodule(sn) for i in range(lo, hi) for sn in subnames]
    if not modules or any(not hasattr(m, "weight") or m.weight.ndim != 2 for m in modules):
        raise ValueError("FP-Solve requires at least one matrix-valued target module")
    original_weights = [(m.weight, m.weight.detach().clone()) for m in modules]
    original_cfg = getattr(tm, "_rse_loop_config", LoopConfig(num_passes=tm.config.loop_num_passes))
    try:
        install_midstack_loop_forward(model, LoopConfig(num_passes=4, rk_K=1))
        return _fit_fpsolve_on_model(model, cal_batches, lam_keep, ridge_frac, max_rel, subnames, layer_range)
    except Exception:
        for weight, original in original_weights:
            weight.copy_(original)
        raise
    finally:
        install_midstack_loop_forward(model, original_cfg)


def _fit_fpsolve_on_model(model, cal_batches, lam_keep, ridge_frac, max_rel, subnames, layer_range):
    device = next(model.parameters()).device

    clean_Y3: dict = {}
    for enc in cal_batches:
        hooks, call_cnt = [], {}
        for l_idx in range(layer_range[0], layer_range[1]):
            layer = model.model.layers[l_idx]
            for sn in subnames:
                mod = layer.get_submodule(sn)
                key = (l_idx, sn)
                call_cnt[key] = 0

                def make_h(k):
                    def hook(_m, _inp, out):
                        c = call_cnt[k]
                        call_cnt[k] += 1
                        p3_idx = 1 if "message_gate" in k[1] else 2
                        if c == p3_idx:
                            y = out[0] if isinstance(out, tuple) else out
                            clean_Y3.setdefault(k, []).append(y.detach().float().reshape(-1, y.shape[-1])[::2].cpu())
                    return hook

                hooks.append(mod.register_forward_hook(make_h(key)))
        try:
            _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        finally:
            for h in hooks:
                h.remove()

    rels = []
    for l_idx in range(layer_range[0], layer_range[1]):
        layer = model.model.layers[l_idx]
        for sn in subnames:
            mod = layer.get_submodule(sn)
            is_mg = "message_gate" in sn
            keep_idxs = {0, 1} if is_mg else {0, 1, 2}
            p4_idx = 2 if is_mg else 3
            xk_list, xe_list, dy_list = [], [], []
            for b_idx, enc in enumerate(cal_batches):
                cnt = [0]
                y3_b = clean_Y3[(l_idx, sn)][b_idx].to(device)

                def hook(_m, inp, out):
                    c = cnt[0]
                    cnt[0] += 1
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])[::2]
                    y_raw = out[0] if isinstance(out, tuple) else out
                    y = y_raw.detach().float().reshape(-1, y_raw.shape[-1])[::2]
                    if c in keep_idxs:
                        xk_list.append(x)
                    elif c == p4_idx:
                        xe_list.append(x)
                        dy_list.append(y3_b - y)

                hk = mod.register_forward_hook(hook)
                try:
                    _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
                finally:
                    hk.remove()

            dW, rel = solve_nullspace_ridge(
                torch.cat(xe_list, dim=0),
                torch.cat(dy_list, dim=0),
                X_keep=torch.cat(xk_list, dim=0),
                lam_keep=lam_keep if not is_mg else lam_keep * 0.7,
                lam_ridge_frac=ridge_frac,
                max_rel=max_rel,
                W_orig=mod.weight,
            )
            mod.weight.data.add_(dW.to(mod.weight.dtype))
            rels.append(rel)
    return {"n_matrices": len(rels), "mean_rel": sum(rels) / max(1, len(rels)), "max_rel": max(rels) if rels else 0.0}


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rse_looper.fpsolve", description="Run 0-backprop, 0-label closed-form FP-Solve.")
    ap.add_argument("--base-model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--model-revision", default=None, help="Immutable Hub model revision to resolve before loading.")
    ap.add_argument("--work-dir", default="./runs/fpsolve")
    ap.add_argument("--out-model", default="./models/Inkling-2-Mini-FPSolve")
    ap.add_argument("--prompts-jsonl", default=None, help="Optional JSONL with a 'prompt' field (48 unlabeled prompts).")
    ap.add_argument("--lam-keep", type=float, default=4.0)
    ap.add_argument("--ridge-frac", type=float, default=0.06)
    ap.add_argument("--max-rel", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    return ap


@torch.no_grad()
def main(argv=None) -> dict:
    import json, os, time
    from rse_looper.checkpoint import export_edited_checkpoint, resolve_checkpoint_dir
    from rse_looper.preflight import enable_fast_math

    args = build_parser().parse_args(argv)
    enable_fast_math()
    os.makedirs(args.work_dir, exist_ok=True)

    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    base_snapshot = resolve_checkpoint_dir(args.base_model, revision=args.model_revision)
    tok = AutoTokenizer.from_pretrained(base_snapshot)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = Inkling2LoopForCausalLM.from_pretrained(
        base_snapshot, dtype=torch.bfloat16, device_map={"": args.device}, experts_implementation="grouped_mm"
    ).eval()

    orig_cpu = {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith("model.layers.") and ("message_gate." in n or "self_attn.o_proj.linear" in n or "mlp.gate.weight" in n)
    }
    if args.prompts_jsonl:
        with open(args.prompts_jsonl) as stream:
            prompts = [json.loads(l)["prompt"] for l in stream if l.strip()][:48]
    else:
        prompts = [f"Explain step by step how to solve equation {k}x + {k+3} = {4*k+7}." for k in range(1, 49)]
    cal_batches = [tok(prompts[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(args.device) for i in range(0, len(prompts), 8)]

    t0 = time.time()
    info = run_fpsolve_on_model(model, cal_batches, lam_keep=args.lam_keep, ridge_frac=args.ridge_frac, max_rel=args.max_rel)
    info["elapsed_sec"] = round(time.time() - t0, 1)
    info["base_model_snapshot"] = str(base_snapshot)
    info["requested_model_revision"] = args.model_revision

    deltas = {}
    for n, p in model.named_parameters():
        if n in orig_cpu:
            d = p.detach().float().cpu() - orig_cpu[n]
            if d.abs().max().item() > 1e-7:
                deltas[n] = d
    export_edited_checkpoint(str(base_snapshot), args.out_model, deltas)
    with open(os.path.join(args.work_dir, "fpsolve_run.json"), "w") as stream:
        json.dump(info, stream, indent=2)
    return info


if __name__ == "__main__":
    main()
