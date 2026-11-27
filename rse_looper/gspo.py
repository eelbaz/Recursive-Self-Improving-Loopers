"""Autonomous Multi-Generation Loop-Advantage GSPO + DAPO (rse_looper.gspo).

Implements:
  - GSPO-style sequence-level geometric-mean importance ratios.
  - DAPO asymmetric Clip-Higher (eps_low=0.20, eps_high=0.28) + dynamic frontier filtering.
  - Dr. GRPO unbiased group advantages (A_i = r_i - mean(r_G)).
  - Loop-Advantage bonus (+delta_r_bonus) when 4-pass or 2-step damped solves a problem 3-pass missed.
  - A fixed 3-mode selection gate, explicit labeled inputs, and durable run provenance.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import torch
import torch.nn.functional as F


def compute_loop_advantage_gspo_groups(
    rollouts_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    delta_r_bonus: float = 0.35,
    rep_penalty: float = 0.50,
    trunc_penalty: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute dynamic-sampled, unbiased Loop-Advantage groups from 3-mode rollouts."""
    active_groups: List[Dict[str, Any]] = []
    n_total = len(rollouts_by_id)
    n_any_solved = 0
    n_all_solved = 0
    n_deep_rescue = 0
    total_rollouts = 0
    total_correct = 0

    for pid, recs in sorted(rollouts_by_id.items()):
        if not recs:
            continue
        total_rollouts += len(recs)
        n_ok = sum(1 for r in recs if r.get("correct") and r.get("completed") and not r.get("repetition"))
        total_correct += n_ok
        if n_ok > 0:
            n_any_solved += 1
        if n_ok == len(recs):
            n_all_solved += 1
        if n_ok == 0 or n_ok == len(recs):
            continue

        r3_recs = [r for r in recs if r.get("regime") == "R3"]
        r3_solved = any(r.get("correct") and r.get("completed") and not r.get("repetition") for r in r3_recs)

        rewards: List[float] = []
        group_had_rescue = False
        for r in recs:
            is_win = bool(r.get("correct") and r.get("completed") and not r.get("repetition"))
            rew = 1.0 if is_win else 0.0
            if r.get("repetition"):
                rew -= float(rep_penalty)
            elif not r.get("completed"):
                rew -= float(trunc_penalty)
            if is_win and (not r3_solved) and r.get("regime") in ("R4", "RK2"):
                rew += float(delta_r_bonus)
                group_had_rescue = True
            rewards.append(rew)

        if group_had_rescue:
            n_deep_rescue += 1

        mean_r = sum(rewards) / len(rewards)
        items = []
        for r, rew in zip(recs, rewards):
            items.append({
                "id": pid,
                "problem": r["problem"],
                "raw_full": r["raw_full"],
                "regime": r.get("regime", "R4"),
                "reward": round(rew, 4),
                "advantage": round(rew - mean_r, 4),
                "old_avg_logprob": float(r["old_avg_logprob"]),
                "correct": bool(r.get("correct")),
                "completed": bool(r.get("completed")),
                "gen_tokens": int(r.get("gen_tokens", 0)),
                **{key: r[key] for key in ("prompt_token_ids", "completion_token_ids") if key in r},
            })
        active_groups.append({"id": pid, "items": items})

    stats = {
        "n_problems": n_total,
        "best_of_k_solved": n_any_solved,
        "best_of_k_rate": round(n_any_solved / max(1, n_total) * 100.0, 2),
        "all_solved": n_all_solved,
        "active_frontier_groups": len(active_groups),
        "deep_loop_rescue_groups": n_deep_rescue,
        "pass1_avg_rate": round(total_correct / max(1, total_rollouts) * 100.0, 2),
    }
    return active_groups, stats


def gspo_dapo_sequence_loss(
    cur_avg_logprob: torch.Tensor,
    old_avg_logprob: torch.Tensor | float,
    advantage: torch.Tensor | float,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """GSPO sequence-level importance ratio loss with asymmetric Clip-Higher."""
    old_lp = torch.as_tensor(old_avg_logprob, dtype=torch.float32, device=cur_avg_logprob.device)
    adv = torch.as_tensor(advantage, dtype=torch.float32, device=cur_avg_logprob.device)
    log_ratio = torch.clamp(cur_avg_logprob.float() - old_lp, min=-4.0, max=4.0)
    seq_ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(seq_ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))
    loss = -torch.minimum(seq_ratio * adv, clipped_ratio * adv).mean()
    info = {
        "seq_ratio": float(seq_ratio.detach().mean().item()),
        "clipped": bool((seq_ratio.detach() < (1.0 - eps_low)).any() or (seq_ratio.detach() > (1.0 + eps_high)).any()),
    }
    return loss, info


def rsi_step_decision(
    cur_gate: Mapping[str, Any],
    cand_gate: Mapping[str, Any],
    max_single_regime_drop: int = 2,
) -> Dict[str, Any]:
    """Return the shared acceptance rule's decision and stable event-log reason."""
    from rse_looper.run_state import assess_gate_candidate
    return assess_gate_candidate(cur_gate, cand_gate, max_single_regime_drop)


def accept_3regime_rsi_step(
    cur_gate: Mapping[str, Any], cand_gate: Mapping[str, Any], max_single_regime_drop: int = 2,
) -> bool:
    """Compatibility wrapper around the logged acceptance decision."""
    return rsi_step_decision(cur_gate, cand_gate, max_single_regime_drop)["accepted"]


def partition_gate_rows(rows: Sequence[dict], local_idx: int, worker_count: int) -> List[dict]:
    """Every GPU evaluates its shard in all three regimes, including a single GPU."""
    if worker_count < 1 or not 0 <= local_idx < worker_count:
        raise ValueError("Invalid gate worker index or count.")
    return list(rows[local_idx::worker_count])


def rollout_sequence(row: Mapping[str, Any]) -> Tuple[torch.Tensor, int]:
    """Use exact sampled token IDs; never re-tokenize or silently truncate a solution."""
    prompt = row.get("prompt_token_ids")
    completion = row.get("completion_token_ids")
    if not isinstance(prompt, list) or not prompt or not isinstance(completion, list) or not completion:
        raise ValueError("Rollout requires nonempty prompt_token_ids and completion_token_ids.")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in prompt + completion):
        raise ValueError("Rollout token IDs must be nonnegative integers.")
    return torch.tensor([prompt + completion], dtype=torch.long), len(prompt)


class EmptyFrontierError(ValueError):
    """A valid rollout batch with no mixed reward groups, rather than a run failure."""

    def __init__(self, stats: dict):
        super().__init__("No active frontier: all sampled problem groups were uniformly correct or incorrect; no update was run.")
        self.stats = stats


def prepare_gspo_training_rows(rollouts_by_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[List[dict], dict]:
    groups, stats = compute_loop_advantage_gspo_groups(rollouts_by_id)
    if not groups:
        raise EmptyFrontierError(stats)
    return [item for group in groups for item in group["items"]], stats


def clip_generation_deltas(
    deltas: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, torch.Tensor],
    max_rel: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Enforce per-generation relative Frobenius norm cap."""
    clipped: Dict[str, torch.Tensor] = {}
    for k, d in deltas.items():
        df = d.detach().float().cpu()
        if not torch.isfinite(df).all():
            raise ValueError(f"Non-finite delta on {k}")
        w = base_weights[k].detach().float().cpu()
        rel = df.norm().item() / (w.norm().item() + 1e-12)
        if rel > max_rel:
            df = df * (max_rel / rel)
        clipped[k] = df
    return clipped


def avg_sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mean per-token log-probability over completion tokens."""
    if input_ids.ndim != 2 or logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Logits and token sequence shapes must agree.")
    if not isinstance(prompt_len, int) or not 1 <= prompt_len < input_ids.shape[1]:
        raise ValueError("prompt_len must leave at least one completion token.")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    st = prompt_len - 1
    logp = F.log_softmax(shift_logits[:, st:, :], dim=-1)
    tok_lp = logp.gather(-1, shift_labels[:, st:].unsqueeze(-1)).squeeze(-1)
    return tok_lp.mean()


def _format_math_prompt(tok: Any, problem: str, effort: str = "high") -> str:
    msgs = [{"role": "user", "content": f"Solve the following problem and put your final answer in \\boxed{{}}.\n\n{problem}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def configure_generation(model: Any, *, max_new_tokens: int, pad_token_id: int,
                         eos_token_id: int, do_sample: bool = True):
    """Use an explicit unfiltered policy matching the raw likelihood calculation.

Reset inherited generation controls: a model's repetition penalties, forced
tokens or truncation filters would otherwise change the behavior distribution.
"""
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig()
    return GenerationConfig(
        do_sample=do_sample, num_beams=1, num_return_sequences=1,
        temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0,
        repetition_penalty=1.0, encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
        min_length=0, min_new_tokens=0, use_cache=True,
        max_new_tokens=max_new_tokens, pad_token_id=pad_token_id, eos_token_id=eos_token_id,
        return_dict_in_generate=False, output_scores=False,
    )


def _search_worker(local_idx: int, gpu_ids: List[int], model_path: str, rows: List[dict], out_dir: str, gen_idx: int, max_new_tokens: int) -> None:
    import json
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.grader import grade_math_output
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    gid = gpu_ids[local_idx]
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    ).eval()
    generation_config = configure_generation(model, max_new_tokens=max_new_tokens,
                                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    my_rows = [r for i, r in enumerate(rows) if i % len(gpu_ids) == local_idx]
    reg_specs = [
        ("R3", LoopConfig(num_passes=3, rk_K=1), 2),
        ("R4", LoopConfig(num_passes=4, rk_K=1), 2),
        ("RK2", LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3), 2),
    ]
    records = []
    for reg_name, cfg, n_samp in reg_specs:
        install_midstack_loop_forward(model, cfg)
        expanded = [(r, s_idx) for r in my_rows for s_idx in range(n_samp)]
        for start in range(0, len(expanded), 32):
            chunk = expanded[start : start + 32]
            texts = [_format_math_prompt(tok, r["problem"]) for r, _ in chunk]
            inputs = tok(texts, return_tensors="pt", padding=True).to(device)
            in_len = inputs["input_ids"].shape[1]
            torch.manual_seed(20000 * gen_idx + 100 * local_idx + start)
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs, generation_config=generation_config,
                )
            for j, (r, s_idx) in enumerate(chunk):
                g_ids = gen_out[j, in_len:]
                ep = (g_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
                if len(ep) > 0:
                    g_ids = g_ids[: ep[0].item() + 1]
                raw = tok.decode(g_ids, skip_special_tokens=False)
                g = grade_math_output(raw, r["answer"])
                prompt_ids = inputs["input_ids"][j][inputs["attention_mask"][j].bool()].tolist()
                completion_ids = g_ids.tolist()
                full_ids, p_len = rollout_sequence({"prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids})
                full_ids = full_ids.to(device)
                with torch.no_grad():
                    logits = model(input_ids=full_ids, use_cache=False).logits
                    old_lp = float(avg_sequence_logprob(logits, full_ids, p_len).item())
                records.append({
                    "id": r["id"], "problem": r["problem"], "answer": r["answer"], "regime": reg_name,
                    "sample_idx": s_idx, "gen_tokens": int(len(g_ids)), "correct": bool(g["correct"]),
                    "completed": bool(g["completed"]), "repetition": bool(g["repetition"]),
                    "old_avg_logprob": old_lp, "raw_full": raw,
                    "prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids,
                })
    with open(os.path.join(out_dir, f"search.shard{local_idx}.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _ddp_update_worker(
    local_rank: int, world_size: int, gpu_ids: List[int], model_path: str,
    gspo_path: str, out_delta_path: str, steps: int, lr: float, seed: int,
) -> None:
    import json
    import torch.distributed as dist
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.checkpoint import is_editable_param
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    gid = gpu_ids[local_rank]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29710 + (seed % 500))
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    )
    for p in model.parameters():
        p.requires_grad_(False)
    trainable, orig_w = [], {}
    for name, p in model.named_parameters():
        if is_editable_param(name):
            p.requires_grad_(True)
            trainable.append((name, p))
            orig_w[name] = p.detach().float().cpu().clone()

    with open(gspo_path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Empty frontier training data; update was not started.")
    prepped = []
    for r in rows:
        ids, p_len = rollout_sequence(r)
        prepped.append((ids, p_len, r["regime"], float(r["advantage"]), float(r["old_avg_logprob"])))

    my_idxs = [i for i in range(len(prepped)) if i % world_size == local_rank] or [0]
    opt = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = random.Random(seed + 41 * local_rank)

    cfgs = {
        "R3": LoopConfig(num_passes=3, rk_K=1),
        "R4": LoopConfig(num_passes=4, rk_K=1),
        "RK2": LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3),
    }

    # Keep deterministic evaluation behavior while autograd is enabled. Changing
    # dropout between the old and current policy would confound the likelihood ratio.
    model.eval()
    for step in range(1, steps + 1):
        prog = max(0.0, (step - 3) / max(1, steps - 3))
        cur_lr = lr * (step / 3.0) if step <= 3 else lr * (0.20 + 0.80 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            ids, p_len, reg, adv, old_lp = prepped[rng.choice(my_idxs)]
            install_midstack_loop_forward(model, cfgs[reg])
            ids_d = ids.to(device)
            logits = model(input_ids=ids_d, use_cache=False).logits
            cur_lp = avg_sequence_logprob(logits, ids_d, p_len)
            loss_rl, _ = gspo_dapo_sequence_loss(cur_lp, old_lp, adv, eps_low=0.20, eps_high=0.28)
            (loss_rl * 0.5).backward()
            del logits, cur_lp, loss_rl

        for _, p in trainable:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()

    if local_rank == 0:
        raw_deltas = {name: p.detach().float().cpu() - orig_w[name] for name, p in trainable}
        clipped = clip_generation_deltas(raw_deltas, orig_w, max_rel=0.05)
        torch.save(clipped, out_delta_path)

    dist.barrier()
    dist.destroy_process_group()


def _gate_worker(local_idx: int, gpu_ids: List[int], model_path: str, delta_path: str | None, scale: float, gate_rows: List[dict], out_dir: str, tag: str, max_new_tokens: int = 1536) -> None:
    import json
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM
    from rse_looper.checkpoint import apply_weight_deltas
    from rse_looper.grader import grade_math_output
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    from rse_looper.preflight import enable_fast_math

    regimes = [("R3", LoopConfig(num_passes=3, rk_K=1)), ("R4", LoopConfig(num_passes=4, rk_K=1)), ("RK2", LoopConfig(num_passes=3, rk_window=(10, 13), rk_K=2, rk_beta=0.3))]
    my_rows = partition_gate_rows(gate_rows, local_idx, len(gpu_ids))
    if not my_rows:
        raise ValueError("Gate worker received an empty shard.")
    gid = gpu_ids[local_idx]
    torch.cuda.set_device(gid)
    enable_fast_math()
    device = f"cuda:{gid}"

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Inkling2LoopForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device}, experts_implementation="grouped_mm"
    ).eval()
    if delta_path:
        d = torch.load(delta_path, map_location="cpu", weights_only=True)
        if abs(scale - 1.0) > 1e-6:
            d = {k: v * float(scale) for k, v in d.items()}
        apply_weight_deltas(model, d)

    generation_config = configure_generation(model, max_new_tokens=max_new_tokens,
                                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                             do_sample=False)

    for reg_name, cfg in regimes:
        install_midstack_loop_forward(model, cfg)
        texts = [_format_math_prompt(tok, r["problem"]) for r in my_rows]
        inputs = tok(texts, return_tensors="pt", padding=True).to(device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=generation_config)
        res = []
        for j, r in enumerate(my_rows):
            g_ids = out[j, in_len:]
            ep = (g_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if len(ep) > 0:
                g_ids = g_ids[: ep[0].item() + 1]
            raw = tok.decode(g_ids, skip_special_tokens=False)
            g = grade_math_output(raw, r["answer"])
            res.append({"id": r["id"], "gen_tokens": int(len(g_ids)), "raw_full": raw, **g})
        with open(os.path.join(out_dir, f"{tag}_{reg_name}.sh{local_idx}.jsonl"), "w") as f:
            for item in res:
                f.write(json.dumps(item) + "\n")


def evaluate_gate(gpu_ids: List[int], model_path: str, delta_path: str | None, scale: float,
                  gate_rows: List[dict], out_dir: str, tag: str, max_new_tokens: int = 1536) -> dict:
    """Run all three regimes for every row, independently of the number of GPUs."""
    import json
    import torch.multiprocessing as mp
    from rse_looper.grader import validate_binary_rows

    if not gpu_ids or not gate_rows:
        raise ValueError("Gate evaluation requires GPUs and nonempty selection data.")
    os.makedirs(out_dir, exist_ok=True)
    n_workers = min(len(gpu_ids), len(gate_rows))
    mp.spawn(_gate_worker, args=(gpu_ids[:n_workers], model_path, delta_path, scale, gate_rows,
                                out_dir, tag, max_new_tokens), nprocs=n_workers, join=True)
    out: dict = {}
    tot_ok, tot_comp, max_rep = 0, 0, 0.0
    expected_ids = sorted(row["id"] for row in gate_rows)
    for reg in ("R3", "R4", "RK2"):
        rows = []
        for shard in range(n_workers):
            with open(os.path.join(out_dir, f"{tag}_{reg}.sh{shard}.jsonl")) as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        validated = validate_binary_rows(rows, name=f"selection/{reg}")
        if sorted(validated) != expected_ids:
            raise ValueError(f"Gate output coverage mismatch for {reg}; missing or duplicate problem IDs.")
        n = len(rows)
        if any("completed" not in row or "repetition" not in row for row in rows):
            raise ValueError("Gate rows require completed and repetition JSON booleans.")
        ok = sum(row["correct"] for row in rows)
        comp = sum(row["completed"] for row in rows)
        rep = round(sum(row["repetition"] for row in rows) / n * 100.0, 2)
        out[reg] = {"n": n, "correct": ok, "completed": comp, "acc": round(ok / n * 100.0, 2), "rep": rep}
        tot_ok += ok
        tot_comp += comp
        max_rep = max(max_rep, rep)
    out.update(total_correct=tot_ok, total_completed=tot_comp,
               mean_acc=round(tot_ok / (3 * len(gate_rows)) * 100.0, 2), max_rep=max_rep)
    return out


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rse_looper.gspo", description="Run autonomous multi-generation Loop-Advantage GSPO + DAPO.")
    ap.add_argument("--init-model", default="./models/Inkling-2-Mini-FPSolve")
    ap.add_argument("--train-jsonl", help="Required labeled training JSONL (id, problem, answer).")
    ap.add_argument("--gate-jsonl", help="Required disjoint selection JSONL; this is not a final test set.")
    ap.add_argument("--work-dir", default="./runs/gspo")
    ap.add_argument("--ckpt-prefix", default="./models/Inkblot-2-mini")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--max-gens", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--problems-per-gen", type=int, default=96)
    ap.add_argument("--gate-size", type=int, default=64)
    ap.add_argument("--steps-per-gen", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--gate-max-new-tokens", type=int, default=1536)
    ap.add_argument("--model-revision", help="Pin a Hub model revision; local checkpoints are hashed directly.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> dict:
    import json
    from pathlib import Path
    import tempfile
    import torch.multiprocessing as mp
    from rse_looper.preflight import enable_fast_math, preflight_gpu_check
    from rse_looper.run_state import RunLedger, checkpoint_fingerprint, load_math_inputs

    args = build_parser().parse_args(argv)
    gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("GPU IDs must be a nonempty list of distinct nonnegative integers.")
    for name in ("max_gens", "patience", "problems_per_gen", "gate_size", "steps_per_gen", "max_new_tokens", "gate_max_new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")

    # Validate input identity and split separation before GPU setup or model downloads.
    inputs = None
    if args.train_jsonl or args.gate_jsonl or not args.dry_run:
        inputs = load_math_inputs(args.train_jsonl, args.gate_jsonl, gate_size=args.gate_size)
    enable_fast_math()
    preflight = preflight_gpu_check(require_causal_conv1d=False)
    summary = {
        "init_model": args.init_model, "work_dir": args.work_dir,
        "ckpt_prefix": args.ckpt_prefix, "gpus": gpu_ids, "max_gens": args.max_gens,
        "preflight": preflight, "inputs_validated": inputs is not None,
        "selection_is_test": False, "output_policy": "last_accepted_checkpoint",
    }
    if args.dry_run:
        summary["dry_run"] = True
        summary["model_resolved"] = False
        if inputs is None:
            summary["required_inputs"] = ["--train-jsonl", "--gate-jsonl"]
        return summary
    if not preflight["cuda_available"]:
        raise RuntimeError("Loop-Advantage GSPO requires CUDA GPUs; validated inputs were not trained.")

    from rse_looper.checkpoint import export_edited_checkpoint, resolve_checkpoint_dir
    initial_path = resolve_checkpoint_dir(args.init_model, revision=args.model_revision)
    initial = checkpoint_fingerprint(initial_path)
    run_config = {key: value for key, value in vars(args).items() if key not in {"resume", "dry_run", "work_dir"}}
    run_config.update(acceptance_rule="three_regime_v1", candidate_rank=["total_correct", "total_completed"],
                      output_policy="last_accepted_checkpoint", likelihood_policy="complete_sampled_token_ids",
                      sampling_policy={"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                       "inherited_generation_controls": "reset", "repetition_penalty": 1.0})

    with RunLedger(args.work_dir, config=run_config, data_fingerprints=inputs.fingerprints,
                   initial_checkpoint=initial, resume=args.resume) as run:
        if run.state["current_gate"] is None:
            initial_gate_root = Path(args.work_dir) / "gate_init"
            initial_gate_root.mkdir(parents=True, exist_ok=True)
            initial_gate_dir = Path(tempfile.mkdtemp(prefix="attempt_", dir=initial_gate_root))
            gate = evaluate_gate(gpu_ids, str(initial_path), None, 1.0, inputs.gate_rows,
                                 str(initial_gate_dir), "init", args.gate_max_new_tokens)
            for artifact in sorted(initial_gate_dir.glob("init_*.jsonl")):
                run.record_artifact("initial_gate_shard", artifact)
            run.initialize_gate(gate)
        if run.state["status"] != "completed":
            for generation in range(run.state["next_generation"], args.max_gens):
                if run.state["consecutive_rejections"] >= args.patience:
                    break
                gdir = run.begin_generation()
                cur_model = run.state["current_checkpoint"]["path"]
                cur_gate = run.state["current_gate"]
                count = min(args.problems_per_gen, len(inputs.train_rows))
                start = (generation * count) % len(inputs.train_rows)
                batch_rows = [inputs.train_rows[(start + index) % len(inputs.train_rows)] for index in range(count)]
                run.record_training_batch([row["id"] for row in batch_rows])
                search_gpus = gpu_ids[:min(len(gpu_ids), len(batch_rows))]
                mp.spawn(_search_worker, args=(search_gpus, cur_model, batch_rows, str(gdir), generation,
                                               args.max_new_tokens), nprocs=len(search_gpus), join=True)
                by_id: dict = {}
                for shard in range(len(search_gpus)):
                    path = gdir / f"search.shard{shard}.jsonl"
                    run.record_artifact("rollouts", path)
                    with path.open() as stream:
                        for line in stream:
                            if line.strip():
                                record = json.loads(line)
                                by_id.setdefault(record["id"], []).append(record)
                try:
                    training_rows, search_stats = prepare_gspo_training_rows(by_id)
                except EmptyFrontierError as exc:
                    run.record_no_update("no_frontier", stats=exc.stats)
                    run.commit_generation()
                    continue
                gspo_path = gdir / "gspo_items.jsonl"
                with gspo_path.open("w") as stream:
                    for row in training_rows:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                run.record_artifact("training_items", gspo_path, stats=search_stats)
                delta_path = gdir / "update_delta.pt"
                update_gpus = gpu_ids[:min(len(gpu_ids), len(training_rows))]
                mp.spawn(_ddp_update_worker,
                         args=(len(update_gpus), update_gpus, cur_model, str(gspo_path), str(delta_path),
                               args.steps_per_gen, args.lr * (0.9 ** run.state["accepted_generations"]), 3000 + generation),
                         nprocs=len(update_gpus), join=True)
                run.record_artifact("update_delta", delta_path)

                winners = []
                for scale in (1.0, 0.6):
                    tag = f"g{generation}_s{int(scale * 10)}"
                    candidate = evaluate_gate(gpu_ids, cur_model, str(delta_path), scale, inputs.gate_rows,
                                              str(gdir / "gate_eval"), tag,
                                              args.gate_max_new_tokens)
                    for artifact in sorted((gdir / "gate_eval").glob(f"{tag}_*.jsonl")):
                        run.record_artifact("candidate_gate_shard", artifact, scale=scale)
                    decision = rsi_step_decision(cur_gate, candidate)
                    run.record_candidate(scale, candidate, **decision)
                    if decision["accepted"]:
                        winners.append((scale, candidate))
                if winners:
                    scale, _ = max(winners, key=lambda item: (item[1]["total_correct"], item[1]["total_completed"]))
                    next_checkpoint = Path(f"{args.ckpt_prefix}-gen{run.state['accepted_generations'] + 1}-attempt{run.state['attempt_count']:04d}")
                    if next_checkpoint.exists():
                        raise FileExistsError(f"Refusing to overwrite checkpoint: {next_checkpoint}")
                    delta = torch.load(delta_path, map_location="cpu", weights_only=True)
                    if scale != 1.0:
                        delta = {name: value * scale for name, value in delta.items()}
                    export_edited_checkpoint(cur_model, str(next_checkpoint), delta)
                    run.commit_generation(checkpoint=next_checkpoint, selected_scale=scale, delta_path=delta_path)
                else:
                    run.commit_generation(delta_path=delta_path)
            run.complete()
        summary.update(status=run.state["status"], accepted_gens=run.state["accepted_generations"],
                       final_model=run.state["current_checkpoint"]["path"], final_gate=run.state["current_gate"],
                       manifest=str(run.manifest_path), event_log=str(run.event_path))
    return summary


if __name__ == "__main__":
    main()
