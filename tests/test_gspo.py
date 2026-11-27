"""Unit tests for rse_looper.gspo (Loop-Advantage GSPO, DAPO Clip-Higher, Dr. GRPO, 3-regime gate)."""

import math
import pytest
import torch

from rse_looper.gspo import (
    accept_3regime_rsi_step,
    clip_generation_deltas,
    compute_loop_advantage_gspo_groups,
    gspo_dapo_sequence_loss,
)


def test_compute_loop_advantage_gspo_dynamic_sampling_and_deep_rescue_bonus():
    rollouts = {
        # Frontier rescue problem: R3 fails, R4 & RK2 solve -> triggers Loop-Advantage bonus (+0.35)!
        "rescue_p1": [
            {"problem": "p1", "raw_full": "r3_fail", "regime": "R3", "gen_tokens": 900, "correct": False, "completed": True, "repetition": False, "old_avg_logprob": -0.30},
            {"problem": "p1", "raw_full": "r3_rep",  "regime": "R3", "gen_tokens": 1600, "correct": False, "completed": False, "repetition": True, "old_avg_logprob": -0.40},
            {"problem": "p1", "raw_full": "r4_win",  "regime": "R4", "gen_tokens": 500, "correct": True,  "completed": True, "repetition": False, "old_avg_logprob": -0.22},
            {"problem": "p1", "raw_full": "rk2_win", "regime": "RK2","gen_tokens": 520, "correct": True,  "completed": True, "repetition": False, "old_avg_logprob": -0.25},
        ],
        # Saturated easy problem (all solved): DAPO Dynamic Sampling MUST drop it!
        "easy_p2": [
            {"problem": "p2", "raw_full": "ok", "regime": "R3", "gen_tokens": 200, "correct": True, "completed": True, "repetition": False, "old_avg_logprob": -0.10}
            for _ in range(4)
        ],
        # Unsolved impossible problem (0 solved): DAPO Dynamic Sampling MUST drop it!
        "impossible_p3": [
            {"problem": "p3", "raw_full": "no", "regime": "R4", "gen_tokens": 1600, "correct": False, "completed": False, "repetition": False, "old_avg_logprob": -0.50}
            for _ in range(4)
        ],
    }
    groups, stats = compute_loop_advantage_gspo_groups(rollouts, delta_r_bonus=0.35, rep_penalty=0.50)
    assert stats["active_frontier_groups"] == 1
    assert stats["deep_loop_rescue_groups"] == 1
    items = groups[0]["items"]
    # Dr. GRPO zero-mean unbiased advantage across group
    assert abs(sum(it["advantage"] for it in items)) < 1e-3
    by_raw = {it["raw_full"]: it for it in items}
    assert math.isclose(by_raw["r4_win"]["reward"], 1.35, abs_tol=1e-4)
    assert math.isclose(by_raw["rk2_win"]["reward"], 1.35, abs_tol=1e-4)
    assert math.isclose(by_raw["r3_rep"]["reward"], -0.50, abs_tol=1e-4)
    assert by_raw["r4_win"]["advantage"] > 0 and by_raw["r3_fail"]["advantage"] < 0


def test_gspo_dapo_sequence_loss_clip_higher_asymmetry_and_gradients():
    # Positive advantage with ratio > 1 + eps_high (1.28) must clip gradient cleanly!
    lp_old = -0.50
    lp_cur = torch.tensor(lp_old + math.log(1.50), requires_grad=True)  # ratio = 1.50 > 1.28
    loss_hi, info_hi = gspo_dapo_sequence_loss(lp_cur, lp_old, advantage=+1.0, eps_low=0.20, eps_high=0.28)
    assert info_hi["clipped"] is True
    assert math.isclose(loss_hi.item(), -1.28, rel_tol=1e-4)
    loss_hi.backward()
    assert lp_cur.grad.item() == 0.0  # clipped flat above 1.28!

    # Within Clip-Higher band [1.20, 1.28]: still active gradient (whereas symmetric PPO 0.20 would have clipped!)
    lp_mid = torch.tensor(lp_old + math.log(1.24), requires_grad=True)
    loss_mid, info_mid = gspo_dapo_sequence_loss(lp_mid, lp_old, advantage=+1.0, eps_low=0.20, eps_high=0.28)
    assert info_mid["clipped"] is False
    loss_mid.backward()
    assert lp_mid.grad.item() < 0.0  # increases cur_avg_logprob!

    # Negative advantage pushes cur_avg_logprob DOWN (positive gradient on cur_avg_logprob)
    lp_neg = torch.tensor(lp_old, requires_grad=True)
    loss_neg, _ = gspo_dapo_sequence_loss(lp_neg, lp_old, advantage=-1.0, eps_low=0.20, eps_high=0.28)
    loss_neg.backward()
    assert lp_neg.grad.item() > 0.0


def test_accept_3regime_rsi_step_enforces_compounding_and_no_collapse():
    base = {
        "R3": {"correct": 40, "completed": 46},
        "R4": {"correct": 37, "completed": 43},
        "RK2": {"correct": 36, "completed": 40},
        "total_correct": 113,
        "total_completed": 129,
        "max_rep": 0.0,
    }
    better = {
        "R3": {"correct": 42, "completed": 47},
        "R4": {"correct": 40, "completed": 45},
        "RK2": {"correct": 36, "completed": 42},
        "total_correct": 118,
        "total_completed": 134,
        "max_rep": 1.56,
    }
    assert accept_3regime_rsi_step(base, better)
    rk_collapse = {**better, "R4": {"correct": 44, "completed": 48}, "RK2": {"correct": 33, "completed": 39}, "total_correct": 119}
    assert not accept_3regime_rsi_step(base, rk_collapse)
    assert not accept_3regime_rsi_step(base, {**better, "max_rep": 4.69})


def test_clip_generation_deltas_caps_norm_and_rejects_nan():
    base = {"w": torch.ones(10, 10)}
    clipped = clip_generation_deltas({"w": torch.full((10, 10), 0.2)}, base, max_rel=0.05)
    assert math.isclose(clipped["w"].norm().item() / base["w"].norm().item(), 0.05, rel_tol=1e-5)
    bad = torch.zeros(10, 10); bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        clip_generation_deltas({"w": bad}, base)


@pytest.mark.parametrize("workers", [1, 2, 4, 7])
def test_given_arbitrary_gpu_count_when_partitioning_gate_then_each_problem_is_evaluated_once(workers):
    """Given any positive GPU count; when sharded; then no gate row is lost or duplicated."""
    from rse_looper.gspo import partition_gate_rows
    rows = [{"id": str(i)} for i in range(11)]
    shards = [partition_gate_rows(rows, i, workers) for i in range(workers)]
    assert sorted(r["id"] for shard in shards for r in shard) == sorted(r["id"] for r in rows)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_given_long_rollout_when_preparing_update_then_original_tokens_are_kept_without_truncation():
    """Given a long generated solution; when preparing its loss; then the final tokens survive."""
    from rse_looper.gspo import rollout_sequence
    row = {"prompt_token_ids": list(range(300)), "completion_token_ids": list(range(1000, 2536))}
    ids, boundary = rollout_sequence(row)
    assert boundary == 300
    assert ids.shape == (1, 1836)
    assert ids[0].tolist() == row["prompt_token_ids"] + row["completion_token_ids"]


def test_given_missing_completion_when_computing_likelihood_then_prompt_tokens_are_not_substituted():
    """Given no answer tokens; when computing likelihood; then invalid boundaries fail explicitly."""
    from rse_looper.gspo import avg_sequence_logprob
    with pytest.raises(ValueError, match="completion|prompt_len"):
        avg_sequence_logprob(torch.zeros(1, 4, 6), torch.ones(1, 4, dtype=torch.long), 4)


def test_given_no_mixed_success_frontier_when_preparing_training_then_failure_is_explicit():
    """Given no usable reward variation; when preparing training; then an empty batch cannot reach CUDA."""
    from rse_looper.gspo import prepare_gspo_training_rows
    records = {"p": [{"correct": False, "completed": True, "repetition": False} for _ in range(6)]}
    with pytest.raises(ValueError, match="frontier"):
        prepare_gspo_training_rows(records)


def test_given_overlapping_input_files_when_starting_then_validation_precedes_gpu_work(tmp_path, monkeypatch):
    """Given leaked selection data; when starting a run; then validation fails before GPU setup."""
    import json
    from rse_looper.gspo import main
    import rse_looper.preflight as preflight
    row = {"id": "same", "problem": "What is 1+1?", "answer": "2"}
    train, gate = tmp_path / "train.jsonl", tmp_path / "gate.jsonl"
    train.write_text(json.dumps(row) + "\n")
    gate.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(preflight, "preflight_gpu_check", lambda **kw: pytest.fail("GPU work preceded input validation"))
    with pytest.raises(ValueError, match="overlap|disjoint"):
        main(["--train-jsonl", str(train), "--gate-jsonl", str(gate), "--gate-size", "1"])


def test_given_accuracy_regression_when_deciding_acceptance_then_auditable_reason_is_returned():
    """Given the published declining total example; when gated; then its rejection is explicit."""
    from rse_looper.gspo import rsi_step_decision
    previous = {"R3": {"correct": 37}, "R4": {"correct": 41}, "RK2": {"correct": 40},
                "total_correct": 118, "total_completed": 136, "max_rep": 1.56}
    candidate = {"R3": {"correct": 40}, "R4": {"correct": 40}, "RK2": {"correct": 34},
                 "total_correct": 114, "total_completed": 137, "max_rep": 0.0}
    decision = rsi_step_decision(previous, candidate)
    assert decision["accepted"] is False
    assert decision["reason"] in {"accuracy_regression", "mode_regression"}


def test_given_nonboolean_worker_result_when_aggregating_gate_then_truthiness_is_rejected(tmp_path, monkeypatch):
    """Given the string false in a worker shard; when aggregated; then it cannot become a success."""
    import json
    import torch.multiprocessing as mp
    from rse_looper.gspo import evaluate_gate
    def fake_spawn(worker, args, nprocs, join):
        for regime in ("R3", "R4", "RK2"):
            (tmp_path / f"test_{regime}.sh0.jsonl").write_text(json.dumps({
                "id": "p", "correct": "false", "completed": True, "repetition": False,
            }) + "\n")
    monkeypatch.setattr(mp, "spawn", fake_spawn)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_gate([0], "unused-model", None, 1.0, [{"id": "p"}], str(tmp_path), "test")


@pytest.mark.parametrize("has_frontier", [True, False])
def test_given_one_training_generation_when_completed_and_resumed_then_decisions_and_artifacts_replay(tmp_path, monkeypatch, has_frontier):
    """Given deterministic worker fixtures; when run and resumed; then only recorded accepted lineage survives."""
    import json
    from pathlib import Path
    import torch.multiprocessing as mp
    import rse_looper.checkpoint as checkpoint
    import rse_looper.gspo as gspo
    import rse_looper.preflight as preflight
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    train, gate = tmp_path / "train.jsonl", tmp_path / "gate.jsonl"
    train.write_text(json.dumps({"id": "t", "problem": "training problem", "answer": "2"}) + "\n")
    gate.write_text("\n".join(json.dumps({"id": f"g{i}", "problem": f"gate problem {i}", "answer": str(i)}) for i in range(10)) + "\n")
    calls = []
    def fake_spawn(worker, args, nprocs, join):
        calls.append(worker.__name__)
        if worker is gspo._search_worker:
            gpu_ids, model_path, rows, directory, generation, limit = args
            for shard in range(nprocs):
                records = []
                for row in rows[shard::nprocs]:
                    for regime in ("R3", "R4", "RK2"):
                        for sample in range(2):
                            records.append({**row, "regime": regime, "sample_idx": sample,
                                            "correct": has_frontier and sample == 0, "completed": True, "repetition": False,
                                            "raw_full": "sample", "old_avg_logprob": -0.5,
                                            "prompt_token_ids": [1, 3], "completion_token_ids": [4, 2]})
                (Path(directory) / f"search.shard{shard}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        elif worker is gspo._ddp_update_worker:
            torch.save({"model.layers.1.mlp.gate.weight": torch.ones(1)}, args[4])
        else:
            pytest.fail("Unexpected worker")
    def fake_gate(gpus, model, delta, scale, rows, directory, tag, limit):
        correct = 3 if delta is None else (4 if scale == 1.0 else 2)
        Path(directory).mkdir(parents=True, exist_ok=True)
        for regime in ("R3", "R4", "RK2"):
            (Path(directory) / f"{tag}_{regime}.sh0.jsonl").write_text("".join(json.dumps({
                "id": row["id"], "correct": index < correct, "completed": True, "repetition": False,
            }) + "\n" for index, row in enumerate(rows)))
        return {**{r: {"n": 10, "correct": correct, "completed": 10} for r in ("R3", "R4", "RK2")},
                "total_correct": correct * 3, "total_completed": 30, "max_rep": 0.0}
    def fake_export(parent, output, deltas):
        Path(output).mkdir(parents=True)
        (Path(output) / "model.safetensors").write_bytes(b"accepted")
    monkeypatch.setattr(mp, "spawn", fake_spawn)
    monkeypatch.setattr(gspo, "evaluate_gate", fake_gate)
    monkeypatch.setattr(checkpoint, "resolve_checkpoint_dir", lambda *args, **kwargs: base)
    monkeypatch.setattr(checkpoint, "export_edited_checkpoint", fake_export)
    monkeypatch.setattr(preflight, "enable_fast_math", lambda: False)
    monkeypatch.setattr(preflight, "preflight_gpu_check", lambda **kwargs: {"cuda_available": True})
    arguments = ["--init-model", str(base), "--train-jsonl", str(train), "--gate-jsonl", str(gate),
                 "--gate-size", "10", "--max-gens", "1" if has_frontier else "5", "--patience", "2",
                 "--steps-per-gen", "1", "--gpus", "0",
                 "--work-dir", str(tmp_path / "run"), "--ckpt-prefix", str(tmp_path / "models" / "candidate")]
    result = gspo.main(arguments)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert result["accepted_gens"] == (1 if has_frontier else 0)
    assert manifest["status"] == "completed"
    decisions = [event for event in manifest["events"] if event["event"] == "candidate_decision"]
    assert len(decisions) == (2 if has_frontier else 0)
    for event in decisions:
        assert gspo.rsi_step_decision(event["previous_gate"], event["gate"]) == {
            "accepted": event["accepted"], "reason": event["reason"],
        }
    kinds = {event.get("artifact_kind") for event in manifest["events"]}
    assert {"initial_gate_shard", "rollouts"} <= kinds
    if has_frontier:
        assert {"candidate_gate_shard", "training_items", "update_delta"} <= kinds
    else:
        skipped = [event for event in manifest["events"] if event["event"] == "generation_no_update"]
        assert len(skipped) == 2
        assert all(event["reason"] == "no_frontier" for event in skipped)
        assert manifest["next_generation"] == 2
        assert manifest["consecutive_rejections"] == 2
        assert calls == ["_search_worker", "_search_worker"]
        assert result["final_model"] == str(base.resolve())
    assert manifest["execution_fingerprint"]["source_sha256"]
    calls_before = list(calls)
    resumed = gspo.main(arguments + ["--resume"])
    assert resumed["final_model"] == result["final_model"]
    assert calls == calls_before


@pytest.mark.parametrize("passes,damped", [(3, False), (4, False), (3, True)])
def test_given_inherited_sampling_filters_when_sampling_then_recorded_likelihood_matches_actual_policy(passes, damped):
    """Given inherited filters; when sampling locally; then generation and update probabilities agree."""
    transformers = pytest.importorskip("transformers")
    from rse_looper.gspo import avg_sequence_logprob, configure_generation
    from rse_looper.looping import LoopConfig, install_midstack_loop_forward
    torch.manual_seed(9)
    config = transformers.Inkling2LoopTextConfig(
        vocab_size=32, unpadded_vocab_size=32, hidden_size=16,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=8,
        d_rel=4, rel_extent=8, sliding_window_size=8, max_position_embeddings=32,
        moe_intermediate_size=8, n_routed_experts=4, num_experts_per_tok=2,
        n_layers_per_attn_res_block=1, loop_layer_index_start=1, loop_layer_index_end=3,
        loop_gate_rank=4, loop_num_passes=3, conv_kernel_size=2,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        pad_token_id=0, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = transformers.Inkling2LoopForCausalLM(config).eval()
    install_midstack_loop_forward(model, LoopConfig(num_passes=passes, rk_window=(1, 2) if damped else None,
                                                    rk_K=2 if damped else 1))
    model.generation_config = transformers.GenerationConfig(
        temperature=0.5, top_p=0.6, top_k=2, repetition_penalty=1.2,
        forced_eos_token_id=2, suppress_tokens=[3], begin_suppress_tokens=[4],
    )
    sampling = configure_generation(model, max_new_tokens=3, pad_token_id=0, eos_token_id=2)
    sampling.output_scores = True
    sampling.return_dict_in_generate = True
    prompt = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=sampling)
        logits = model(generated.sequences, use_cache=False).logits
    completion = generated.sequences[:, prompt.shape[1]:]
    behavior_logprobs = torch.stack([
        torch.log_softmax(scores, dim=-1).gather(1, completion[:, index:index + 1]).squeeze(1)
        for index, scores in enumerate(generated.scores)
    ]).mean()
    update_logprobs = avg_sequence_logprob(logits, generated.sequences, prompt.shape[1])
    assert torch.allclose(behavior_logprobs, update_logprobs, atol=1e-5, rtol=1e-5)
    assert all(torch.isfinite(scores).all() for scores in generated.scores)
