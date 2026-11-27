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
