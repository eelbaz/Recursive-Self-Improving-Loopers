"""Given/When/Then scenarios using a tiny initialized upstream model; no model downloads."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

from rse_looper.looping import LoopConfig, install_midstack_loop_forward
from rse_looper.fpsolve import run_fpsolve_on_model


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    config = Inkling2LoopTextConfig(
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
    return Inkling2LoopForCausalLM(config).eval()


def test_given_native_three_pass_model_when_configured_then_logits_and_forward_are_unchanged(tiny_model):
    """Given native three-pass inference, When normal looping is configured, Then preserve exact logits."""
    ids = torch.tensor([[1, 3, 5, 7]])
    original_forward = tiny_model.model.forward
    with torch.no_grad():
        expected = tiny_model(ids, use_cache=False).logits
        install_midstack_loop_forward(tiny_model, LoopConfig(num_passes=3))
        actual = tiny_model(ids, use_cache=False).logits
    assert tiny_model.model.forward == original_forward
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("cfg", [LoopConfig(num_passes=4), LoopConfig(num_passes=3, rk_window=(1, 2), rk_K=2)])
def test_given_loop_mode_when_decoding_with_cache_then_match_full_sequence_and_advance_once(tiny_model, cfg):
    """Given a loop mode, When decoding one token with KV/conv caches, Then match uncached decoding."""
    ids = torch.tensor([[1, 3, 5, 7]])
    install_midstack_loop_forward(tiny_model, cfg)
    with torch.no_grad():
        expected = tiny_model(ids, use_cache=False).logits[:, -1]
        prefix = tiny_model(ids[:, :-1], use_cache=True)
        result = tiny_model(ids[:, -1:], cache=prefix.cache, use_cache=True)
    assert torch.allclose(result.logits[:, -1], expected, atol=1e-5, rtol=1e-4)
    assert result.cache.get_seq_length() == ids.shape[1]
    assert all(layer.get_seq_length() == ids.shape[1] for layer in result.cache.layers)


def test_given_damped_mode_when_returning_to_normal_then_no_wrapper_accumulates(tiny_model):
    """Given installed damping, When switching regimes repeatedly, Then native execution is restored."""
    ids = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        expected = tiny_model(ids, use_cache=False).logits
        for _ in range(2):
            install_midstack_loop_forward(tiny_model, LoopConfig(rk_window=(1, 2), rk_K=2))
            assert torch.isfinite(tiny_model(ids, use_cache=False).logits).all()
            install_midstack_loop_forward(tiny_model, LoopConfig())
        actual = tiny_model(ids, use_cache=False).logits
    assert torch.equal(actual, expected)


def test_given_tiny_model_when_fpsolve_runs_then_only_requested_weights_change_and_mode_restores(tiny_model):
    """Given unlabeled inputs, When fitting two routers, Then leave all other weights and grad mode intact."""
    ids = torch.tensor([[1, 3, 5]])
    original = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    with torch.enable_grad():
        result = run_fpsolve_on_model(
            tiny_model, [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}],
            subnames=("mlp.gate",), layer_range=(1, 3),
        )
        assert torch.is_grad_enabled()
    assert result["n_matrices"] == 2
    assert tiny_model.config.loop_num_passes == 3
    for name, value in tiny_model.state_dict().items():
        if name not in {"model.layers.1.mlp.gate.weight", "model.layers.2.mlp.gate.weight"}:
            assert torch.equal(value, original[name]), name
    assert all(p.grad is None for p in tiny_model.parameters())


def test_given_failed_calibration_when_fpsolve_runs_then_remove_hooks_and_restore_weights(tiny_model, monkeypatch):
    """Given a failure after the first edit, When fitting, Then roll back edits, hooks and loop mode."""
    ids = torch.tensor([[1, 3, 5]])
    original_forward = tiny_model.forward
    original = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    calls = 0

    def fail_late(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("calibration failed")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "forward", fail_late)
    with pytest.raises(RuntimeError, match="calibration failed"):
        run_fpsolve_on_model(
            tiny_model, [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}],
            subnames=("mlp.gate",), layer_range=(1, 3),
        )
    assert calls == 3
    assert tiny_model.config.loop_num_passes == 3
    assert all(not mod._forward_hooks for mod in tiny_model.modules())
    for name, value in tiny_model.state_dict().items():
        assert torch.equal(value, original[name]), name


def test_given_empty_calibration_when_fpsolve_runs_then_reject_before_model_changes(tiny_model):
    """Given no calibration samples, When fitting, Then fail before installing another loop mode."""
    with pytest.raises(ValueError, match="calibration"):
        run_fpsolve_on_model(tiny_model, [], subnames=("mlp.gate",), layer_range=(1, 3))
    assert tiny_model.config.loop_num_passes == 3
