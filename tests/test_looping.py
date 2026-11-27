"""Unit tests for rse_looper.looping (bit-exact RK identity and KV/conv cache isolation)."""

import pytest
import torch

from rse_looper.looping import (
    LoopConfig,
    restore_cache_layer,
    run_damped_rk_layer,
    snapshot_cache_layer,
)


class _DummyCacheLayer:
    def __init__(self, k, v):
        self.keys = k
        self.values = v


class _DummyCache:
    def __init__(self):
        self.layers = [_DummyCacheLayer(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))]
        self.conv_states = {0: {"attn": torch.ones(1, 4, 3), "mlp": torch.full((1, 4, 3), 2.0)}}


def test_loop_config_validation():
    cfg = LoopConfig(num_passes=4, rk_window=(10, 13), rk_K=2, rk_beta=0.3)
    assert cfg.num_passes == 4
    with pytest.raises(ValueError):
        LoopConfig(num_passes=0)
    with pytest.raises(ValueError):
        LoopConfig(rk_beta=0.0)
    with pytest.raises(ValueError):
        LoopConfig(rk_window=(13, 10))


def test_cache_snapshot_restore_prevents_state_corruption():
    cache = _DummyCache()
    snap = snapshot_cache_layer(cache, 0)
    cache.layers[0].keys.fill_(99.0)
    cache.conv_states[0]["attn"].fill_(-5.0)
    restore_cache_layer(cache, 0, snap)
    assert torch.equal(cache.layers[0].keys, torch.zeros(1, 2, 4))
    assert torch.equal(cache.conv_states[0]["attn"], torch.ones(1, 4, 3))
    assert torch.equal(cache.conv_states[0]["mlp"], torch.full((1, 4, 3), 2.0))


def test_damped_rk_bit_exact_when_k1_and_cache_isolated_for_k3():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 8)
    cache = _DummyCache()

    calls = {"n": 0}
    def stateful_layer(x):
        calls["n"] += 1
        cache.layers[0].keys = cache.layers[0].keys + 1.0
        cache.conv_states[0]["attn"] = cache.conv_states[0]["attn"] + 1.0
        return x * 1.1 + 0.25

    # K=1 matches single call bit-for-bit
    out_k1 = run_damped_rk_layer(stateful_layer, h, rk_K=1, rk_beta=0.3, cache=cache, cache_idx=0)
    assert torch.equal(out_k1, h * 1.1 + 0.25)
    assert calls["n"] == 1
    assert torch.equal(cache.layers[0].keys, torch.ones(1, 2, 4))

    # Reset and run K=3: cache must increment ONCE (only from the final commit step, NOT 3x!)
    cache = _DummyCache()
    calls["n"] = 0
    out_k3 = run_damped_rk_layer(stateful_layer, h, rk_K=3, rk_beta=0.3, cache=cache, cache_idx=0)
    assert calls["n"] == 3
    assert torch.equal(cache.layers[0].keys, torch.ones(1, 2, 4))
    assert torch.equal(cache.conv_states[0]["attn"], torch.full((1, 4, 3), 2.0))
    assert out_k3.shape == h.shape and torch.isfinite(out_k3).all()
