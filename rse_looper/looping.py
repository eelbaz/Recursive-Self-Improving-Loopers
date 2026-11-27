"""Native Inkling recurrence and damped per-layer substeps.

Damping is fixed-point relaxation, not a second-order Runge-Kutta guarantee.
Each cached substep retains its own previous-token history.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass(frozen=True)
class LoopConfig:
    num_passes: int = 3
    rk_window: Optional[Tuple[int, int]] = None
    rk_K: int = 1
    rk_beta: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.num_passes, int) or self.num_passes < 1:
            raise ValueError("num_passes must be a positive integer")
        if not isinstance(self.rk_K, int) or self.rk_K < 1:
            raise ValueError("rk_K must be a positive integer")
        if not (0.0 < self.rk_beta <= 1.0):
            raise ValueError("rk_beta must be in (0, 1]")
        if self.rk_window is not None:
            lo, hi = self.rk_window
            if not isinstance(lo, int) or not isinstance(hi, int) or lo > hi or lo < 0:
                raise ValueError(f"Invalid rk_window {self.rk_window}")


def _clone_state(value):
    """Clone mutable cache state without detaching tensors from autograd."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(v) for v in value)
    return copy.copy(value)


def snapshot_cache_layer(cache: Any, cache_idx: int) -> Optional[Dict[str, Any]]:
    """Capture a complete slot, including lazy initialization and conv counters."""
    if cache is None:
        return None
    if not hasattr(cache, "layers") or not 0 <= cache_idx < len(cache.layers):
        raise ValueError(f"Cache slot {cache_idx} does not exist")
    snap = {"layer_state": _clone_state(vars(cache.layers[cache_idx]))}
    conv = getattr(cache, "conv_states", None)
    if isinstance(conv, (dict, list)):
        present = cache_idx in conv if isinstance(conv, dict) else cache_idx < len(conv)
        snap["external_conv_present"] = present
        if present:
            snap["external_conv"] = _clone_state(conv[cache_idx])
    return snap


def restore_cache_layer(cache: Any, cache_idx: int, snap: Optional[Dict[str, Any]]) -> None:
    """Restore all slot state, including the absence of lazily initialized tensors."""
    if cache is None or snap is None:
        return
    state = vars(cache.layers[cache_idx])
    state.clear()
    state.update(_clone_state(snap["layer_state"]))
    conv = getattr(cache, "conv_states", None)
    if snap.get("external_conv_present"):
        if isinstance(conv, list):
            while len(conv) <= cache_idx:
                conv.append(None)
        conv[cache_idx] = _clone_state(snap["external_conv"])
    elif isinstance(conv, dict):
        conv.pop(cache_idx, None)
    elif isinstance(conv, list) and cache_idx < len(conv):
        conv[cache_idx] = None


def _swap_slots(cache, left, right):
    cache.layers[left], cache.layers[right] = cache.layers[right], cache.layers[left]
    conv = getattr(cache, "conv_states", None)
    if isinstance(conv, dict):
        missing = object()
        a, b = conv.pop(left, missing), conv.pop(right, missing)
        if b is not missing:
            conv[left] = b
        if a is not missing:
            conv[right] = a
    elif isinstance(conv, list):
        while len(conv) <= max(left, right):
            conv.append(None)
        conv[left], conv[right] = conv[right], conv[left]


def run_damped_rk_layer(
    layer_fn, h: torch.Tensor, rk_K: int = 1, rk_beta: float = 0.3,
    cache: Any = None, cache_idx: int = 0,
) -> torch.Tensor:
    """Apply K damped substeps; K=1 is exactly one native layer call.

    Extra stage slots live in the native cache container so crop/reorder also
    handles their histories. A regime change requires a fresh dynamic cache.
    """
    if rk_K <= 1:
        return layer_fn(h)
    LoopConfig(rk_K=rk_K, rk_beta=rk_beta)
    stage_slots, snapshots = (), {}
    original_length, created = 0, False
    key = (cache_idx, rk_K, float(rk_beta))
    if cache is not None:
        if getattr(cache, "offloading", False) or getattr(cache, "is_compileable", False):
            raise ValueError("Damped substeps require an eager, non-offloaded dynamic cache")
        base = snapshot_cache_layer(cache, cache_idx)
        original_length = len(cache.layers)
        stages = getattr(cache, "_rse_damped_slots", {})
        if key not in stages:
            slot = cache.layers[cache_idx]
            if hasattr(slot, "get_seq_length") and slot.get_seq_length() != 0:
                raise ValueError("Changing a loop configuration requires a fresh cache")
            if any(k[0] == cache_idx for k in stages):
                raise ValueError("Changing damped substeps requires a fresh cache")
            stage_slots = tuple(range(original_length, original_length + rk_K - 1))
            for idx in stage_slots:
                cache.layers.append(copy.copy(slot))
                cache.layers[idx].__dict__ = {}
                restore_cache_layer(cache, idx, base)
            stages[key] = stage_slots
            cache._rse_damped_slots = stages
            created = True
        else:
            stage_slots = stages[key]
        snapshots = {idx: snapshot_cache_layer(cache, idx) for idx in (cache_idx, *stage_slots)}
    cur = h
    try:
        for k in range(rk_K):
            provisional = cache is not None and k < rk_K - 1
            if provisional:
                _swap_slots(cache, cache_idx, stage_slots[k])
            try:
                out = layer_fn(cur)
            finally:
                if provisional:
                    _swap_slots(cache, cache_idx, stage_slots[k])
            cur = cur + float(rk_beta) * (out - cur) if rk_beta != 1.0 else out
        return cur
    except Exception:
        for idx, snap in snapshots.items():
            restore_cache_layer(cache, idx, snap)
        if cache is not None and created:
            del cache.layers[original_length:]
            cache._rse_damped_slots.pop(key)
            conv = getattr(cache, "conv_states", None)
            if isinstance(conv, dict):
                for idx in stage_slots:
                    conv.pop(idx, None)
            elif isinstance(conv, list):
                del conv[original_length:]
        raise


def install_midstack_loop_forward(model: torch.nn.Module, cfg: LoopConfig) -> None:
    """Keep native masks, recurrence, output types and cache indexing intact.

    Only selected decoder layers receive removable wrappers for damped substeps.
    Reinstalling a normal mode removes those wrappers.
    """
    tm = model.model if hasattr(model, "model") else model
    config = tm.config
    required = ("loop_layer_index_start", "loop_layer_index_end", "loop_num_passes")
    if not all(hasattr(config, name) for name in required) or not hasattr(tm, "layers"):
        raise TypeError("Expected the supported upstream Inkling2Loop text model")
    if cfg.rk_window is not None and cfg.rk_window[1] >= len(tm.layers):
        raise ValueError("rk_window extends past the model's decoder layers")
    for layer in tm.layers:
        if hasattr(layer, "_rse_original_forward"):
            layer.forward = layer._rse_original_forward
            delattr(layer, "_rse_original_forward")
    config.loop_num_passes = cfg.num_passes
    if hasattr(model, "config"):
        model.config.loop_num_passes = cfg.num_passes
    tm._rse_loop_config = cfg
    if cfg.rk_window is None or cfg.rk_K == 1:
        return

    def wrap(original, layer_idx):
        @wraps(original)
        def damped(hidden_states, *args, **kwargs):
            cache = kwargs.get("cache", kwargs.get("past_key_values"))
            cache_idx = layer_idx + kwargs.get("cache_offset", 0)
            return run_damped_rk_layer(
                lambda x: original(x, *args, **kwargs), hidden_states,
                rk_K=cfg.rk_K, rk_beta=cfg.rk_beta, cache=cache, cache_idx=cache_idx,
            )
        return damped

    for idx in range(cfg.rk_window[0], cfg.rk_window[1] + 1):
        layer = tm.layers[idx]
        layer._rse_original_forward = layer.forward
        layer.forward = wrap(layer.forward, idx)
