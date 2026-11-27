"""CPU unit tests for rse_looper.preflight (TF32 fast math & GPU/kernel preflight checks)."""

import importlib.util
from importlib.machinery import ModuleSpec
import pytest
import torch

from rse_looper.preflight import enable_fast_math, preflight_gpu_check


@pytest.fixture
def restore_tf32_flags():
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_enable_fast_math_safe_noop_when_cuda_unavailable(monkeypatch, restore_tf32_flags):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    assert enable_fast_math() is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_enable_fast_math_enables_tf32_when_cuda_available(monkeypatch, restore_tf32_flags):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    assert enable_fast_math() is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_preflight_gpu_check_structure_and_missing_causal_conv1d(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    report = preflight_gpu_check(require_causal_conv1d=False)
    assert set(report.keys()) == {
        "cuda_available",
        "allow_tf32",
        "flash_sdp",
        "causal_conv1d_installed",
        "warnings",
    }
    assert report["cuda_available"] is False
    assert report["flash_sdp"] is False
    assert report["causal_conv1d_installed"] is False
    assert isinstance(report["allow_tf32"], bool)
    assert isinstance(report["warnings"], list) and len(report["warnings"]) >= 2

    with pytest.raises(RuntimeError, match="causal_conv1d"):
        preflight_gpu_check(require_causal_conv1d=True)


def test_preflight_gpu_check_clean_cuda_environment(monkeypatch, restore_tf32_flags):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: ModuleSpec(name, loader=None) if name == "causal_conv1d" else None,
    )

    report = preflight_gpu_check(require_causal_conv1d=True)
    assert report == {
        "cuda_available": True,
        "allow_tf32": True,
        "flash_sdp": True,
        "causal_conv1d_installed": True,
        "warnings": [],
    }
