"""Unit tests for rse_looper.fpsolve (100% backprop-free Closed-Form Null-Space FP-Solve)."""

import math
import pytest
import torch

from rse_looper.fpsolve import solve_nullspace_ridge


def test_solve_nullspace_ridge_exact_target_and_orthogonal_preservation():
    torch.manual_seed(42)
    # Construct disjoint subspaces: X_keep lives in dims 0..7, X_edit lives in dims 8..15
    D_in, D_out = 16, 12
    X_keep = torch.zeros(64, D_in)
    X_keep[:, :8] = torch.randn(64, 8)
    X_edit = torch.zeros(64, D_in)
    X_edit[:, 8:] = torch.randn(64, 8)

    true_dW = torch.zeros(D_out, D_in)
    true_dW[:, 8:] = torch.randn(D_out, 8)
    dY = X_edit @ true_dW.T

    dW, _ = solve_nullspace_ridge(X_edit, dY, X_keep=X_keep, lam_keep=100.0, lam_ridge_frac=1e-4)
    # Must match target residual accurately on X_edit AND produce ~0 perturbation on X_keep!
    err_edit = (X_edit @ dW.T - dY).norm().item() / dY.norm().item()
    drift_keep = (X_keep @ dW.T).norm().item() / dY.norm().item()
    assert err_edit < 1e-2, f"Edit residual error too large: {err_edit}"
    assert drift_keep < 1e-5, f"Null-space preservation violated: {drift_keep}"


def test_solve_nullspace_ridge_shared_subspace_tradeoff_and_norm_cap():
    torch.manual_seed(0)
    X_edit = torch.randn(48, 20)
    X_keep = torch.randn(48, 20)
    W_orig = torch.randn(10, 20)
    dY = torch.randn(48, 10) * 5.0

    dW_low_keep, _ = solve_nullspace_ridge(X_edit, dY, X_keep=X_keep, lam_keep=0.1, lam_ridge_frac=0.05)
    dW_high_keep, _ = solve_nullspace_ridge(X_edit, dY, X_keep=X_keep, lam_keep=50.0, lam_ridge_frac=0.05)
    assert (X_keep @ dW_high_keep.T).norm().item() < (X_keep @ dW_low_keep.T).norm().item()

    dW_cap, rel = solve_nullspace_ridge(X_edit, dY, X_keep=X_keep, lam_keep=1.0, max_rel=0.10, W_orig=W_orig)
    assert math.isclose(rel, 0.10, rel_tol=1e-5)
    assert math.isclose(dW_cap.norm().item() / W_orig.norm().item(), 0.10, rel_tol=1e-5)


def test_solve_nullspace_ridge_rejects_bad_shapes_and_nan():
    with pytest.raises(ValueError):
        solve_nullspace_ridge(torch.randn(10, 8), torch.randn(12, 4))
    bad = torch.randn(10, 8)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        solve_nullspace_ridge(bad, torch.randn(10, 4))


@pytest.mark.parametrize("kwargs", [
    {"lam_keep": -1.0}, {"lam_keep": float("nan")},
    {"lam_ridge_frac": -1.0}, {"lam_ridge_frac": float("inf")},
    {"max_rel": -0.1}, {"max_rel": float("nan")},
])
def test_given_invalid_regularization_when_solving_then_fail_before_linear_algebra(kwargs):
    """Given invalid regularization, When solving, Then reject invalid optimization objectives."""
    with pytest.raises(ValueError):
        solve_nullspace_ridge(torch.eye(2), torch.ones(2, 1), **kwargs)


def test_given_empty_or_nonfinite_preservation_data_when_solving_then_reject():
    """Given malformed calibration data, When solving, Then do not return a misleading edit."""
    with pytest.raises(ValueError, match="empty"):
        solve_nullspace_ridge(torch.empty(0, 2), torch.empty(0, 1))
    with pytest.raises(ValueError, match="finite"):
        solve_nullspace_ridge(torch.eye(2), torch.ones(2, 1), X_keep=torch.full((1, 2), float("nan")))
    with pytest.raises(ValueError, match="W_orig"):
        solve_nullspace_ridge(torch.eye(2), torch.ones(2, 1), W_orig=torch.ones(3, 2))


def test_given_spd_ridge_system_when_solving_then_use_cholesky_without_gradients(monkeypatch):
    """Given the stated SPD objective, When solving, Then use Cholesky and match its normal equations."""
    calls = []
    original = torch.linalg.cholesky

    def observed_cholesky(matrix, *args, **kwargs):
        calls.append(matrix.detach().clone())
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", observed_cholesky)
    x = torch.tensor([[1., 0.], [0., 2.]], requires_grad=True)
    y = torch.tensor([[1.], [2.]], requires_grad=True)
    delta, _ = solve_nullspace_ridge(x, y, lam_ridge_frac=0.1)
    assert len(calls) == 1
    assert torch.allclose(calls[0] @ delta.T, x.detach().T @ y.detach() / 2)
    assert not delta.requires_grad


def test_given_model_revision_when_parsing_fpsolve_then_pin_the_model_snapshot():
    """Given an immutable Hub revision, When parsing FP-Solve arguments, Then preserve it for loading."""
    from rse_looper.fpsolve import build_parser
    args = build_parser().parse_args(["--model-revision", "deadbeef"])
    assert args.model_revision == "deadbeef"
