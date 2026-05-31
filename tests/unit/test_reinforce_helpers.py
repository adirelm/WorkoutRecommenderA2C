"""Direct tests for src/services/reinforce_helpers.py — error-path coverage."""

from __future__ import annotations

import pytest
import torch

from src.services.reinforce_helpers import compute_returns, reinforce_loss


def test_compute_returns_rejects_gamma_below_zero():
    """Covers reinforce_helpers.py gamma range raise."""
    with pytest.raises(ValueError, match="gamma must be in"):
        compute_returns([1.0, 2.0, 3.0], gamma=-0.1)


def test_compute_returns_rejects_gamma_above_one():
    with pytest.raises(ValueError, match="gamma must be in"):
        compute_returns([1.0], gamma=1.5)


def test_compute_returns_two_step_matches_hand_calc():
    """Sanity: G_0 = r_0 + gamma * r_1 ; G_1 = r_1."""
    returns = compute_returns([1.0, 1.0], gamma=0.9)
    assert returns[1] == pytest.approx(1.0)
    assert returns[0] == pytest.approx(1.9)


def test_compute_returns_empty_input_returns_empty():
    assert compute_returns([], gamma=0.99) == []


def test_reinforce_loss_rejects_length_mismatch():
    """Covers reinforce_helpers.py length-mismatch raise."""
    log_probs = [torch.tensor(0.0, requires_grad=True) for _ in range(3)]
    returns = [1.0, 2.0]  # len 2 vs log_probs len 3
    with pytest.raises(ValueError, match="length mismatch"):
        reinforce_loss(log_probs, returns)


def test_reinforce_loss_rejects_empty_trajectory():
    """Covers reinforce_helpers.py empty-trajectory raise."""
    with pytest.raises(ValueError, match="empty trajectory"):
        reinforce_loss([], [])


def test_reinforce_loss_returns_scalar_tensor_with_grad():
    """Loss must be a 0-d differentiable tensor."""
    log_probs = [torch.tensor(-0.5, requires_grad=True) for _ in range(3)]
    returns = [1.0, 2.0, 3.0]
    loss = reinforce_loss(log_probs, returns, baseline=0.5)
    assert loss.dim() == 0
    loss.backward()
    assert all(lp.grad is not None for lp in log_probs)


def test_reinforce_loss_gradient_sign_positive_return():
    """High return → gradient on log_prob of taken action is negative (we want to INCREASE it).
    L = -E[(G-b) log π]; dL/dlogπ = -(G-b). With G=1, b=0 → dL/dlogπ = -1 < 0.
    """
    log_probs = [torch.tensor(0.0, requires_grad=True)]
    returns = [1.0]
    loss = reinforce_loss(log_probs, returns, baseline=0.0)
    loss.backward()
    assert log_probs[0].grad < 0, f"expected negative grad for positive return; got {log_probs[0].grad}"


def test_reinforce_loss_gradient_sign_negative_return():
    """Negative return → gradient is positive (we want to DECREASE the log_prob)."""
    log_probs = [torch.tensor(0.0, requires_grad=True)]
    returns = [-1.0]
    loss = reinforce_loss(log_probs, returns, baseline=0.0)
    loss.backward()
    assert log_probs[0].grad > 0, f"expected positive grad for negative return; got {log_probs[0].grad}"


def test_reinforce_loss_baseline_equals_return_zero_loss():
    """Baseline subtraction: when baseline == G, advantage == 0 → loss == 0 (exact)."""
    log_probs = [torch.tensor(0.5, requires_grad=True), torch.tensor(-0.3, requires_grad=True)]
    returns = [5.0, 5.0]
    loss = reinforce_loss(log_probs, returns, baseline=5.0)
    assert loss.abs().item() < 1e-6


def test_compute_returns_unit_gamma_reverse_cumsum():
    """gamma=1 → returns are reverse cumulative sum of rewards."""
    returns = compute_returns([1.0, 2.0, 3.0], gamma=1.0)
    assert returns == pytest.approx([6.0, 5.0, 3.0])
