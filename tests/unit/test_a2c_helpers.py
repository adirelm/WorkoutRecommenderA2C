"""Tests for src/services/a2c_helpers.py — TD advantage + actor/critic losses."""

from __future__ import annotations

import pytest
import torch

from src.services.a2c_helpers import (
    actor_loss,
    compute_advantages_td,
    critic_loss,
)


def test_advantage_td_with_done_no_bootstrap():
    """When done=True at step t, advantage = r - V (no gamma*V' bootstrap)."""
    adv = compute_advantages_td(rewards=[5.0], values=[2.0], next_values=[100.0], dones=[True], gamma=0.99)
    assert adv == pytest.approx([5.0 - 2.0])


def test_advantage_td_without_done_full_formula():
    """delta = r + gamma * V' - V."""
    adv = compute_advantages_td(rewards=[1.0], values=[3.0], next_values=[10.0], dones=[False], gamma=0.9)
    assert adv == pytest.approx([1.0 + 0.9 * 10.0 - 3.0])


def test_advantage_handles_zero_gamma():
    """gamma=0 -> delta = r - V (myopic, no bootstrapping)."""
    adv = compute_advantages_td(rewards=[2.0], values=[1.0], next_values=[100.0], dones=[False], gamma=0.0)
    assert adv == pytest.approx([1.0])


def test_advantage_rejects_gamma_out_of_range():
    with pytest.raises(ValueError, match="gamma must be in"):
        compute_advantages_td([1.0], [0.0], [0.0], [False], gamma=1.5)


def test_advantage_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_advantages_td([1.0, 2.0], [0.0], [0.0], [False])


def test_actor_loss_gradient_sign_positive_advantage():
    """positive advantage -> grad on log_prob is negative (increase log_prob)."""
    log_probs = [torch.tensor(0.0, requires_grad=True)]
    advantages = [1.0]
    loss = actor_loss(log_probs, advantages, entropy_coef=0.0)
    loss.backward()
    assert log_probs[0].grad < 0


def test_actor_loss_entropy_bonus_changes_loss():
    log_probs = [torch.tensor(0.0, requires_grad=True)]
    entropies = [torch.tensor(2.0, requires_grad=True)]
    advantages = [1.0]
    no_entropy = actor_loss([log_probs[0]], advantages, entropy_coef=0.0).item()
    with_entropy = actor_loss([log_probs[0]], advantages, entropy_coef=0.1, entropies=entropies).item()
    assert no_entropy != with_entropy


def test_actor_loss_rejects_empty():
    with pytest.raises(ValueError, match="empty trajectory"):
        actor_loss([], [])


def test_critic_loss_zero_when_value_equals_target():
    values = [torch.tensor(5.0, requires_grad=True)]
    loss = critic_loss(values, [5.0])
    assert loss.item() == pytest.approx(0.0)


def test_critic_loss_quadratic():
    """Value off by 2 -> loss = 4 (mse on single sample)."""
    values = [torch.tensor(3.0, requires_grad=True)]
    loss = critic_loss(values, [5.0])
    assert loss.item() == pytest.approx(4.0)


def test_critic_loss_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        critic_loss([torch.tensor(1.0)], [1.0, 2.0])
