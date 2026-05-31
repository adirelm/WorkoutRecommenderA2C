"""Tests for src/model/actor_critic.py — Actor + Critic (brief §7.5)."""

from __future__ import annotations

import pytest
import torch

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.actor_critic import ActorCriticNet


def test_forward_returns_logits_and_value():
    net = ActorCriticNet(seed=0)
    state = torch.randn(STATE_DIM)
    logits, value = net.forward(state)
    assert logits.shape == (ACTION_COUNT,)
    assert value.dim() == 0 or value.shape == ()


def test_sample_returns_three_tuple():
    net = ActorCriticNet(seed=0)
    state = torch.randn(STATE_DIM)
    action, log_prob, value = net.sample(state)
    assert isinstance(action, int)
    assert 0 <= action < ACTION_COUNT
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_actor_and_critic_separate_parameters():
    """Brief §5.2: disjoint parameter sets between actor + critic."""
    net = ActorCriticNet(actor_hidden=32, critic_hidden=16, seed=0)
    actor_params = {id(p) for p in [*net.actor_fc.parameters(), *net.actor_head.parameters()]}
    critic_params = {id(p) for p in [*net.critic_fc.parameters(), *net.critic_head.parameters()]}
    assert actor_params.isdisjoint(critic_params)


def test_seeded_init_deterministic():
    a = ActorCriticNet(seed=42)
    b = ActorCriticNet(seed=42)
    state = torch.randn(STATE_DIM)
    la, va = a.forward(state)
    lb, vb = b.forward(state)
    assert torch.allclose(la, lb)
    assert torch.allclose(va, vb)


def test_mask_zeros_illegal_action_probability():
    net = ActorCriticNet(seed=0)
    state = torch.randn(STATE_DIM)
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    mask[6] = True
    for _ in range(15):
        action, _, _ = net.sample(state, action_mask=mask)
        assert action == 6


def test_value_is_finite():
    net = ActorCriticNet(seed=0)
    state = torch.randn(STATE_DIM)
    _, value = net.forward(state)
    assert torch.isfinite(value).item()


def test_log_prob_and_value_differentiable():
    net = ActorCriticNet(seed=0)
    state = torch.randn(STATE_DIM)
    log_p, value = net.log_prob_and_value(state, action_id=3)
    (log_p + value).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_forward_rejects_wrong_state_ndim():
    net = ActorCriticNet(seed=0)
    bad = torch.zeros((1, 1, STATE_DIM), dtype=torch.float32)
    with pytest.raises(ValueError, match="state must be 1-D"):
        net(bad)


def test_log_prob_rejects_action_id_out_of_range():
    net = ActorCriticNet(seed=0)
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    with pytest.raises(ValueError, match="action_id"):
        net.log_prob_and_value(state, action_id=99)


def test_mask_wrong_shape_raises():
    net = ActorCriticNet(seed=0)
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    bad_mask = torch.ones((ACTION_COUNT + 1,), dtype=torch.bool)
    with pytest.raises(ValueError, match="action_mask"):
        net.sample(state, action_mask=bad_mask)


def test_num_parameters_positive():
    net = ActorCriticNet()
    assert net.num_parameters > 0


def test_seed_none_path_does_not_crash():
    net = ActorCriticNet(seed=None)
    assert net.num_parameters > 0
