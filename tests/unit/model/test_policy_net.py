"""Tests for src/model/policy_net.py — stochastic policy π_θ(a|s) (brief §7.4 + TR5).

Edge-case + ValueError-path tests live in test_policy_net_extra.py
so this file stays ≤150 LOC (CLAUDE.md §1).
"""

from __future__ import annotations

import torch

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.policy_net import PolicyNet


def test_forward_output_shape_logits():
    """Single state in (STATE_DIM,) → logits (ACTION_COUNT,)."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    logits = net.forward(state)
    assert logits.shape == (ACTION_COUNT,)
    assert torch.isfinite(logits).all()


def test_forward_batch_shape():
    """(batch, STATE_DIM) → (batch, ACTION_COUNT)."""
    net = PolicyNet(seed=0)
    batch = torch.randn(16, STATE_DIM)
    logits = net.forward(batch)
    assert logits.shape == (16, ACTION_COUNT)


def test_sample_returns_int_in_range():
    """sample() returns an action_id in [0, ACTION_COUNT)."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    for _ in range(20):
        action_id, _ = net.sample(state)
        assert isinstance(action_id, int)
        assert 0 <= action_id < ACTION_COUNT


def test_sample_returns_log_prob_finite():
    """log_prob is a finite scalar tensor."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    _, log_prob = net.sample(state)
    assert isinstance(log_prob, float)
    assert torch.isfinite(torch.tensor(log_prob))


def test_mask_zeros_illegal_action_probability():
    """With mask of all-False except Mobility (id=6), sample always returns 6."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    mask[6] = True  # only Mobility legal
    for _ in range(30):
        action_id, log_prob = net.sample(state, action_mask=mask)
        assert action_id == 6
        # log(1.0) == 0.0 for the only legal action
        assert abs(log_prob - 0.0) < 1e-5


def test_log_prob_differentiable():
    """log_prob(state, action).backward() does not crash and yields grads."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    log_p = net.log_prob(state, action_id=2)
    log_p.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_seeded_init_deterministic():
    """Two PolicyNets with the same seed produce identical logits."""
    net_a = PolicyNet(seed=42)
    net_b = PolicyNet(seed=42)
    state = torch.randn(STATE_DIM)
    out_a = net_a.forward(state)
    out_b = net_b.forward(state)
    assert torch.allclose(out_a, out_b)
