"""Tests for src/model/policy_net.py — stochastic policy π_θ(a|s) (brief §7.4 + TR5)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

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


def test_categorical_not_argmax_during_training():
    """With all actions legal, sampling over many draws yields >1 unique action.

    TR5: Categorical sampling — not argmax — keeps exploration on-policy.
    """
    torch.manual_seed(0)
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    mask = torch.ones(ACTION_COUNT, dtype=torch.bool)
    actions = {net.sample(state, action_mask=mask)[0] for _ in range(200)}
    assert len(actions) > 1, "policy collapsed to argmax (no exploration)"


def test_log_prob_differentiable():
    """log_prob(state, action).backward() does not crash and yields grads."""
    net = PolicyNet(seed=0)
    state = torch.randn(STATE_DIM)
    log_p = net.log_prob(state, action_id=2)
    log_p.backward()
    # at least one parameter received a gradient
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


def test_one_layer_fc_128_architecture():
    """TR5 grading anchor: exactly one Linear(STATE_DIM, 128) before head Linear(128, ACTION_COUNT)."""
    net = PolicyNet(hidden=128, seed=0)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2, f"expected exactly 2 Linear layers (1 hidden + head); got {len(linears)}"
    hidden, head = linears
    assert hidden.in_features == STATE_DIM
    assert hidden.out_features == 128
    assert head.in_features == 128
    assert head.out_features == ACTION_COUNT


def test_forward_rejects_wrong_state_ndim():
    """Covers policy_net.py line 42 — ValueError on 3-D input."""
    net = PolicyNet()
    bad = torch.zeros((1, 1, STATE_DIM), dtype=torch.float32)
    with pytest.raises(ValueError, match="state must be 1-D"):
        net(bad)


def test_forward_rejects_wrong_state_last_dim():
    """Covers policy_net.py line 46 — ValueError on bad last-dim."""
    net = PolicyNet()
    bad = torch.zeros((STATE_DIM + 1,), dtype=torch.float32)
    with pytest.raises(ValueError, match="last dim must be"):
        net(bad)


def test_sample_rejects_bad_action_mask_shape():
    """Covers policy_net.py line 56 — ValueError on wrong mask shape."""
    net = PolicyNet()
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    bad_mask = torch.ones((ACTION_COUNT + 1,), dtype=torch.bool)
    with pytest.raises(ValueError, match="action_mask"):
        net.sample(state, action_mask=bad_mask)


def test_log_prob_rejects_action_id_out_of_range():
    """Covers policy_net.py line 89 — ValueError on bad action_id."""
    net = PolicyNet()
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    with pytest.raises(ValueError, match="action_id must be in"):
        net.log_prob(state, action_id=99)


def test_num_parameters_property_returns_positive():
    """Covers policy_net.py line 100 — num_parameters property."""
    net = PolicyNet(hidden=64)
    assert net.num_parameters > 0
    assert isinstance(net.num_parameters, int)


def test_seed_none_path_does_not_crash():
    """Covers policy_net.py line 32->34 branch — seed=None path."""
    net = PolicyNet(hidden=16, seed=None)
    assert net.num_parameters > 0
