"""Extra tests for src/model/policy_net.py — split to keep test_policy_net.py ≤150 LOC."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.policy_net import PolicyNet


def test_categorical_sample_frequency_matches_softmax_distribution():
    """P1 v4p4: stronger than just 'more than one unique action' — verify the
    empirical sample frequencies match softmax(logits) within a Chi-Square tolerance.

    A pure-argmax mutant fails this. An epsilon-greedy mutant with eps=0.1 also fails.
    An identity-temperature Categorical passes.
    """
    torch.manual_seed(0)
    net = PolicyNet(hidden=32, seed=0)
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    logits = net.forward(state)
    probs = torch.softmax(logits, dim=-1).detach().numpy()

    n_samples = 2000
    counts = [0] * ACTION_COUNT
    for _ in range(n_samples):
        action, _ = net.sample(state)
        counts[action] += 1
    empirical = [c / n_samples for c in counts]

    for i in range(ACTION_COUNT):
        sigma = (probs[i] * (1 - probs[i]) / n_samples) ** 0.5
        deviation = abs(empirical[i] - probs[i])
        assert deviation < 5 * sigma + 0.02, (
            f"action {i}: empirical {empirical[i]:.3f} vs softmax {probs[i]:.3f}, "
            f"deviation {deviation:.4f} > 5*sigma ({5 * sigma:.4f}) + 0.02 — sampler may not be Categorical"
        )


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
