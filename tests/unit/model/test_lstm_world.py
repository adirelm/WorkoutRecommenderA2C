"""Tests for src/model/lstm_world.py — LSTM transition model f_φ(s_t, a_t, h_t) → ŝ_{t+1}.

Edge-case ValueError + 48h-recovery tests live in test_lstm_world_extra.py
so this file stays ≤150 LOC (CLAUDE.md §1).
"""

from __future__ import annotations

import torch

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.lstm_world import LSTMWorldModel


def _make_inputs(batch: int = 8, seq: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Random state_seq + valid action_seq tensors for the model."""
    state_seq = torch.randn(batch, seq, STATE_DIM, dtype=torch.float32)
    action_seq = torch.randint(low=0, high=ACTION_COUNT, size=(batch, seq), dtype=torch.long)
    return state_seq, action_seq


def test_forward_output_shape():
    model = LSTMWorldModel(hidden_size=64, num_layers=1)
    state_seq, action_seq = _make_inputs(batch=8, seq=7)
    out = model(state_seq, action_seq)
    assert out.shape == (8, STATE_DIM)


def test_forward_one_hot_action_concat():
    """Action ids must influence output (mutation-killer: same state, different
    actions → different predictions, proving action signal is not dropped)."""
    torch.manual_seed(0)
    model = LSTMWorldModel(hidden_size=32, num_layers=2)
    state_seq, action_seq = _make_inputs(batch=4, seq=5)
    out = model(state_seq, action_seq)
    assert out.shape == (4, STATE_DIM)
    # action ids must be respected — every valid id should work
    for a in range(ACTION_COUNT):
        action_seq_const = torch.full((2, 3), a, dtype=torch.long)
        s = torch.randn(2, 3, STATE_DIM, dtype=torch.float32)
        out2 = model(s, action_seq_const)
        assert out2.shape == (2, STATE_DIM)
    # Mutation-killer: same state, different action ids → different outputs.
    same_state = torch.randn(1, 4, STATE_DIM, dtype=torch.float32)
    action_a = torch.zeros(1, 4, dtype=torch.long)
    action_b = torch.full((1, 4), ACTION_COUNT - 1, dtype=torch.long)
    out_a = model(same_state, action_a)
    out_b = model(same_state, action_b)
    assert not torch.allclose(out_a, out_b), (
        "outputs identical for different actions — action signal is being dropped"
    )


def test_forward_dtype_float32():
    model = LSTMWorldModel()
    state_seq, action_seq = _make_inputs()
    out = model(state_seq, action_seq)
    assert out.dtype == torch.float32


def test_freeze_sets_requires_grad_false():
    model = LSTMWorldModel()
    # Sanity — before freeze, params are trainable
    assert any(p.requires_grad for p in model.parameters())
    model.freeze()
    for p in model.parameters():
        assert p.requires_grad is False


def test_is_frozen_reports_correctly():
    model = LSTMWorldModel()
    assert model.is_frozen() is False
    model.freeze()
    assert model.is_frozen() is True


def test_num_parameters_nonzero():
    model = LSTMWorldModel(hidden_size=64, num_layers=1)
    assert model.num_parameters > 0
    # bigger model should have more params
    bigger = LSTMWorldModel(hidden_size=128, num_layers=2)
    assert bigger.num_parameters > model.num_parameters


def test_seeded_init_deterministic():
    torch.manual_seed(42)
    m1 = LSTMWorldModel(hidden_size=64, num_layers=1)
    torch.manual_seed(42)
    m2 = LSTMWorldModel(hidden_size=64, num_layers=1)
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        torch.testing.assert_close(sd1[k], sd2[k])


def test_gradient_flows_when_not_frozen():
    model = LSTMWorldModel()
    state_seq, action_seq = _make_inputs(batch=4, seq=5)
    target = torch.randn(4, STATE_DIM, dtype=torch.float32)
    out = model(state_seq, action_seq)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    # At least one parameter (the linear head, definitely) must have non-None grad
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)
    # Mutation-killer: a detached-but-allocated grad path would have all-zero
    # grads. At least one parameter must have non-zero gradient signal.
    assert any(g.abs().sum().item() > 0.0 for g in grads), (
        "all gradients are zero — autograd graph appears detached"
    )
