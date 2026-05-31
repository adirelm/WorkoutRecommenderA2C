"""Tests for LSTMEnvAdapter — bridges the frozen LSTM world model into
WorkoutEnv via the same interface as SyntheticTrainee (brief §7.3).

Phase-4 swap target: env.transition_provider goes from SyntheticTrainee
to LSTMEnvAdapter once the LSTM is trained + frozen.
"""

from __future__ import annotations

import pytest
import torch

from src.env.state import State
from src.model.lstm_env_adapter import LSTMEnvAdapter
from src.model.lstm_world import LSTMWorldModel


def _frozen_model(seed: int = 0, hidden: int = 16) -> LSTMWorldModel:
    torch.manual_seed(seed)
    m = LSTMWorldModel(hidden_size=hidden, num_layers=1, dropout=0.0)
    m.freeze()
    m.eval()
    return m


def test_reset_initializes_history() -> None:
    adapter = LSTMEnvAdapter(_frozen_model(), window_len=7)
    s0 = State.initial()
    adapter.reset(s0)
    assert adapter.history_len == 7


def test_next_state_keeps_rolling_window() -> None:
    """After one step the history stays at window_len (rolling buffer)."""
    adapter = LSTMEnvAdapter(_frozen_model(), window_len=5)
    s0 = State.initial()
    adapter.reset(s0)
    _ = adapter.next_state(s0, action_id=1, prescribed_volume=10.0)
    assert adapter.history_len == 5


def test_lstm_must_be_frozen_at_call_time() -> None:
    torch.manual_seed(0)
    unfrozen = LSTMWorldModel(hidden_size=16, num_layers=1, dropout=0.0)
    assert not unfrozen.is_frozen()
    adapter = LSTMEnvAdapter(unfrozen, window_len=7)
    adapter.reset(State.initial())
    with pytest.raises(RuntimeError, match="frozen"):
        adapter.next_state(State.initial(), action_id=0, prescribed_volume=0.0)


def test_next_state_returns_state_dataclass() -> None:
    adapter = LSTMEnvAdapter(_frozen_model(), window_len=7)
    s0 = State.initial()
    adapter.reset(s0)
    out = adapter.next_state(s0, action_id=2, prescribed_volume=8.0)
    assert isinstance(out, State)


def test_output_state_values_in_valid_ranges() -> None:
    """Soreness / fatigue / readiness clamped to [0, 1]; rolling_7d_volume ≥ 0;
    streak / days_since_last_rest non-negative ints."""
    adapter = LSTMEnvAdapter(_frozen_model(seed=3), window_len=7)
    s = State.initial()
    adapter.reset(s)
    for step in range(30):
        s = adapter.next_state(s, action_id=step % 7, prescribed_volume=float(step % 12))
        assert 0.0 <= s.fatigue <= 1.0
        assert 0.0 <= s.soreness_push <= 1.0
        assert 0.0 <= s.soreness_pull <= 1.0
        assert 0.0 <= s.soreness_legs <= 1.0
        assert 0.0 <= s.soreness_core <= 1.0
        assert 0.0 <= s.readiness <= 1.0
        assert s.rolling_7d_volume >= 0.0
        assert isinstance(s.streak_days_trained, int)
        assert isinstance(s.days_since_last_rest, int)
        assert s.streak_days_trained >= 0
        assert s.days_since_last_rest >= 0
        assert -1.0 <= s.muscle_balance_push_vs_pull <= 1.0


def test_seeded_lstm_gives_deterministic_outputs() -> None:
    """Two adapters wrapping identically-seeded frozen models produce
    bit-identical outputs for the same input sequence."""
    a1 = LSTMEnvAdapter(_frozen_model(seed=42), window_len=7)
    a2 = LSTMEnvAdapter(_frozen_model(seed=42), window_len=7)
    s0 = State.initial()
    a1.reset(s0)
    a2.reset(s0)
    out1 = a1.next_state(s0, action_id=4, prescribed_volume=12.0)
    out2 = a2.next_state(s0, action_id=4, prescribed_volume=12.0)
    assert out1.to_array().tolist() == out2.to_array().tolist()
