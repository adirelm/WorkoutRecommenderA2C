"""Tests for LSTMEnvAdapter — bridges the frozen LSTM world model into
WorkoutEnv via the same interface as SyntheticTrainee (brief §7.3).

Phase-4 swap target: env.transition_provider goes from SyntheticTrainee
to LSTMEnvAdapter once the LSTM is trained + frozen.
"""

from __future__ import annotations

import dataclasses

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
    """After pushes the history stays at window_len AND the oldest entry is
    evicted (FIFO). A mutant that grew the buffer or overwrote the wrong slot
    would either fail the length check or leave the seeded fatigue=0.42 still
    visible at the buffer head."""
    window_len = 5
    adapter = LSTMEnvAdapter(_frozen_model(), window_len=window_len)
    # Seed with a state whose fatigue field is uniquely identifiable.
    sentinel = dataclasses.replace(State.initial(), fatigue=0.42)
    adapter.reset(sentinel)
    # Push window_len NEW states whose fatigue is 0.10, 0.20, ... so the
    # original sentinel must roll off the front of the buffer.
    s = sentinel
    for i in range(1, window_len + 1):
        s = dataclasses.replace(State.initial(), fatigue=float(i) * 0.1)
        _ = adapter.next_state(s, action_id=1, prescribed_volume=10.0)
    assert adapter.history_len == window_len
    # The sentinel (0.42) must no longer be at index 0 — buffer rolled.
    head_fatigue = adapter._state_history[0].fatigue
    assert head_fatigue != pytest.approx(0.42), f"sentinel still present at buffer head: {head_fatigue}"
    # And the latest pushed state must sit at the tail.
    assert adapter._state_history[-1].fatigue == pytest.approx(s.fatigue)


def test_lstm_must_be_frozen_at_construction_time() -> None:
    """Frozen-check happens at __init__ — fail fast before any rollout."""
    torch.manual_seed(0)
    unfrozen = LSTMWorldModel(hidden_size=16, num_layers=1, dropout=0.0)
    assert not unfrozen.is_frozen()
    with pytest.raises(RuntimeError, match="freeze"):
        LSTMEnvAdapter(unfrozen, window_len=7)


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
    byte-identical outputs over a ≥10-step rollout (not just one call) —
    a mutant introducing per-call nondeterminism would diverge before step 10."""
    a1 = LSTMEnvAdapter(_frozen_model(seed=42), window_len=7)
    a2 = LSTMEnvAdapter(_frozen_model(seed=42), window_len=7)
    s0 = State.initial()
    a1.reset(s0)
    a2.reset(s0)
    s_a, s_b = s0, s0
    rollout_len = 10
    for step in range(rollout_len):
        action = step % 7
        vol = float(step % 13)
        s_a = a1.next_state(s_a, action_id=action, prescribed_volume=vol)
        s_b = a2.next_state(s_b, action_id=action, prescribed_volume=vol)
        assert s_a.to_array().tolist() == s_b.to_array().tolist(), (
            f"determinism broke at step {step}: {s_a.to_array().tolist()} vs {s_b.to_array().tolist()}"
        )


def test_init_rejects_non_positive_window_len():
    """Covers lstm_env_adapter.py line 31 — ValueError on window_len <= 0."""
    model = LSTMWorldModel()
    model.freeze()
    with pytest.raises(ValueError, match="window_len must be positive"):
        LSTMEnvAdapter(model, window_len=0)


def test_init_rejects_unfrozen_model():
    """Covers __init__ frozen-check (fail-fast at construction)."""
    model = LSTMWorldModel()  # not frozen
    with pytest.raises(RuntimeError, match="freeze"):
        LSTMEnvAdapter(model, window_len=7)


def test_next_state_rejects_action_id_out_of_range():
    """Covers line 70 — ValueError on action_id outside [0, ACTION_COUNT)."""
    model = LSTMWorldModel()
    model.freeze()
    adapter = LSTMEnvAdapter(model, window_len=7)
    adapter.reset(State.initial())
    with pytest.raises(ValueError, match="action_id"):
        adapter.next_state(State.initial(), action_id=99)
    with pytest.raises(ValueError, match="action_id"):
        adapter.next_state(State.initial(), action_id=-1)
