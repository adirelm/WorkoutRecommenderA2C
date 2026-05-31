"""Tests for SDK shared types (Phase 6 / PRD §6)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.sdk.types import (
    LogbookHandle,
    PolicyHandle,
    WorkoutRecommendation,
    WorldModelHandle,
)


def test_logbook_handle_frozen():
    h = LogbookHandle(program_name="PPL", n_days=42, state_dim=12)
    with pytest.raises(FrozenInstanceError):
        h.n_days = 99  # type: ignore[misc]


def test_workout_recommendation_action_id_in_range():
    rec = WorkoutRecommendation(
        action_id=3,
        action_name="legs",
        probs=tuple([1.0 / 7] * 7),
        next_state_predicted=tuple([0.0] * 12),
        expected_reward=0.5,
    )
    assert 0 <= rec.action_id < 7
    assert len(rec.probs) == 7
    assert len(rec.next_state_predicted) == 12


def test_handles_round_trip():
    lb = LogbookHandle(program_name="UL", n_days=30, state_dim=12)
    wm = WorldModelHandle(n_params=12345, val_loss_final=0.042, epochs_trained=10)
    pol = PolicyHandle(algorithm="A2C", final_reward=1.23, episodes_trained=500)
    assert lb.program_name == "UL"
    assert wm.epochs_trained == 10
    assert pol.algorithm == "A2C"
