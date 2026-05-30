"""Tests for env.state — State dataclass + action space (ADR-002, brief §7.3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.env.state import (
    ACTION_COUNT,
    ACTION_NAMES,
    STATE_CHANNEL_NAMES,
    STATE_DIM,
    State,
)


def test_state_initial_returns_rested_baseline() -> None:
    s = State.initial()
    assert s.fatigue == 0.0
    assert s.soreness_push == 0.0
    assert s.soreness_pull == 0.0
    assert s.soreness_legs == 0.0
    assert s.soreness_core == 0.0
    assert s.readiness == 1.0
    assert s.streak_days_trained == 0


def test_state_to_array_dim_12() -> None:
    arr = State.initial().to_array()
    assert arr.shape == (STATE_DIM,) == (12,)
    assert arr.dtype == np.float32


def test_state_to_array_order_matches_channel_names() -> None:
    s = State(
        fatigue=0.1,
        soreness_push=0.2,
        soreness_pull=0.3,
        soreness_legs=0.4,
        soreness_core=0.5,
        readiness=0.6,
        rolling_7d_volume=7.0,
        streak_days_trained=8,
        days_since_last_rest=9,
        muscle_balance_push_vs_pull=-0.10,
        adherence_signal=0.11,
        weekly_progress=0.12,
    )
    arr = s.to_array()
    expected = {
        "fatigue": 0.1,
        "soreness_push": 0.2,
        "soreness_pull": 0.3,
        "soreness_legs": 0.4,
        "soreness_core": 0.5,
        "readiness": 0.6,
        "rolling_7d_volume": 7.0,
        "streak_days_trained": 8.0,
        "days_since_last_rest": 9.0,
        "muscle_balance_push_vs_pull": -0.10,
        "adherence_signal": 0.11,
        "weekly_progress": 0.12,
    }
    for i, name in enumerate(STATE_CHANNEL_NAMES):
        assert arr[i] == pytest.approx(expected[name], rel=1e-5), f"Channel {i} ({name}) mismatch"


def test_state_is_frozen() -> None:
    s = State.initial()
    with pytest.raises(FrozenInstanceError):
        s.fatigue = 0.5  # type: ignore[misc]


def test_action_count_seven() -> None:
    assert ACTION_COUNT == 7
    assert len(ACTION_NAMES) == 7
