"""Tests for env.reward — eq. 15 reward decomposition (ADR-003, brief §7.4)."""

from __future__ import annotations

import pytest

from src.env.reward import RewardConfig, RewardFunction
from src.env.state import State


def _state(rolling_7d: float = 0.0, weekly_progress: float = 0.0) -> State:
    base = State.initial()
    return State(
        fatigue=base.fatigue,
        soreness_push=base.soreness_push,
        soreness_pull=base.soreness_pull,
        soreness_legs=base.soreness_legs,
        soreness_core=base.soreness_core,
        readiness=base.readiness,
        rolling_7d_volume=rolling_7d,
        streak_days_trained=base.streak_days_trained,
        days_since_last_rest=base.days_since_last_rest,
        muscle_balance_push_vs_pull=base.muscle_balance_push_vs_pull,
        adherence_signal=base.adherence_signal,
        weekly_progress=weekly_progress,
    )


_BALANCED = {"push": 0.25, "pull": 0.25, "legs": 0.25, "core": 0.25}
_SKEWED = {"push": 0.85, "pull": 0.05, "legs": 0.05, "core": 0.05}


def test_default_config_values_match_adr_003() -> None:
    cfg = RewardConfig()
    assert cfg.lambda_1 == 2.0
    assert cfg.lambda_2 == 1.0
    assert cfg.w_progress == 0.7
    assert cfg.w_variety == 0.3


def test_gain_progress_within_target_is_positive() -> None:
    rf = RewardFunction()
    s, ns = _state(weekly_progress=0.2), _state(weekly_progress=0.5)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out["progress"] > 0.0
    assert out["gain"] > 0.0


def test_overload_zero_below_threshold() -> None:
    rf = RewardFunction()
    # rolling_7d = 110, baseline = 100 → ratio 1.1 < 1.2
    s, ns = _state(rolling_7d=110.0), _state(rolling_7d=110.0)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out["overload"] == 0.0


def test_overload_superlinear_above_threshold() -> None:
    rf = RewardFunction()
    # rolling_7d = 150, baseline = 100 → 50% over baseline, 30% over threshold (1.2)
    s, ns = _state(rolling_7d=150.0), _state(rolling_7d=150.0)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    # exponent 1.5 on (0.3) → 0.164... — strictly positive and superlinear shape verified
    assert out["overload"] > 0.0
    # Confirm super-linearity: doubling the over-threshold excess more than doubles overload.
    s2, ns2 = _state(rolling_7d=180.0), _state(rolling_7d=180.0)  # 60% over baseline
    out2 = rf.compute(
        s2,
        ns2,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out2["overload"] > 2.0 * out["overload"]


def test_imbalance_zero_when_distribution_matches_target() -> None:
    rf = RewardFunction()
    s, ns = _state(), _state()
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out["imbalance"] == pytest.approx(0.0, abs=1e-9)


def test_imbalance_positive_when_distribution_skewed() -> None:
    rf = RewardFunction()
    s, ns = _state(), _state()
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_SKEWED,
        target_muscle_dist=_BALANCED,
    )
    assert out["imbalance"] > 0.0


def test_total_reward_eq_15_decomposition() -> None:
    rf = RewardFunction()
    s, ns = _state(rolling_7d=150.0, weekly_progress=0.2), _state(rolling_7d=150.0, weekly_progress=0.5)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_SKEWED,
        target_muscle_dist=_BALANCED,
    )
    cfg = RewardConfig()
    expected = out["gain"] - cfg.lambda_1 * out["overload"] - cfg.lambda_2 * out["imbalance"]
    assert out["reward"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
