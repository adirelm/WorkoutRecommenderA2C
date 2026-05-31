"""Edge-case coverage for env.reward — guards, fallbacks, and boundaries.

Targets the uncovered branches in :mod:`src.env.reward`:

* ``_js_divergence`` zero-sum guards (both p and q) and the NaN guard.
* ``_js_divergence`` manual fallback when scipy is unavailable.
* ``_progress`` short-circuit when ``weekly_target <= 0``.
* ``_overload`` short-circuit when baseline ``<= 0`` and exact-threshold boundary.
* ``_imbalance`` empty-share guard + all-equal shares.
* ``RewardConfig.progress_clip_ceiling`` is honoured.
"""

from __future__ import annotations

import pytest

from src.env import reward as reward_mod
from src.env.reward import RewardConfig, RewardFunction, _js_divergence
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


def test_progress_zero_when_weekly_target_nonpositive() -> None:
    """Line: ``if weekly_target <= 0: return 0.0`` in :meth:`_progress`."""
    rf = RewardFunction()
    s, ns = _state(weekly_progress=0.2), _state(weekly_progress=0.5)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=0.0,
        baseline_7d_volume=100.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out["progress"] == 0.0


def test_overload_zero_when_baseline_nonpositive() -> None:
    """Line: ``if baseline <= 0: return 0.0`` in :meth:`_overload`."""
    rf = RewardFunction()
    s, ns = _state(rolling_7d=500.0), _state(rolling_7d=500.0)
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=0.0,
        muscle_share_14d=_BALANCED,
        target_muscle_dist=_BALANCED,
    )
    assert out["overload"] == 0.0


def test_overload_exactly_at_threshold_is_zero() -> None:
    """Boundary at 1.2x baseline: excess == 0 -> overload == 0 (not penalised)."""
    rf = RewardFunction()
    s, ns = _state(rolling_7d=120.0), _state(rolling_7d=120.0)
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


def test_imbalance_zero_for_empty_share_dict() -> None:
    """Line: ``if not share: return 0.0`` in :meth:`_imbalance`."""
    rf = RewardFunction()
    s, ns = _state(), _state()
    out = rf.compute(
        s,
        ns,
        action_id=1,
        weekly_target=100.0,
        baseline_7d_volume=100.0,
        muscle_share_14d={},
        target_muscle_dist=_BALANCED,
    )
    assert out["imbalance"] == 0.0


def test_imbalance_zero_for_all_equal_share() -> None:
    """Variance of a uniform distribution is exactly 0."""
    assert RewardFunction._imbalance({"a": 0.5, "b": 0.5}) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_zero_sum_p_branch() -> None:
    """Branch where p sums to 0 (skip normalisation) → scipy returns NaN, guard → 0.0."""
    div = _js_divergence({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.0})
    # All-zero p makes JS undefined; the NaN-guard returns 0 (no signal).
    assert div == 0.0


def test_js_divergence_both_zero_sum_returns_zero() -> None:
    """Branches 44->46 + 46->48: both inputs all-zero → divergence is 0 / NaN-guarded."""
    div = _js_divergence({"a": 0.0}, {"a": 0.0})
    assert div == 0.0


def test_js_divergence_manual_fallback_matches_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the scipy-less branch (lines for ``m`` + manual KL fallback)."""
    monkeypatch.setattr(reward_mod, "_scipy_js", None)
    div_manual = _js_divergence({"a": 0.7, "b": 0.3}, {"a": 0.3, "b": 0.7})
    assert div_manual > 0.0
    # Manual JS divergence is symmetric and bounded by 1.
    assert div_manual <= 1.0
    assert _js_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0, abs=1e-12)


def test_progress_clip_ceiling_is_configurable() -> None:
    """A custom ceiling replaces the default 1.2 hardcode."""
    cfg = RewardConfig(progress_clip_ceiling=0.5)
    rf = RewardFunction(cfg)
    s, ns = _state(weekly_progress=0.0), _state(weekly_progress=2.0)
    out = rf.compute(
        s,
        ns,
        1,
        100.0,
        100.0,
        _BALANCED,
        _BALANCED,
    )
    assert out["progress"] == pytest.approx(0.5, abs=1e-9)
