"""Tests for ActionMaskService (ADR-004, brief §7.6.1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.action_mask import ActionMaskService
from src.env.state import ACTION_COUNT, State

REST, PUSH, PULL, LEGS, FULLBODY, CONDITIONING, MOBILITY = 0, 1, 2, 3, 4, 5, 6


def _state(soreness_legs: float = 0.0, rolling_7d_volume: float = 0.0) -> State:
    return State(
        fatigue=0.0,
        soreness_push=0.0,
        soreness_pull=0.0,
        soreness_legs=soreness_legs,
        soreness_core=0.0,
        readiness=1.0,
        rolling_7d_volume=rolling_7d_volume,
        streak_days_trained=0,
        days_since_last_rest=0,
        muscle_balance_push_vs_pull=0.0,
        adherence_signal=0.0,
        weekly_progress=0.0,
    )


def test_rest_masked_after_three_rest_history():
    svc = ActionMaskService()
    mask = svc.mask(_state(), history=[REST, REST, REST])
    assert mask[REST] is np.False_ or mask[REST] == False  # noqa: E712


def test_rest_legal_after_mixed_history():
    svc = ActionMaskService()
    mask = svc.mask(_state(), history=[REST, PUSH, REST])
    assert bool(mask[REST]) is True


def test_legs_masked_when_soreness_legs_above_threshold():
    svc = ActionMaskService()
    mask = svc.mask(_state(soreness_legs=0.9), history=[])
    assert bool(mask[LEGS]) is False


def test_legs_legal_when_soreness_below_threshold():
    svc = ActionMaskService()
    mask = svc.mask(_state(soreness_legs=0.5), history=[])
    assert bool(mask[LEGS]) is True


def test_mobility_always_legal():
    svc = ActionMaskService()
    for sore in (0.0, 0.5, 0.99):
        for hist in ([], [REST, REST, REST], [PUSH, PULL, LEGS]):
            mask = svc.mask(_state(soreness_legs=sore, rolling_7d_volume=99.0), history=hist)
            assert bool(mask[MOBILITY]) is True


def test_conditioning_masked_when_overload_exceeded():
    svc = ActionMaskService(conditioning_overload_threshold=1.0)
    # rolling_7d_volume used as proxy for overload signal (>1.0 → masked)
    mask = svc.mask(_state(rolling_7d_volume=1.5), history=[])
    assert bool(mask[CONDITIONING]) is False


def test_apply_to_logits_sets_minus_inf_for_masked():
    svc = ActionMaskService()
    logits = np.zeros(ACTION_COUNT, dtype=np.float32)
    mask = np.array([False, True, True, False, True, True, True])
    masked = svc.apply_to_logits(logits, mask)
    assert np.isneginf(masked[0])
    assert np.isneginf(masked[3])


def test_apply_to_logits_preserves_legal_positions():
    svc = ActionMaskService()
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    mask = np.array([False, True, True, True, True, True, True])
    masked = svc.apply_to_logits(logits, mask)
    np.testing.assert_array_equal(masked[1:], logits[1:])


def test_softmax_after_mask_gives_zero_prob_to_masked():
    svc = ActionMaskService()
    logits = np.ones(ACTION_COUNT, dtype=np.float32)
    mask = np.array([False, True, True, False, True, True, True])
    masked = svc.apply_to_logits(logits, mask)
    e = np.exp(masked - np.max(masked))
    probs = e / e.sum()
    assert probs[0] == pytest.approx(0.0)
    assert probs[3] == pytest.approx(0.0)
    assert probs.sum() == pytest.approx(1.0)
