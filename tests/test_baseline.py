"""Direct tests for src/services/baseline.py — running-mean baseline."""

from __future__ import annotations

import pytest

from src.services.baseline import RunningMeanBaseline


def test_init_rejects_alpha_below_zero():
    with pytest.raises(ValueError, match="alpha must be in"):
        RunningMeanBaseline(alpha=-0.01)


def test_init_rejects_alpha_above_one():
    with pytest.raises(ValueError, match="alpha must be in"):
        RunningMeanBaseline(alpha=1.5)


def test_initial_value_is_zero_by_default():
    b = RunningMeanBaseline()
    assert b.value == 0.0
    assert b.updates == 0


def test_single_update_moves_toward_observation():
    b = RunningMeanBaseline(alpha=0.5)
    b.update(100.0)
    assert b.value == pytest.approx(50.0)  # (1-0.5)*0 + 0.5*100


def test_alpha_zero_is_no_op_baseline_stays_frozen():
    b = RunningMeanBaseline(alpha=0.0, initial=7.0)
    b.update(1000.0)
    assert b.value == 7.0
    assert b.updates == 1  # still counted


def test_alpha_one_jumps_to_observation():
    b = RunningMeanBaseline(alpha=1.0)
    b.update(42.0)
    assert b.value == 42.0


def test_reset_returns_to_initial():
    b = RunningMeanBaseline(alpha=0.5)
    b.update(100.0)
    b.update(200.0)
    assert b.updates == 2
    b.reset(initial=3.14)
    assert b.value == 3.14
    assert b.updates == 0


def test_many_updates_converge_to_true_mean():
    """Asymptotic property: feeding constant G → b → G."""
    b = RunningMeanBaseline(alpha=0.1)
    for _ in range(200):
        b.update(10.0)
    assert b.value == pytest.approx(10.0, abs=1e-3)
