"""Tests for src/model/dataset.py — sliding-window builder + chronological split."""

from __future__ import annotations

import numpy as np

from src.env.state import STATE_DIM, State
from src.model.dataset import build_windows, split_train_val


def _make_trajectory(length: int) -> list[tuple[State, int, State]]:
    """Build a deterministic trajectory of given length using State.initial() variants."""
    traj: list[tuple[State, int, State]] = []
    for i in range(length):
        s = State(
            fatigue=i * 0.01,
            soreness_push=0.0,
            soreness_pull=0.0,
            soreness_legs=0.0,
            soreness_core=0.0,
            readiness=1.0,
            rolling_7d_volume=float(i),
            streak_days_trained=i,
            days_since_last_rest=i,
            muscle_balance_push_vs_pull=0.0,
            adherence_signal=0.0,
            weekly_progress=0.0,
        )
        ns = State(
            fatigue=(i + 1) * 0.01,
            soreness_push=0.0,
            soreness_pull=0.0,
            soreness_legs=0.0,
            soreness_core=0.0,
            readiness=1.0,
            rolling_7d_volume=float(i + 1),
            streak_days_trained=i + 1,
            days_since_last_rest=i + 1,
            muscle_balance_push_vs_pull=0.0,
            adherence_signal=0.0,
            weekly_progress=0.0,
        )
        traj.append((s, i % 7, ns))
    return traj


def test_build_windows_count_equals_T_minus_window_plus_1():  # noqa: N802 — T matches docstring notation
    traj = _make_trajectory(14)
    windows = build_windows(traj, window_len=7)
    assert len(windows) == 14 - 7 + 1 == 8


def test_window_state_seq_shape_matches_window_len():
    traj = _make_trajectory(14)
    windows = build_windows(traj, window_len=7)
    assert windows[0].state_seq.shape == (7, STATE_DIM)
    assert windows[0].action_seq.shape == (7,)
    assert windows[0].next_state.shape == (STATE_DIM,)


def test_window_next_state_matches_trajectory():
    traj = _make_trajectory(14)
    window_len = 7
    windows = build_windows(traj, window_len=window_len)
    for i, w in enumerate(windows):
        # window i corresponds to trajectory index (i + window_len - 1)
        expected_next = traj[i + window_len - 1][2].to_array()
        np.testing.assert_array_equal(w.next_state, expected_next)


def test_split_train_val_chronological():
    traj = _make_trajectory(20)
    windows = build_windows(traj, window_len=7)  # 14 windows
    train, val = split_train_val(windows, val_days=7)
    assert len(val) == 7
    assert len(train) == len(windows) - 7
    # last 7 of `windows` must equal `val` (chronological tail)
    for w_full, w_val in zip(windows[-7:], val, strict=True):
        np.testing.assert_array_equal(w_full.next_state, w_val.next_state)


def test_window_too_short_returns_empty():
    traj = _make_trajectory(3)
    assert build_windows(traj, window_len=7) == []


def test_build_windows_rejects_non_positive_window_len():
    """Covers dataset.py line 27 — ValueError on window_len <= 0."""
    import pytest

    traj = _make_trajectory(10)
    with pytest.raises(ValueError, match="window_len must be positive"):
        build_windows(traj, window_len=0)
    with pytest.raises(ValueError, match="window_len must be positive"):
        build_windows(traj, window_len=-1)


def test_split_train_val_zero_val_days_puts_everything_in_train():
    """Covers dataset.py line 62 — val_days<=0 path."""
    traj = _make_trajectory(20)
    windows = build_windows(traj, window_len=7)
    train, val = split_train_val(windows, val_days=0)
    assert len(train) == len(windows)
    assert val == []


def test_split_train_val_val_days_at_or_beyond_total_puts_everything_in_val():
    """Covers dataset.py line 64 — val_days >= len(windows) path."""
    traj = _make_trajectory(10)
    windows = build_windows(traj, window_len=7)  # 4 windows
    train, val = split_train_val(windows, val_days=99)
    assert train == []
    assert len(val) == len(windows)
