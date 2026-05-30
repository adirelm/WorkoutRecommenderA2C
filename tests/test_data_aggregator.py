"""Tests for src/data/aggregator.py (brief eq. 13 + §7.2.4 rest-day insertion)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.aggregator import KNOWN_MUSCLE_GROUPS, daily_aggregate, insert_rest_days
from src.data.types import DailyEntry


def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal exercise DataFrame; caller supplies week/day/sets/reps/muscle_group."""
    return pd.DataFrame(rows)


def test_daily_aggregate_sums_sets_times_reps():
    df = _df(
        [
            {
                "week": 1,
                "day": 1,
                "muscle_group": "push",
                "sets": 3,
                "reps": 10,
                "exercise_name": "bench",
                "intensity": 0.7,
                "title": "p",
            },
            {
                "week": 1,
                "day": 1,
                "muscle_group": "push",
                "sets": 4,
                "reps": 8,
                "exercise_name": "ohp",
                "intensity": 0.7,
                "title": "p",
            },
        ]
    )
    entries = daily_aggregate(df)
    assert len(entries) == 1
    assert entries[0].total_volume == pytest.approx(3 * 10 + 4 * 8)


def test_daily_aggregate_muscle_distribution_shares_sum_to_1():
    df = _df(
        [
            {
                "week": 1,
                "day": 2,
                "muscle_group": "push",
                "sets": 3,
                "reps": 10,
                "exercise_name": "bench",
                "intensity": 0.7,
                "title": "p",
            },
            {
                "week": 1,
                "day": 2,
                "muscle_group": "pull",
                "sets": 3,
                "reps": 10,
                "exercise_name": "row",
                "intensity": 0.7,
                "title": "p",
            },
            {
                "week": 1,
                "day": 2,
                "muscle_group": "legs",
                "sets": 5,
                "reps": 5,
                "exercise_name": "squat",
                "intensity": 0.8,
                "title": "p",
            },
        ]
    )
    entries = daily_aggregate(df)
    assert len(entries) == 1
    shares = entries[0].muscle_distribution
    assert sum(shares.values()) == pytest.approx(1.0)
    # Canonical key set: all KNOWN_MUSCLE_GROUPS, absent buckets are 0.
    assert set(shares.keys()) == set(KNOWN_MUSCLE_GROUPS)
    assert {k for k, v in shares.items() if v > 0} == {"push", "pull", "legs"}
    # Keys are sorted lexicographically (deterministic feature ordering).
    assert list(shares.keys()) == sorted(KNOWN_MUSCLE_GROUPS)


def test_daily_aggregate_empty_input_returns_empty_list():
    df = pd.DataFrame(
        columns=["week", "day", "muscle_group", "sets", "reps", "exercise_name", "intensity", "title"]
    )
    assert daily_aggregate(df) == []


def _entry(day_in_cycle: int, total_volume: float = 100.0) -> DailyEntry:
    return DailyEntry(
        day_in_cycle=day_in_cycle,
        week_index=day_in_cycle // 7,
        is_rest_day=False,
        total_volume=total_volume,
        session_duration_min=45,
        muscle_distribution={"push": 1.0},
    )


def test_insert_rest_days_fills_gap():
    entries = [_entry(0), _entry(1), _entry(2), _entry(5)]
    out = insert_rest_days(entries, cycle_days=7)
    assert len(out) == 7
    rest_days = [e.day_in_cycle for e in out if e.is_rest_day]
    assert rest_days == [3, 4, 6]


def test_insert_rest_days_total_volume_zero_on_rest():
    entries = [_entry(0)]
    out = insert_rest_days(entries, cycle_days=3)
    for e in out:
        if e.is_rest_day:
            assert e.total_volume == 0
            assert e.session_duration_min == 0


def test_rest_day_muscle_distribution_matches_populated_key_set():
    """Rest-day entries must expose the same muscle-group keys as populated days
    coming from daily_aggregate, so downstream feature vectors have a
    deterministic schema (v5p1 finding)."""
    df = _df(
        [
            {
                "week": 1,
                "day": 1,
                "muscle_group": "push",
                "sets": 3,
                "reps": 10,
                "exercise_name": "bench",
                "intensity": 0.7,
                "title": "p",
            },
        ]
    )
    populated = daily_aggregate(df)
    out = insert_rest_days(populated, cycle_days=3)
    populated_keys = set(populated[0].muscle_distribution.keys())
    assert populated_keys == set(KNOWN_MUSCLE_GROUPS)
    for e in out:
        assert set(e.muscle_distribution.keys()) == populated_keys
        # Populated and rest entries both use the canonical sorted key list.
        assert list(e.muscle_distribution.keys()) == sorted(KNOWN_MUSCLE_GROUPS)
        if e.is_rest_day:
            assert all(v == 0.0 for v in e.muscle_distribution.values())


def test_daily_aggregate_zero_volume_slice_returns_all_zero_shares():
    """Cover aggregator.py line 25 branch: total_volume <= 0 -> all-zero shares
    with the canonical key set (not an empty dict)."""
    df = _df(
        [
            {
                "week": 1,
                "day": 1,
                "muscle_group": "push",
                "sets": 0,
                "reps": 10,
                "exercise_name": "skipped",
                "intensity": 0.0,
                "title": "p",
            },
            {
                "week": 1,
                "day": 1,
                "muscle_group": "pull",
                "sets": 3,
                "reps": 0,
                "exercise_name": "skipped",
                "intensity": 0.0,
                "title": "p",
            },
        ]
    )
    entries = daily_aggregate(df)
    assert len(entries) == 1
    shares = entries[0].muscle_distribution
    assert set(shares.keys()) == set(KNOWN_MUSCLE_GROUPS)
    assert all(v == 0.0 for v in shares.values())
    assert entries[0].total_volume == 0.0


def test_insert_rest_days_preserves_existing_entries():
    entries = [_entry(0, total_volume=42.0), _entry(2, total_volume=99.0)]
    out = insert_rest_days(entries, cycle_days=4)
    by_day = {e.day_in_cycle: e for e in out}
    assert by_day[0].total_volume == 42.0
    assert by_day[0].is_rest_day is False
    assert by_day[2].total_volume == 99.0
    assert by_day[2].is_rest_day is False
