"""Edge-case tests for src/data/aggregator.py — keeps the main test file under 150 LOC.

Covers:
  * test_rest_day_muscle_distribution_matches_populated_key_set (v5p1 finding)
  * test_daily_aggregate_zero_volume_slice_returns_all_zero_shares (v4p1 line-25 coverage)
"""

from __future__ import annotations

import pandas as pd

from src.data.aggregator import KNOWN_MUSCLE_GROUPS, daily_aggregate, insert_rest_days


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


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
