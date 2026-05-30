"""Daily aggregation + rest-day insertion (brief eq. 13 and §7.2.4).

Input is assumed already cleaned by `apply_data_quality_contract`:
negative-volume rows dropped, time-encoded reps reclassified to an
equivalent rep count, muscle_group normalised to the canonical vocabulary.
"""

from __future__ import annotations

import pandas as pd

from src.data.types import DailyEntry

_DAYS_PER_WEEK = 7
_DEFAULT_SESSION_MIN = 45  # used when the row-level CSV does not carry a per-session duration

# Canonical muscle-group vocabulary (matches src/data/preprocessor.py buckets).
# Used so populated-day and rest-day DailyEntry instances share an identical key
# set, sorted lexicographically for deterministic downstream feature vectors.
KNOWN_MUSCLE_GROUPS: tuple[str, ...] = (
    "cardio",
    "core",
    "legs",
    "mobility",
    "pull",
    "push",
)


def _row_volume(group: pd.DataFrame) -> float:
    """Σ(sets · reps) for one (week, day) slice — eq. 13 numerator."""
    return float((group["sets"] * group["reps"]).sum())


def _muscle_distribution(group: pd.DataFrame, total_volume: float) -> dict[str, float]:
    """Per-muscle-group share of total_volume — canonical lexicographic key order.

    Always returns a dict whose keys are exactly KNOWN_MUSCLE_GROUPS (sorted),
    so rest-day and populated-day entries share an identical key set. Groups
    absent from the slice get a 0.0 share. Returns all-zeros when total_volume
    is non-positive.
    """
    shares: dict[str, float] = dict.fromkeys(sorted(KNOWN_MUSCLE_GROUPS), 0.0)
    if total_volume <= 0:
        return shares
    per_group = (group["sets"] * group["reps"]).groupby(group["muscle_group"]).sum()
    observed = {str(k): float(v) / total_volume for k, v in sorted(per_group.items())}
    for k, v in observed.items():
        shares[k] = v
    return shares


def daily_aggregate(exercises: pd.DataFrame) -> list[DailyEntry]:
    """Brief eq. 13: collapse exercise rows into one DailyEntry per (week, day).

    Returns an empty list for an empty frame. The output is sorted by day_in_cycle
    ascending so downstream rest-day insertion is deterministic.
    """
    if exercises.empty:
        return []
    entries: list[DailyEntry] = []
    grouped = exercises.groupby(["week", "day"], sort=True)
    for (week, day), group in grouped:
        total_volume = _row_volume(group)
        day_in_cycle = (int(week) - 1) * _DAYS_PER_WEEK + (int(day) - 1)
        entries.append(
            DailyEntry(
                day_in_cycle=day_in_cycle,
                week_index=day_in_cycle // _DAYS_PER_WEEK,
                is_rest_day=False,
                total_volume=total_volume,
                session_duration_min=_DEFAULT_SESSION_MIN,
                muscle_distribution=_muscle_distribution(group, total_volume),
            )
        )
    entries.sort(key=lambda e: e.day_in_cycle)
    return entries


def _rest_entry(day_in_cycle: int) -> DailyEntry:
    """Synthesised zero-volume rest day for §7.2.4 gap-filling.

    muscle_distribution uses the same key set as populated days (all zeros),
    so downstream feature vectors have a stable, deterministic schema.
    """
    return DailyEntry(
        day_in_cycle=day_in_cycle,
        week_index=day_in_cycle // _DAYS_PER_WEEK,
        is_rest_day=True,
        total_volume=0.0,
        session_duration_min=0,
        muscle_distribution=dict.fromkeys(sorted(KNOWN_MUSCLE_GROUPS), 0.0),
    )


def insert_rest_days(entries: list[DailyEntry], cycle_days: int) -> list[DailyEntry]:
    """Brief §7.2.4: emit exactly `cycle_days` entries indexed 0..cycle_days-1.

    Existing entries are preserved verbatim; any missing day_in_cycle slot is
    filled with a synthesised rest-day entry (total_volume = 0, duration = 0).
    """
    by_day = {e.day_in_cycle: e for e in entries}
    return [by_day.get(t, _rest_entry(t)) for t in range(cycle_days)]
