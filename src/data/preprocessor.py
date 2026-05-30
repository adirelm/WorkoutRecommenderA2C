"""Data quality contract + muscle-group normalisation (brief §7.2.3).

Implements the rules defined in the PRD §1.5.0 data quality contract:

  (a) Drop rows where sets < 0 OR (sets > 0 AND reps == 0)  — true bad rows.
  (b) Time-encoded reps: if exercise_name (case-insensitive) contains any
      TIME_ENCODED_KEYWORDS  OR  reps < 0  OR  (reps > 60 AND sets <= 1),
      treat |reps| as seconds and convert via
      reps_equivalent = |reps| / seconds_per_rep.
  (c) Add a 'reps_equivalent' column; return (cleaned_df, DataQualityReport).
"""

from __future__ import annotations

import pandas as pd

from src.data.types import DataQualityReport

# Per PRD §1.5.0 data quality contract — keywords that imply time-based reps.
TIME_ENCODED_KEYWORDS: tuple[str, ...] = (
    "plank",
    "hold",
    "bridge",
    "wall-sit",
    "l-sit",
    "hollow-body",
    "wall sit",
)

# Static muscle-group dictionary used by normalise_muscle_group.
_MUSCLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "push": ("bench", "press", "push", "dip", "overhead", "shoulder", "tricep"),
    "pull": ("pull", "row", "chin", "curl", "lat", "deadlift"),
    "legs": ("squat", "lunge", "leg", "calf", "glute", "hip thrust", "step-up"),
    "core": ("plank", "crunch", "sit-up", "ab", "hollow", "bridge", "l-sit"),
    "cardio": (
        "burpee",
        "run",
        "sprint",
        "jog",
        "cycle",
        "row erg",
        "jump rope",
        "jumping",
        "mountain climber",
    ),
    "mobility": ("stretch", "mobility", "foam roll", "yoga", "warm-up", "warmup"),
}


# PRD §1.5.0 rule (b) heuristic: reps above this with single-set treated as seconds.
HIGH_REPS_SINGLE_SET_THRESHOLD = 60


def _is_time_encoded(exercise_name: str, sets: int, reps: int) -> bool:
    """Return True if the row should be treated as time-encoded (rule b)."""
    name_lc = (exercise_name or "").lower()
    if any(kw in name_lc for kw in TIME_ENCODED_KEYWORDS):
        return True
    if reps < 0:
        return True
    return reps > HIGH_REPS_SINGLE_SET_THRESHOLD and sets <= 1


def _reps_equivalent(reps: int, sets: int, exercise_name: str, seconds_per_rep: float) -> float:
    """Compute reps_equivalent for a single row."""
    if _is_time_encoded(exercise_name, sets, reps):
        return abs(reps) / seconds_per_rep
    return float(reps)


def apply_data_quality_contract(
    df: pd.DataFrame,
    seconds_per_rep: float = 3.0,
    drop_negative_with_log: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Apply the §7.2.3 data quality contract to a raw exercise-row DataFrame.

    Returns a tuple of (cleaned_df_with_reps_equivalent_column, report).
    """
    rows_in = len(df)
    work = df.copy().reset_index(drop=True)

    # Rule (a): identify true bad rows.
    negative_mask = (work["sets"] < 0) | ((work["sets"] > 0) & (work["reps"] == 0))
    negative_dropped = int(negative_mask.sum())

    if drop_negative_with_log:
        work = work.loc[~negative_mask].reset_index(drop=True)

    # Rule (b): compute reps_equivalent + count reclassifications.
    reclassified = 0
    reps_eq: list[float] = []
    for _, row in work.iterrows():
        if _is_time_encoded(str(row["exercise_name"]), int(row["sets"]), int(row["reps"])):
            reclassified += 1
        reps_eq.append(
            _reps_equivalent(
                int(row["reps"]),
                int(row["sets"]),
                str(row["exercise_name"]),
                seconds_per_rep,
            )
        )
    work["reps_equivalent"] = reps_eq

    report = DataQualityReport(
        rows_in=rows_in,
        rows_out=len(work),
        negative_volume_dropped=negative_dropped,
        time_encoded_reps_reclassified=reclassified,
        rest_days_inserted=0,  # rule (c) handled by the aggregator, not here
    )
    return work, report


def normalise_muscle_group(exercise_name: str) -> str:
    """Heuristic mapping exercise_name -> {push,pull,legs,core,cardio,mobility}.

    Falls back to 'push' when no keyword matches (callers should treat the
    result as best-effort — Phase 1 normalisation, not a learned classifier).
    """
    name_lc = (exercise_name or "").lower()
    # Order matters: check more-specific buckets (core, cardio, mobility) before
    # the broad push/pull/legs buckets so "Plank" doesn't get caught by "pull".
    priority = ("core", "cardio", "mobility", "legs", "pull", "push")
    for bucket in priority:
        for kw in _MUSCLE_KEYWORDS[bucket]:
            if kw in name_lc:
                return bucket
    return "push"
