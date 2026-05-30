"""Typed dataclasses for the data layer (Phase 1, brief §7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProgramRow:
    """One row of the program-level CSV (joins on `title`)."""

    title: str
    level: str
    goal: str
    equipment: str
    program_length_weeks: int
    time_per_workout_min: int


@dataclass(frozen=True)
class ExerciseRow:
    """One row of the exercise-level CSV (a single exercise in a session)."""

    title: str  # foreign key to ProgramRow.title
    week: int  # 1-indexed
    day: int  # 1-indexed within the week
    exercise_name: str
    muscle_group: str  # push / pull / legs / core / cardio / mobility (after normalisation)
    sets: int
    reps: int  # may encode seconds if exercise is time-based (see data_quality)
    intensity: float  # 0..1 normalised RPE proxy


@dataclass(frozen=True)
class DailyEntry:
    """Aggregated daily entry of the chosen program — the trajectory's t-th step.

    All values are *prescribed* (from the program), not observed: trainee
    state (fatigue/soreness/readiness) is layered on in Phase 2 (env).
    """

    day_in_cycle: int  # 0-indexed absolute day, e.g. 0..27 for a 4-week cycle
    week_index: int  # 0-indexed (day_in_cycle // 7)
    is_rest_day: bool  # True iff no exercises prescribed
    total_volume: float  # Σ sets·reps_equivalent over the day
    session_duration_min: int  # 0 on rest days
    muscle_distribution: dict[str, float] = field(default_factory=dict)
    # keys: 'push','pull','legs','core','cardio','mobility' -> share of total_volume in [0,1]


@dataclass(frozen=True)
class DataQualityReport:
    """Output of apply_data_quality_contract — auditable counts per rule."""

    rows_in: int
    rows_out: int
    negative_volume_dropped: int  # rule (a)
    time_encoded_reps_reclassified: int  # rule (b)
    rest_days_inserted: int  # rule (c)
