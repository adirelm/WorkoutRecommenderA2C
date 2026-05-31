"""End-to-end Phase 1 pipeline integration test (brief §7.2.3 + §7.2.4).

Loads the bundled CSV fixtures and exercises the full data flow:
read CSVs -> pick_program -> filter by title -> apply_data_quality_contract
-> daily_aggregate -> insert_rest_days, asserting cross-stage invariants.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.aggregator import daily_aggregate, insert_rest_days
from src.data.preprocessor import apply_data_quality_contract
from src.data.program_filter import pick_program

_FIX = Path(__file__).parent / "fixtures"
_CYCLE_DAYS = 14
_FLOAT_TOL = 1e-9


def test_phase1_pipeline_end_to_end_with_fixtures():
    programs_df = pd.read_csv(_FIX / "program_summary_sample.csv")
    exercises_df = pd.read_csv(_FIX / "fitness_exercises_sample.csv")

    chosen = pick_program(programs_df)
    assert chosen == "PHUL"

    phul_rows = exercises_df[exercises_df.title == chosen]
    cleaned, report = apply_data_quality_contract(phul_rows, seconds_per_rep=3.0)

    # Bad Set Row (sets=-2) is the rule-(a) drop.
    assert report.negative_volume_dropped >= 1
    # Plank Hold (reps=-60) + Wall Sit (reps=90, sets<=1) -> both time-encoded.
    assert report.time_encoded_reps_reclassified >= 2

    entries = daily_aggregate(cleaned)
    trajectory = insert_rest_days(entries, cycle_days=_CYCLE_DAYS)

    assert len(trajectory) == _CYCLE_DAYS
    assert sum(1 for e in trajectory if e.is_rest_day) >= 2
    assert sum(e.total_volume for e in trajectory) > 0

    # day_in_cycle indices form a contiguous 0..13 sequence.
    assert [e.day_in_cycle for e in trajectory] == list(range(_CYCLE_DAYS))

    # Muscle distribution shares sum to ~1.0 on every non-rest day.
    for e in trajectory:
        if not e.is_rest_day:
            assert sum(e.muscle_distribution.values()) == pytest.approx(1.0, abs=_FLOAT_TOL)
