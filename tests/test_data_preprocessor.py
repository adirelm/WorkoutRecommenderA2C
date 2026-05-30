"""Tests for src/data/preprocessor.py (brief §7.2.3 data quality contract)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessor import (
    apply_data_quality_contract,
    normalise_muscle_group,
)


def _row(**overrides) -> dict:
    base = {
        "title": "P1",
        "week": 1,
        "day": 1,
        "exercise_name": "Bench Press",
        "muscle_group": "push",
        "sets": 4,
        "reps": 10,
        "intensity": 0.7,
    }
    base.update(overrides)
    return base


def test_drop_negative_sets():
    df = pd.DataFrame([_row(sets=-2), _row()])
    cleaned, report = apply_data_quality_contract(df)
    assert len(cleaned) == 1
    assert report.negative_volume_dropped == 1


def test_plank_negative_reps_treated_as_seconds():
    df = pd.DataFrame([_row(exercise_name="Plank Hold", sets=1, reps=-60)])
    cleaned, report = apply_data_quality_contract(df, seconds_per_rep=3.0)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(20.0)
    assert report.time_encoded_reps_reclassified == 1


def test_wall_sit_high_reps_treated_as_seconds():
    df = pd.DataFrame([_row(exercise_name="Wall Sit", sets=1, reps=90)])
    cleaned, report = apply_data_quality_contract(df, seconds_per_rep=3.0)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(30.0)
    assert report.time_encoded_reps_reclassified == 1


def test_normal_row_unchanged():
    df = pd.DataFrame([_row(sets=4, reps=10)])
    cleaned, _ = apply_data_quality_contract(df)
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(10.0)
    assert cleaned.iloc[0]["sets"] == 4


def test_report_counts_match_input():
    df = pd.DataFrame([_row(), _row(sets=-1), _row()])
    cleaned, report = apply_data_quality_contract(df)
    assert report.rows_in == 3
    assert report.rows_out == len(cleaned)
    assert report.rows_out == report.rows_in - report.negative_volume_dropped


def test_drop_negative_with_log_false():
    df = pd.DataFrame([_row(sets=-3), _row()])
    cleaned, report = apply_data_quality_contract(df, drop_negative_with_log=False)
    assert len(cleaned) == 2  # negative kept
    assert report.negative_volume_dropped == 1  # still flagged


def test_normalise_muscle_group_handles_known_names():
    assert normalise_muscle_group("Bench Press") == "push"
    assert normalise_muscle_group("Pull-up") == "pull"
    assert normalise_muscle_group("Squat") == "legs"
    assert normalise_muscle_group("Plank") == "core"
    assert normalise_muscle_group("Burpee") == "cardio"
