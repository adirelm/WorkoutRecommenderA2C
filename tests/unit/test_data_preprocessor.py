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


def test_normalise_muscle_group_mobility_and_fallback():
    """Cover preprocessor.py line 118 area: mobility bucket and the
    push fallback path (v4p1 branch-coverage gap)."""
    # Mobility bucket — previously uncovered.
    assert normalise_muscle_group("Foam Roll") == "mobility"
    assert normalise_muscle_group("Yoga Flow") == "mobility"
    # Fallback path: no keyword match -> defaults to 'push'.
    assert normalise_muscle_group("Some Unknown Move") == "push"
    # Empty / None-ish input still resolves deterministically (push fallback).
    assert normalise_muscle_group("") == "push"


def test_high_reps_single_set_threshold_boundary():
    """Cover preprocessor.py line 51 area: HIGH_REPS_SINGLE_SET_THRESHOLD
    boundary behaviour for rule (b) without relying on the keyword path."""
    # reps == 60 + sets == 1 -> NOT time-encoded (strict > threshold).
    df_at = pd.DataFrame([_row(exercise_name="Mystery Move", sets=1, reps=60)])
    cleaned, report = apply_data_quality_contract(df_at, seconds_per_rep=3.0)
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(60.0)
    assert report.time_encoded_reps_reclassified == 0

    # reps == 61 + sets == 2 -> NOT time-encoded (sets > 1).
    df_multi = pd.DataFrame([_row(exercise_name="Mystery Move", sets=2, reps=61)])
    cleaned, report = apply_data_quality_contract(df_multi, seconds_per_rep=3.0)
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(61.0)
    assert report.time_encoded_reps_reclassified == 0

    # reps == 61 + sets == 1 -> IS time-encoded (no keyword needed).
    df_hi = pd.DataFrame([_row(exercise_name="Mystery Move", sets=1, reps=61)])
    cleaned, report = apply_data_quality_contract(df_hi, seconds_per_rep=3.0)
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(61.0 / 3.0)
    assert report.time_encoded_reps_reclassified == 1


def test_negative_reps_no_keyword_treated_as_seconds():
    """Cover preprocessor.py line 60-61 branch in isolation: reps<0 path
    without the keyword match contaminating it."""
    df = pd.DataFrame([_row(exercise_name="Mystery Move", sets=1, reps=-30)])
    cleaned, report = apply_data_quality_contract(df, seconds_per_rep=3.0)
    assert cleaned.iloc[0]["reps_equivalent"] == pytest.approx(10.0)
    assert report.time_encoded_reps_reclassified == 1
