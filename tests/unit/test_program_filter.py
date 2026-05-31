"""Tests for src/data/program_filter.py — brief §7.2.4 selection criteria."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.program_filter import ProgramNotFoundError, pick_program


def _sample_df() -> pd.DataFrame:
    """Mirror of tests/fixtures/program_summary_sample.csv."""
    return pd.DataFrame(
        [
            {
                "title": "PHUL",
                "level": "intermediate",
                "goal": "strength+hypertrophy",
                "equipment": "Full Gym",
                "program_length": 12,
                "time_per_workout": 60,
            },
            {
                "title": "GZCLP",
                "level": "beginner",
                "goal": "strength",
                "equipment": "Full Gym",
                "program_length": 12,
                "time_per_workout": 60,
            },
            {
                "title": "nSuns 5/3/1",
                "level": "intermediate",
                "goal": "strength",
                "equipment": "Full Gym",
                "program_length": 10,
                "time_per_workout": 90,
            },
            {
                "title": "Beginner Bodyweight",
                "level": "beginner",
                "goal": "general",
                "equipment": "Bodyweight",
                "program_length": 4,
                "time_per_workout": 30,
            },
        ]
    )


def test_picks_primary_when_present_and_valid() -> None:
    """PHUL with weeks=12, equip=Full Gym, time=60 -> 'PHUL'."""
    assert pick_program(_sample_df()) == "PHUL"


def test_falls_back_when_primary_missing() -> None:
    """Drop PHUL from df -> returns 'GZCLP'."""
    df = _sample_df()
    df = df[df["title"] != "PHUL"].reset_index(drop=True)
    assert pick_program(df) == "GZCLP"


def test_falls_back_when_primary_fails_criteria() -> None:
    """PHUL but program_length=4 (< min_weeks=8) -> 'GZCLP'."""
    df = _sample_df()
    df.loc[df["title"] == "PHUL", "program_length"] = 4
    assert pick_program(df) == "GZCLP"


def test_raises_when_all_fail() -> None:
    """Only Beginner Bodyweight available -> ProgramNotFoundError."""
    df = _sample_df()
    df = df[df["title"] == "Beginner Bodyweight"].reset_index(drop=True)
    with pytest.raises(ProgramNotFoundError):
        pick_program(df)


def test_error_message_lists_failed_criteria() -> None:
    """Each tried program must be named in the exception message."""
    df = _sample_df()
    df = df[df["title"] == "Beginner Bodyweight"].reset_index(drop=True)
    with pytest.raises(ProgramNotFoundError) as exc_info:
        pick_program(df)
    msg = str(exc_info.value)
    for name in ("PHUL", "GZCLP", "nSuns 5/3/1"):
        assert name in msg, f"Expected '{name}' in error message, got: {msg}"


def test_equipment_mismatch_recorded_in_failure_report() -> None:
    """Branch: row['equipment'] != equipment -> reason appended (line 33-35)."""
    df = _sample_df()
    df.loc[df["title"].isin(["PHUL", "GZCLP", "nSuns 5/3/1"]), "equipment"] = "Bodyweight"
    with pytest.raises(ProgramNotFoundError) as exc_info:
        pick_program(df, equipment="Full Gym")
    msg = str(exc_info.value)
    assert "equipment=" in msg
    assert "'Bodyweight'" in msg
    assert "!= required 'Full Gym'" in msg


def test_time_below_min_recorded_in_failure_report() -> None:
    """Branch: minutes < min_minutes -> reason appended (line 42-45)."""
    df = _sample_df()
    df.loc[df["title"].isin(["PHUL", "GZCLP", "nSuns 5/3/1"]), "time_per_workout"] = 30
    with pytest.raises(ProgramNotFoundError) as exc_info:
        pick_program(df, min_minutes=45, max_minutes=120)
    msg = str(exc_info.value)
    assert "time_per_workout=30 min outside [45,120]" in msg


def test_time_above_max_recorded_in_failure_report() -> None:
    """Branch: minutes > max_minutes -> reason appended (line 42-45)."""
    df = _sample_df()
    df.loc[df["title"].isin(["PHUL", "GZCLP", "nSuns 5/3/1"]), "time_per_workout"] = 180
    with pytest.raises(ProgramNotFoundError) as exc_info:
        pick_program(df, min_minutes=45, max_minutes=120)
    msg = str(exc_info.value)
    assert "time_per_workout=180 min outside [45,120]" in msg


def test_program_length_failure_recorded_in_failure_report() -> None:
    """Branch: program_length < min_weeks -> reason appended (line 36-39)."""
    df = _sample_df()
    df.loc[df["title"].isin(["PHUL", "GZCLP", "nSuns 5/3/1"]), "program_length"] = 4
    with pytest.raises(ProgramNotFoundError) as exc_info:
        pick_program(df, min_weeks=8)
    msg = str(exc_info.value)
    assert "program_length=4 weeks < min_weeks=8" in msg


def test_falls_back_when_primary_has_wrong_equipment() -> None:
    """PHUL equipment switched -> GZCLP picked (equipment-branch fallback)."""
    df = _sample_df()
    df.loc[df["title"] == "PHUL", "equipment"] = "Bodyweight"
    assert pick_program(df) == "GZCLP"


def test_falls_back_when_primary_time_out_of_range() -> None:
    """PHUL time=30 (< 45) -> GZCLP picked (time-branch fallback)."""
    df = _sample_df()
    df.loc[df["title"] == "PHUL", "time_per_workout"] = 30
    assert pick_program(df) == "GZCLP"
