"""Tests for the real-Kaggle program loader (brief §7.2.1-§7.2.4).

These run against the committed REAL Kaggle inputs (data/raw/program_summary.csv
+ the §7.2.4 PHUL subset), proving the data layer is wired into the pipeline
rather than tested in isolation (closes audit finding F-2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.program_loader import load_chosen_program


def test_load_chosen_program_picks_real_phul() -> None:
    """The loader selects the real Kaggle PHUL title from the committed summary."""
    name, cleaned = load_chosen_program()
    assert name == "Optimized PHUL (Power Hypertrophy Upper Lower)"
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty


def test_loader_adds_muscle_group_and_uses_reps_equivalent() -> None:
    """Real detailed schema lacks muscle_group; loader infers it + converts time-encoded reps."""
    _name, cleaned = load_chosen_program()
    assert "muscle_group" in cleaned.columns
    assert set(cleaned["muscle_group"].unique()).issubset(
        {"push", "pull", "legs", "core", "cardio", "mobility"}
    )
    # reps column is the time-converted equivalent (no negative reps survive).
    assert "reps_equivalent" in cleaned.columns
    assert (cleaned["reps"] == cleaned["reps_equivalent"]).all()
    assert (cleaned["reps"] >= 0).all()


def test_loader_only_returns_chosen_program_rows() -> None:
    """Filtering keeps only the chosen program's exercise rows."""
    name, cleaned = load_chosen_program()
    assert (cleaned["title"] == name).all()


def test_loader_persists_data_quality_report() -> None:
    """§7.2.3 audit trail: results/data_quality_report.txt written with rule counts."""
    load_chosen_program()
    text = Path("results/data_quality_report.txt").read_text(encoding="utf-8")
    assert "rows_in: 312" in text
    assert "time_encoded_reps_reclassified (rule b): 26" in text
