"""Load + clean the chosen real Kaggle program (brief §7.2.1-§7.2.4).

End-to-end data-layer entry point that the LSTM trajectory builder consumes
(closes audit finding F-2 — previously ``src/data/*`` was imported only by
tests). Steps, all driven by ``config/config.yaml`` (no hardcoded values):

  1. Read the real ``program_summary.csv`` + the §7.2.4 PHUL exercise subset
     from ``paths.data_raw``.
  2. ``pick_program`` selects the configured primary (PHUL) / fallbacks by the
     §7.2.4 criteria (Full Gym, ≥8 weeks, 45-120 min).
  3. Keep only the chosen program's exercise rows.
  4. Apply the §7.2.3 data-quality contract (drop bad rows; convert
     time-encoded reps — the real data carries 26 negative-rep rows).
  5. Infer ``muscle_group`` from ``exercise_name`` (the real detailed schema
     has no muscle_group column) and set ``reps`` to the converted equivalent
     so downstream Σ(sets·reps) volume is honest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocessor import apply_data_quality_contract, normalise_muscle_group
from src.data.program_filter import pick_program
from src.utils.config_loader import load_config


def load_chosen_program(config: dict | None = None) -> tuple[str, pd.DataFrame]:
    """Return ``(chosen_program_title, cleaned_exercise_rows)`` from real Kaggle data.

    Args:
        config: Parsed ``config.yaml`` dict; loaded from disk when ``None``.

    Returns:
        The chosen program title and a cleaned DataFrame whose ``reps`` column
        holds time-converted equivalents and which carries an inferred
        ``muscle_group`` column.
    """
    cfg = config if config is not None else load_config()
    ds = cfg["dataset"]
    raw_dir = Path(cfg["paths"]["data_raw"])

    summary = pd.read_csv(raw_dir / ds["summary_csv"])
    detailed = pd.read_csv(raw_dir / ds["detailed_csv"])

    name = pick_program(
        summary,
        primary=ds["primary_program"],
        fallbacks=tuple(ds["fallback_programs"]),
        min_weeks=int(ds["program_filter_min_weeks"]),
        equipment=ds["program_filter_equipment"],
        min_minutes=int(ds["program_filter_min_minutes"]),
        max_minutes=int(ds["program_filter_max_minutes"]),
    )

    rows = detailed.loc[detailed["title"] == name].copy()
    cleaned, report = apply_data_quality_contract(
        rows, seconds_per_rep=float(cfg["data_quality"]["seconds_per_rep"])
    )
    cleaned["muscle_group"] = cleaned["exercise_name"].map(normalise_muscle_group)
    cleaned["reps"] = cleaned["reps_equivalent"]
    _write_quality_report(name, report, Path(cfg["paths"]["results"]))
    return name, cleaned


def _write_quality_report(name: str, report, results_dir: Path) -> None:
    """Persist the §7.2.3 data-quality audit trail (PRD F3a / TODO T-MF3)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "data_quality_report.txt").write_text(
        f"Data quality report — chosen program: {name}\n"
        f"rows_in: {report.rows_in}\n"
        f"rows_out: {report.rows_out}\n"
        f"negative_volume_dropped (rule a): {report.negative_volume_dropped}\n"
        f"time_encoded_reps_reclassified (rule b): {report.time_encoded_reps_reclassified}\n"
        f"rest_days_inserted (rule c, by aggregator): see trajectory builder\n",
        encoding="utf-8",
    )
