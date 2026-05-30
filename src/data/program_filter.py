"""Program selection — brief §7.2.4 selection criteria.

Pick the first program (primary, then fallbacks in order) that exists in
the dataset AND passes the equipment / length / session-time criteria.
If none pass, raise ProgramNotFoundError with a full audit trail of why
each candidate was rejected.
"""
from __future__ import annotations

import pandas as pd


class ProgramNotFoundError(LookupError):
    """No program in the dataset matches the selection criteria."""


def _evaluate(
    programs: pd.DataFrame,
    name: str,
    *,
    min_weeks: int,
    equipment: str,
    min_minutes: int,
    max_minutes: int,
) -> list[str]:
    """Return a list of failure reasons for `name`. Empty list -> passes."""
    rows = programs.loc[programs["title"] == name]
    if rows.empty:
        return ["not present in dataset"]
    row = rows.iloc[0]
    reasons: list[str] = []
    if row["equipment"] != equipment:
        reasons.append(
            f"equipment={row['equipment']!r} != required {equipment!r}"
        )
    if int(row["program_length"]) < min_weeks:
        reasons.append(
            f"program_length={int(row['program_length'])} weeks < min_weeks={min_weeks}"
        )
    minutes = int(row["time_per_workout"])
    if minutes < min_minutes or minutes > max_minutes:
        reasons.append(
            f"time_per_workout={minutes} min outside "
            f"[{min_minutes},{max_minutes}]"
        )
    return reasons


def pick_program(
    programs: pd.DataFrame,
    primary: str = "PHUL",
    fallbacks: tuple[str, ...] = ("GZCLP", "nSuns 5/3/1"),
    min_weeks: int = 8,
    equipment: str = "Full Gym",
    min_minutes: int = 45,
    max_minutes: int = 120,
) -> str:
    """Select a program by brief §7.2.4 criteria.

    1) If `primary` exists AND passes criteria -> return primary.
    2) Else try fallbacks in order.
    3) Else raise ProgramNotFoundError with a per-candidate failure report.
    """
    candidates = (primary, *fallbacks)
    failure_report: list[str] = []
    for name in candidates:
        reasons = _evaluate(
            programs,
            name,
            min_weeks=min_weeks,
            equipment=equipment,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
        )
        if not reasons:
            return name
        failure_report.append(f"  - {name}: " + "; ".join(reasons))

    raise ProgramNotFoundError(
        "No program matched §7.2.4 selection criteria "
        f"(min_weeks={min_weeks}, equipment={equipment!r}, "
        f"time in [{min_minutes},{max_minutes}] min).\n"
        "Tried:\n" + "\n".join(failure_report)
    )
