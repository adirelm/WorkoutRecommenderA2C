"""User-facing labels + help tooltips for state channels and common controls.

Pages MUST use these maps instead of raw snake_case field names so the GUI
reads as a polished product rather than a code dump (V3 §10 + writing audit).
"""
from __future__ import annotations

# Mirrors src/env/state.STATE_CHANNEL_NAMES — keep in sync.
STATE_CHANNEL_LABEL: dict[str, str] = {
    "fatigue": "Fatigue (0-1)",
    "soreness_push": "Push soreness (0-1)",
    "soreness_pull": "Pull soreness (0-1)",
    "soreness_legs": "Leg soreness (0-1)",
    "soreness_core": "Core soreness (0-1)",
    "readiness": "Readiness (0-1)",
    "rolling_7d_volume": "7-day rolling volume",
    "streak_days_trained": "Consecutive training days",
    "days_since_last_rest": "Days since last rest",
    "muscle_balance_push_vs_pull": "Push-vs-pull balance",
    "adherence_signal": "Program adherence (0-1)",
    "weekly_progress": "Weekly progress (0-1)",
}

STATE_CHANNEL_HELP: dict[str, str] = {
    "fatigue": "Accumulated systemic fatigue. 0 = fresh, 1 = exhausted.",
    "soreness_push": "Push-muscle (chest/shoulders/triceps) soreness, 48 h recovery rule.",
    "soreness_pull": "Pull-muscle (back/biceps) soreness, 48 h recovery rule.",
    "soreness_legs": "Lower-body soreness, 48 h recovery rule.",
    "soreness_core": "Core soreness, 24 h recovery rule.",
    "readiness": "Subjective readiness signal (HRV-style proxy).",
    "rolling_7d_volume": "Sum of training volume over the last 7 days (sets).",
    "streak_days_trained": "Consecutive days the trainee has trained without a rest.",
    "days_since_last_rest": "Days elapsed since the last rest day.",
    "muscle_balance_push_vs_pull": "Cumulative push minus pull volume — 0 = balanced.",
    "adherence_signal": "How closely the trainee has been following the prescribed program.",
    "weekly_progress": "Progress through the current training week (0..1).",
}

# Shared section heading style — keep one convention across pages.
SECTION_HEADER_PREFIX = "§"  # use as "§7.4 — REINFORCE"
