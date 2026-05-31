"""Helpers for §7.7 Recommend page — kept separate so 07_recommend.py stays ≤150 LOC."""

from __future__ import annotations

from dataclasses import fields

import streamlit as st

from src.env.state import ACTION_NAMES, STATE_CHANNEL_NAMES, State
from src.env.workout_env import WorkoutEnv
from src.gui.charts import action_probability_bar, reward_decomposition_bar
from src.gui.components import info_card, metric_row
from src.gui.labels import STATE_CHANNEL_HELP, STATE_CHANNEL_LABEL
from src.gui.state import GUIState, get_last_a2c_history, get_last_reinforce_history

BOUNDS: dict[str, tuple[float, float, float]] = {
    "fatigue": (0.0, 1.0, 0.05),
    "soreness_push": (0.0, 1.0, 0.05),
    "soreness_pull": (0.0, 1.0, 0.05),
    "soreness_legs": (0.0, 1.0, 0.05),
    "soreness_core": (0.0, 1.0, 0.05),
    "readiness": (0.0, 1.0, 0.05),
    "rolling_7d_volume": (0.0, 200.0, 1.0),
    "streak_days_trained": (0.0, 30.0, 1.0),
    "days_since_last_rest": (0.0, 14.0, 1.0),
    "muscle_balance_push_vs_pull": (-1.0, 1.0, 0.05),
    "adherence_signal": (-1.0, 1.0, 0.05),
    "weekly_progress": (0.0, 1.2, 0.05),
}
INT_CHANNELS: frozenset[str] = frozenset({"streak_days_trained", "days_since_last_rest"})


def last_trained_algo() -> str | None:
    """Auto-detect which trainer last ran via cross-page session keys."""
    if get_last_a2c_history() is not None:
        return "A2C"
    if get_last_reinforce_history() is not None:
        return "REINFORCE"
    return None


def state_from_sliders(use_initial: bool) -> State:
    """Build ``State`` from ``State.initial()`` or one slider per channel."""
    base = State.initial()
    if use_initial:
        for n in STATE_CHANNEL_NAMES:
            st.sidebar.caption(f"{STATE_CHANNEL_LABEL[n]}: {getattr(base, n)}")
        return base
    values: dict[str, float | int] = {}
    for f in fields(State):
        lo, hi, step = BOUNDS[f.name]
        v = st.sidebar.slider(
            STATE_CHANNEL_LABEL[f.name],
            lo,
            hi,
            float(getattr(base, f.name)),
            step,
            help=STATE_CHANNEL_HELP[f.name],
        )
        values[f.name] = int(v) if f.name in INT_CHANNELS else float(v)
    return State(**values)


def no_policy_card() -> None:
    """Guide the user to train a policy first (product copy, no CLI jargon)."""
    info_card(
        "No trained policy detected",
        "Open the <b>REINFORCE</b> or <b>A2C</b> page and click <i>Train</i>, then "
        "come back here to get a recommendation. "
        "(CLI: <code>uv run main.py</code> &rarr; option 3 (REINFORCE) or 4 (A2C).)",
    )


def render_result(rec, mask: list[bool]) -> None:
    """Render the recommended action + probability + reward-decomposition panels."""
    st.markdown(
        f"<h1 style='color:#003D7A;margin-top:8px;'>{rec.action_name}</h1>",
        unsafe_allow_html=True,
    )
    metric_row(
        {
            "action_id": str(rec.action_id),
            "P(picked)": f"{rec.probs[rec.action_id]:.2f}",
            "expected_reward": f"{rec.expected_reward:+.3f}",
        }
    )
    st.plotly_chart(
        action_probability_bar(rec.probs, list(ACTION_NAMES), mask=mask),
        use_container_width=True,
    )
    st.plotly_chart(
        reward_decomposition_bar({"expected_reward": float(rec.expected_reward)}),
        use_container_width=True,
    )


def apply_action(action_id: int, gui: GUIState) -> None:
    """Advance one env.step from a fresh env and stash the resulting next-state."""
    env = WorkoutEnv(seed=42)
    env.reset()
    next_state, reward, done, info = env.step(int(action_id))
    gui.set("last_next_state", next_state)
    gui.set("last_step_reward", float(reward))
    gui.set("last_step_done", bool(done))
    gui.set("last_step_info", info)
