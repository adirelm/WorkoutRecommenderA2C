"""Section 7.7 Recommend page - brief 7.6 / 7.7 deliverable for sdk.recommend.

Pure SDK consumer (CLAUDE.md section 3). Takes a 12-channel ``State`` via
``State.initial()`` or per-channel sliders, calls ``sdk.recommend``, and
renders action + probability + reward-decomposition charts. Friendly
fallback when no policy has been trained yet (SDK keeps only the latest).
"""

from __future__ import annotations

from dataclasses import fields

import streamlit as st

from src.env.state import ACTION_NAMES, STATE_CHANNEL_NAMES, State
from src.env.workout_env import WorkoutEnv
from src.gui.charts import action_probability_bar, reward_decomposition_bar
from src.gui.components import hero, info_card, metric_row, page_footer
from src.gui.labels import STATE_CHANNEL_HELP, STATE_CHANNEL_LABEL
from src.gui.state import (
    GUIState,
    get_last_a2c_history,
    get_last_reinforce_history,
    get_sdk,
)

_BOUNDS: dict[str, tuple[float, float, float]] = {
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
_INT_CHANNELS: frozenset[str] = frozenset({"streak_days_trained", "days_since_last_rest"})
_RESET_TOGGLE = "Reset to fresh trainee defaults (Use State.initial())"


def _last_trained_algo() -> str | None:
    """Auto-detect which trainer last ran via cross-page session keys."""
    if get_last_a2c_history() is not None:
        return "A2C"
    if get_last_reinforce_history() is not None:
        return "REINFORCE"
    return None


def _state_from_sliders(use_initial: bool) -> State:
    """Build ``State`` from ``State.initial()`` or one slider per channel."""
    base = State.initial()
    if use_initial:
        for n in STATE_CHANNEL_NAMES:
            st.sidebar.caption(f"{STATE_CHANNEL_LABEL[n]}: {getattr(base, n)}")
        return base
    values: dict[str, float | int] = {}
    for f in fields(State):
        lo, hi, step = _BOUNDS[f.name]
        v = st.sidebar.slider(
            STATE_CHANNEL_LABEL[f.name],
            lo,
            hi,
            float(getattr(base, f.name)),
            step,
            help=STATE_CHANNEL_HELP[f.name],
        )
        values[f.name] = int(v) if f.name in _INT_CHANNELS else float(v)
    return State(**values)


def _no_policy_card() -> None:
    """Guide the user to train a policy first (product copy, no CLI jargon)."""
    info_card(
        "No trained policy detected",
        "Open the <b>REINFORCE</b> or <b>A2C</b> page and click <i>Train</i>, then "
        "come back here to get a recommendation. "
        "(CLI: <code>uv run main.py</code> &rarr; option 3 (REINFORCE) or 4 (A2C).)",
    )


def _render_result(rec, mask: list[bool]) -> None:
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


def _apply_action(action_id: int, gui: GUIState) -> None:
    """Advance one env.step from a fresh env and stash the resulting next-state."""
    env = WorkoutEnv(seed=42)
    env.reset()
    next_state, reward, done, info = env.step(int(action_id))
    gui.set("last_next_state", next_state)
    gui.set("last_step_reward", float(reward))
    gui.set("last_step_done", bool(done))
    gui.set("last_step_info", info)


def render() -> None:
    """Render the section 7.7 recommend page."""
    gui = GUIState("recommend")
    hero("Recommend", "Brief 7.6 / 7.7 - sdk.recommend(state) gives the next-day pick.", "💡")
    info_card(
        "What this page does",
        "Pass a 12-channel <b>State</b> (fatigue, 4x soreness, readiness, volume, streak, "
        "days-since-rest, push/pull balance, adherence, weekly progress) to the trained "
        "policy. Returns a masked-softmax recommendation; <i>expected reward</i> is V(s) "
        "for A2C and a sentinel (0.0 = <i>unknown</i>) for REINFORCE.",
    )
    algo = _last_trained_algo()
    if algo is None:
        _no_policy_card()
        page_footer()
        return
    st.sidebar.header("State input")
    use_initial = st.sidebar.toggle(
        _RESET_TOGGLE,
        value=True,
        help="On = use the fresh-trainee default state. Off = dial each channel by hand.",
    )
    st.sidebar.caption(f"Active policy (auto-detected): **{algo}**")
    state = _state_from_sliders(use_initial)
    if st.button("Recommend", type="primary", help="Call sdk.recommend(state) with the inputs above."):
        try:
            gui.set("last_recommendation", get_sdk().recommend(state))
            gui.set("last_state", state)
        except RuntimeError as exc:
            info_card("Recommendation failed", str(exc))
    rec = gui.get("last_recommendation")
    if rec is not None:
        env = WorkoutEnv(seed=42)
        env.reset()
        _render_result(rec, [bool(m) for m in env.action_mask().tolist()])
        if st.button(
            "Apply this action", help="Step the env with the recommended action and show next state."
        ):
            _apply_action(rec.action_id, gui)
        nxt = gui.get("last_next_state")
        if nxt is not None:
            st.success(f"Applied -> reward={gui.get('last_step_reward'):+.3f}")
            st.json({STATE_CHANNEL_LABEL[n]: getattr(nxt, n) for n in STATE_CHANNEL_NAMES})
    page_footer()


render()
