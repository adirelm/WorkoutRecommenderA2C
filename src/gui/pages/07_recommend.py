"""Section 7.7 Recommend page - brief 7.6 / 7.7 deliverable for sdk.recommend.

Pure SDK consumer (CLAUDE.md section 3). Takes a 12-channel ``State`` via
``State.initial()`` or per-channel sliders, calls ``sdk.recommend``, and
renders action + probability + reward-decomposition charts. Friendly
fallback when no policy has been trained yet (SDK keeps only the latest).
"""

from __future__ import annotations

import streamlit as st

from src.env.state import STATE_CHANNEL_NAMES
from src.env.workout_env import WorkoutEnv
from src.gui.components import hero, info_card, page_footer
from src.gui.labels import STATE_CHANNEL_LABEL
from src.gui.pages._recommend_ui import (
    apply_action,
    last_trained_algo,
    no_policy_card,
    render_result,
    state_from_sliders,
)
from src.gui.state import GUIState, get_sdk

_RESET_TOGGLE = "Reset to fresh trainee defaults (Use State.initial())"


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
    algo = last_trained_algo()
    if algo is None:
        no_policy_card()
        page_footer()
        return
    st.sidebar.header("State input")
    use_initial = st.sidebar.toggle(
        _RESET_TOGGLE,
        value=True,
        help="On = use the fresh-trainee default state. Off = dial each channel by hand.",
    )
    st.sidebar.caption(f"Active policy (auto-detected): **{algo}**")
    state = state_from_sliders(use_initial)
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
        render_result(rec, [bool(m) for m in env.action_mask().tolist()])
        if st.button(
            "Apply this action", help="Step the env with the recommended action and show next state."
        ):
            apply_action(rec.action_id, gui)
        nxt = gui.get("last_next_state")
        if nxt is not None:
            st.success(f"Applied -> reward={gui.get('last_step_reward'):+.3f}")
            st.json({STATE_CHANNEL_LABEL[n]: getattr(nxt, n) for n in STATE_CHANNEL_NAMES})
    page_footer()


render()
