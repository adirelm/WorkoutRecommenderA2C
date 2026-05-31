"""§7.2 Data & Environment page — wraps WorkoutSDK.prepare_data().

Surfaces the LogbookHandle as a metric row, draws the 28-day
masked-uniform state heatmap, and lets the user pick a single day
to see its (gain, overload, imbalance) reward decomposition."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.env.state import State
from src.gui.charts import muscle_distribution_heatmap, reward_decomposition_bar
from src.gui.components import (
    hero,
    info_card,
    metric_row,
    page_footer,
    reference_callout,
)
from src.gui.state import GUIState, get_sdk
from src.sdk.sdk import WorkoutSDK
from src.sdk.types import LogbookHandle

_KAGGLE_SLUG = "adnanelouardi/600k-fitness-exercise-and-workout-program-dataset"
_KAGGLE_URL = f"https://www.kaggle.com/datasets/{_KAGGLE_SLUG}"
_SECTION_BLURB = (
    "§7.2 — the data layer hydrates a Kaggle workout-program CSV "
    "(filtered to a Power Hypertrophy Upper Lower (PHUL) trainee) or "
    "falls back to the deterministic <code>SyntheticTrainee</code>. Either "
    "source is replayed through <code>WorkoutEnv</code> for 28 days, "
    "producing the state trajectory the LSTM world-model and the "
    "REINFORCE/A2C policies all train on."
)


def _roll_trajectory(sdk: WorkoutSDK, days: int = 28) -> tuple[list[State], list[dict]]:
    """Roll a masked-uniform rollout, capturing per-step state + reward info."""
    sdk.prepare_data()
    env = sdk.ensure_env()
    state = env.reset()
    states: list[State] = [state]
    infos: list[dict] = []
    rng = np.random.default_rng(sdk.seed)
    for _ in range(days):
        mask = env.action_mask()
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            break
        action = int(rng.choice(legal))
        state, _r, done, info = env.step(action)
        states.append(state)
        infos.append(info)
        if done:
            break
    return states, infos


def _handle_metrics(handle: LogbookHandle) -> dict[str, str]:
    """Map LogbookHandle into the metric_row signature."""
    pretty = handle.program_name.replace("_", " ").title()
    return {
        "Program": pretty,
        "Episode length (days)": str(handle.n_days),
        "State dim (channels)": str(handle.state_dim),
    }


def _load_section(state: GUIState, sdk: WorkoutSDK) -> LogbookHandle | None:
    """Render the load-button + trigger prepare_data(); persist handle."""
    cols = st.columns([1, 3])
    with cols[0]:
        if st.button(
            "Load PHUL trainee",
            type="primary",
            use_container_width=True,
            help=(
                "Hydrate the Power Hypertrophy Upper Lower (PHUL) trainee "
                "from Kaggle (or synthetic fallback if Kaggle unavailable)."
            ),
        ):
            handle = sdk.prepare_data()
            state.set("handle", handle)
            states, infos = _roll_trajectory(sdk, days=int(handle.n_days))
            state.set("trajectory", states)
            state.set("infos", infos)
            st.toast(f"Loaded {handle.program_name} ({handle.n_days} days).", icon="✅")
    with cols[1]:
        st.caption(
            "Calls `WorkoutSDK.prepare_data()`; the resulting `LogbookHandle` "
            "is persisted in session state for downstream pages."
        )
    return state.get("handle")


def _heatmap_section(trajectory: list[State]) -> None:
    """Plotly muscle / state heatmap over the 28-day rollout."""
    st.subheader("State trajectory (28 days, masked-uniform rollout)")
    if not trajectory:
        st.info("Load the trainee to populate the trajectory.")
        return
    fig = muscle_distribution_heatmap(trajectory)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(trajectory)} state vectors (initial state + 28 daily "
        f"transitions) × 12 channels. Darker = higher; this is the same "
        f"data the LSTM world-model fits."
    )


def _reward_section(state: GUIState, infos: list[dict]) -> None:
    """Sample-day reward-decomposition bar (gain / overload / imbalance)."""
    st.subheader("Reward decomposition for a sample day")
    if not infos:
        st.info("Load the trainee to populate per-step reward components.")
        return
    default_day = int(state.get("sample_day", 1))
    day = st.slider(
        "Rollout day",
        min_value=1,
        max_value=len(infos),
        value=min(default_day, len(infos)),
        step=1,
        help="Select which day of the 28-day rollout to inspect.",
    )
    state.set("sample_day", day)
    info = infos[day - 1]
    decomp = {
        "gain": float(info.get("gain", 0.0)),
        "overload": float(info.get("overload", 0.0)),
        "imbalance": float(info.get("imbalance", 0.0)),
    }
    st.plotly_chart(reward_decomposition_bar(decomp), use_container_width=True)
    st.caption(
        f"Action on day {day}: **{info.get('muscle_group', '—')}** "
        f"(volume Δ = {info.get('volume_delta', 0.0):+.1f}). "
        f"Total shaped reward = `gain − 2·overload − imbalance`."
    )


def render() -> None:
    """Render the §7.2 Data & Environment page."""
    state = GUIState("data")
    sdk = get_sdk()
    hero(
        title="Data & Environment",
        subtitle="§7.2 data layer · WorkoutEnv rollout · reward decomposition.",
        icon="📚",
    )
    info_card("Data ingestion", _SECTION_BLURB)
    reference_callout(
        "Kaggle dataset",
        f"<a href='{_KAGGLE_URL}'>{_KAGGLE_SLUG}</a> — 600k Fitness / Exercise "
        f"& Workout Program Dataset (Adnan Elouardi). PHUL trainee rows are "
        f"the default filter; the synthetic trainee is the deterministic fallback.",
    )
    handle = _load_section(state, sdk)
    if handle is not None:
        metric_row(_handle_metrics(handle))
        _heatmap_section(state.get("trajectory") or [])
        _reward_section(state, state.get("infos") or [])
    page_footer()


render()
