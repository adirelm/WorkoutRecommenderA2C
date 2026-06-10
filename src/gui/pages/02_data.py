"""§7.2 Data & Environment page — wraps WorkoutSDK.prepare_data().

Surfaces the LogbookHandle as a metric row, draws the 28-day
masked-uniform state heatmap, and lets the user pick a single day
to see its (gain, overload, imbalance) reward decomposition."""

from __future__ import annotations

import streamlit as st

from src.gui.components import (
    hero,
    info_card,
    metric_row,
    page_footer,
    reference_callout,
)
from src.gui.pages._data_ui import (
    handle_metrics,
    heatmap_section,
    load_section,
    reward_section,
)
from src.gui.state import GUIState, get_sdk

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


def render() -> None:
    """Render the §7.2 Data & Environment page."""
    state = GUIState("data")
    sdk = get_sdk()
    hero(
        title="Data & Environment",
        subtitle="§7.2 data layer · WorkoutEnv rollout · reward decomposition.",
        icon="📚",
    )
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Load the PHUL trainee (real Kaggle program, Optimized PHUL) and inspect "
            "the 28-day state trajectory + per-day reward decomposition (brief §7.2)."
        )
    info_card("Data ingestion", _SECTION_BLURB)
    reference_callout(
        "Kaggle dataset",
        f"<a href='{_KAGGLE_URL}'>{_KAGGLE_SLUG}</a> — 600k Fitness / Exercise "
        f"& Workout Program Dataset (Adnan Elouardi). PHUL trainee rows are "
        f"the default filter; the synthetic trainee is the deterministic fallback.",
    )
    handle = load_section(state, sdk)
    if handle is not None:
        metric_row(handle_metrics(handle))
        heatmap_section(state.get("trajectory") or [])
        reward_section(state, state.get("infos") or [])
    page_footer()


render()
