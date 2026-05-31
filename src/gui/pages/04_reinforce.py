# streamlit page filenames must start with NN_ for ordering
"""REINFORCE training page (brief §7.4, eq. 4 + eq. 16).

Thin orchestrator. Sidebar + theory block live in
:mod:`src.gui.pages._reinforce_ui` so this file stays under the
150-LOC CLAUDE.md §1 limit. Training is delegated to
:func:`src.gui.callbacks.train_reinforce_live`, which streams
per-episode rewards into a live Plotly chart.

Per CLAUDE.md §3 + §1.4, this page imports *only* from :mod:`src.gui.*`
and :mod:`src.sdk.sdk` (via ``get_sdk``); no direct service imports.
"""

from __future__ import annotations

import streamlit as st

from src.gui.callbacks import LiveTrainingCallback, train_reinforce_live
from src.gui.charts import reward_curve
from src.gui.components import hero, metric_row
from src.gui.pages._reinforce_ui import info_block, sidebar
from src.gui.state import GUIState, get_sdk, set_last_reinforce_history

PAGE = "reinforce"
MEAN_LAST_WINDOW = 10  # episodes used for the "mean_last_10" KPI


def _run_training(params: dict[str, float | int | bool]) -> None:
    """Drive ``train_reinforce_live`` and surface the resulting history."""
    sdk = get_sdk()
    chart_slot = st.empty()
    progress_slot = st.empty() if params["show_progress"] else None
    rewards_stream: list[float] = []
    total = int(params["episodes"])

    def _on_step(step_idx: int, metrics: dict) -> None:
        rewards_stream.append(float(metrics.get("reward", 0.0)))
        if params["show_progress"]:
            chart_slot.plotly_chart(
                reward_curve(tuple(rewards_stream), title="Live REINFORCE reward"),
                use_container_width=True,
                key=f"reinforce_live_chart_step_{step_idx}",
            )
            if progress_slot is not None:
                progress_slot.progress(
                    (step_idx + 1) / max(total, 1),
                    text=f"episode {step_idx + 1}/{total}",
                )

    def _on_done(final) -> None:
        chart_slot.plotly_chart(
            reward_curve(final.rewards, title="REINFORCE episode reward"),
            use_container_width=True,
            key="reinforce_live_chart_done",
        )
        if progress_slot is not None:
            progress_slot.empty()

    history = train_reinforce_live(
        sdk,
        episodes=total,
        callback=LiveTrainingCallback(on_step=_on_step, on_done=_on_done),
    )
    set_last_reinforce_history(history)
    GUIState(PAGE).set("last_history", history)


def _render_final_metrics() -> None:
    """Show the four KPI metrics for the most recent REINFORCE run on this page."""
    history = GUIState(PAGE).get("last_history")
    if history is None:
        st.info("Train REINFORCE to see episode metrics here.")
        return
    rewards = history.rewards
    window = MEAN_LAST_WINDOW
    last_n = rewards[-window:] if len(rewards) >= window else rewards
    mean_last_10 = sum(last_n) / max(len(last_n), 1)
    metric_row(
        {
            "episodes_run": f"{history.episodes_run}",
            "final_reward": f"{rewards[-1]:+.2f}",
            "mean_last_10": f"{mean_last_10:+.2f}",
            "baseline_at_end": f"{history.baseline[-1]:+.2f}",
        }
    )


def render() -> None:
    """Entry point invoked by ``st.navigation`` for the REINFORCE page."""
    hero("REINFORCE", "Monte-Carlo policy gradient with running-mean baseline", icon="🎯")
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Train the REINFORCE policy gradient with a running-mean baseline "
            "(brief §7.4); live reward curve + KPI strip below."
        )
    info_block()
    params = sidebar()
    if st.button(
        "Train REINFORCE",
        type="primary",
        help="Run REINFORCE with running-mean baseline (eq. 16).",
    ):
        with st.spinner("Training REINFORCE..."):
            _run_training(params)
    _render_final_metrics()


render()
