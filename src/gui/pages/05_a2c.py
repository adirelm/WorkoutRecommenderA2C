# streamlit page filenames must start with NN_ for ordering
"""A2C training page (brief §7.5, eq. 9 + eq. 10 + eq. 12).

Thin orchestrator: sidebar + theory block live in
:mod:`src.gui.pages._a2c_ui` to keep this file under the 150-LOC
CLAUDE.md §1 limit. Training is delegated to
:func:`src.gui.callbacks.train_a2c_live` and the live 3-panel
diagnostics chart is produced by
:func:`src.gui.charts.actor_critic_panels`. Per CLAUDE.md §3 + §1.4
this page imports only from src.gui.* and src.sdk.sdk.
"""

from __future__ import annotations

import streamlit as st

from src.gui.callbacks import LiveTrainingCallback, train_a2c_live
from src.gui.charts import actor_critic_panels
from src.gui.components import hero, metric_row
from src.gui.pages._a2c_ui import info_block, sidebar
from src.gui.state import GUIState, get_sdk, set_last_a2c_history
from src.services.a2c_types import A2CHistory

PAGE = "a2c"


def _run_training(params: dict[str, float | int]) -> None:
    """Drive ``train_a2c_live`` and stream the 3-panel diagnostics chart."""
    sdk = get_sdk()
    chart_slot = st.empty()
    progress_slot = st.empty()
    actor_s: list[float] = []
    critic_s: list[float] = []
    reward_s: list[float] = []
    adv_s: list[float] = []
    total = int(params["episodes"])

    def _on_step(step_idx: int, metrics: dict) -> None:
        reward_s.append(float(metrics.get("reward", 0.0)))
        actor_s.append(float(metrics.get("actor_loss", 0.0)))
        critic_s.append(float(metrics.get("critic_loss", 0.0)))
        adv_s.append(0.0)  # advantage stream not exposed by streaming callback
        interim = A2CHistory(
            episodes_run=len(reward_s),
            rewards=tuple(reward_s),
            actor_losses=tuple(actor_s),
            critic_losses=tuple(critic_s),
            advantages_mean=tuple(adv_s),
            seed=int(sdk.seed),
        )
        chart_slot.plotly_chart(
            actor_critic_panels(interim),
            use_container_width=True,
            key=f"a2c_live_chart_step_{step_idx}",
        )
        progress_slot.progress((step_idx + 1) / max(total, 1), text=f"episode {step_idx + 1}/{total}")

    def _on_done(final: A2CHistory) -> None:
        progress_slot.empty()
        chart_slot.plotly_chart(
            actor_critic_panels(final),
            use_container_width=True,
            key="a2c_live_chart_done",
        )

    history = train_a2c_live(
        sdk, episodes=total, callback=LiveTrainingCallback(on_step=_on_step, on_done=_on_done)
    )
    set_last_a2c_history(history)
    GUIState(PAGE).set("last_history", history)
    GUIState(PAGE).set("last_params", params)


def _render_final_metrics() -> None:
    """Show the five KPI metrics for the most recent A2C run on this page."""
    history = GUIState(PAGE).get("last_history")
    if history is None:
        st.info("Train A2C to see episode metrics here.")
        return
    mean_adv = sum(history.advantages_mean) / len(history.advantages_mean) if history.advantages_mean else 0.0
    metric_row(
        {
            "Episodes run": f"{history.episodes_run}",
            "Final reward": f"{history.rewards[-1]:+.2f}",
            "Final actor loss": f"{history.actor_losses[-1]:+.4f}",
            "Final critic loss": f"{history.critic_losses[-1]:.4f}",
            "Mean advantage": f"{mean_adv:+.4f}",
        }
    )


def render() -> None:
    """Entry point invoked by ``st.navigation`` for the A2C page."""
    hero("A2C", "Synchronous advantage actor-critic with entropy bonus", icon="⚡")
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Train synchronous A2C with TD-advantage + entropy bonus "
            "(brief §7.5); live 3-panel diagnostics chart below."
        )
    info_block()
    params = sidebar()
    if st.button("Train A2C", type="primary"):
        with st.spinner("Training A2C..."):
            _run_training(params)
    _render_final_metrics()


render()
