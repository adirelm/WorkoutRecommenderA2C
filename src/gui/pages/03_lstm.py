"""§7.3 LSTM world-model training page (Phase 9 GUI).

ADR-006 GUI-layer exception: this page instantiates :class:`LSTMWorldModel`
+ :class:`LSTMTrainer` directly so the user can live-stream the loss curve
while tuning hidden-size / layers / epochs with sliders (``WorkoutSDK.
train_world_model`` runs a fixed config; interactive sliders need the trainer
directly). The trajectory comes from the real chosen Kaggle program via
``src.model.program_trajectory.generate_program_trajectory`` — the same
source the SDK uses. All other pages stay SDK-only.
"""

from __future__ import annotations

import streamlit as st

from src.gui.components import hero, info_card, latex_block, page_footer
from src.gui.pages._lstm_ui import (
    render_results,
    run_training,
    sidebar_controls,
)
from src.gui.state import GUIState


def render() -> None:
    """Render the LSTM training page in full."""
    state = GUIState("lstm")
    hero("LSTM World Model", "§7.3 — recurrent transition model f_φ", icon="🧠")
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Train the LSTM world model f_φ(s_t, a_t, h_t) → ŝ_{t+1} on the "
            "current trainee's trajectory windows (brief §7.3)."
        )
    info_card(
        "§7.3 — recurrent world model",
        "The trainee environment is a POMDP; we learn an LSTM f_φ whose hidden "
        "state is a sufficient statistic of history h_t. REINFORCE/A2C later "
        "roll out against this frozen simulator. Loss (eq. 14): mean-squared "
        "error between the predicted and observed next state.",
    )
    latex_block(
        r"\mathcal{L}(\varphi) = \frac{1}{N}\sum_{t} \left\| f_\varphi(s_t, a_t, h_t) - s_{t+1} \right\|_2^2",
        label="Equation 14 — supervised LSTM objective",
    )
    params = sidebar_controls()
    if st.button(
        "Train LSTM",
        type="primary",
        use_container_width=True,
        help="Train the LSTM world model f_φ(s_t, a_t, h_t) → ŝ_{t+1} on the current trainee's trajectory windows.",
    ):
        run_training(state, params)
    history = state.get("history")
    if history is not None:
        render_results(history)
    page_footer()


render()
