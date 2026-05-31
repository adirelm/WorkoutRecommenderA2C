"""§7.3 LSTM world-model training page (Phase 9 GUI).

ADR-006 GUI-layer exception: :class:`WorkoutSDK` does not expose
``train_world_model`` (it raises ``NotImplementedError`` — the SDK is
policy-only), so this page instantiates :class:`LSTMWorldModel` +
:class:`LSTMTrainer` directly to live-stream the loss curve. The
trajectory comes from ``src.model.trajectory_builder.generate_trajectory``
(it returns the ``(state, action, next_state)`` triples that
``build_windows`` expects) and never leaves the GUI layer. All other
pages stay SDK-only.
"""

from __future__ import annotations

import streamlit as st

from src.gui import charts
from src.gui.components import hero, info_card, latex_block, metric_row, page_footer
from src.gui.state import GUIState, get_sdk, set_last_world_model_history
from src.model.dataset import build_windows, split_train_val
from src.model.lstm_trainer import LSTMTrainer
from src.model.lstm_world import LSTMWorldModel
from src.model.trajectory_builder import generate_trajectory
from src.model.types import LSTMTrainConfig, LSTMTrainHistory

_HIDDEN_OPTIONS: tuple[int, ...] = (16, 32, 64, 128)
_LAYERS_OPTIONS: tuple[int, ...] = (1, 2)


def _sidebar_controls() -> dict[str, float | int]:
    """Render the sidebar hyperparameter widgets and return their values."""
    st.sidebar.header("LSTM hyperparameters")
    hidden_size = st.sidebar.select_slider("hidden_size", options=list(_HIDDEN_OPTIONS), value=64)
    num_layers = st.sidebar.select_slider("num_layers", options=list(_LAYERS_OPTIONS), value=1)
    epochs = st.sidebar.slider("epochs", min_value=5, max_value=50, value=15, step=1)
    lr = st.sidebar.select_slider(
        "lr",
        options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        value=1e-3,
        format_func=lambda v: f"{v:.0e}",
    )
    window_len = st.sidebar.slider("window_len", min_value=3, max_value=14, value=7, step=1)
    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "epochs": int(epochs),
        "lr": float(lr),
        "window_len": int(window_len),
    }


def _build_config(params: dict[str, float | int]) -> LSTMTrainConfig:
    """Translate sidebar values into the typed :class:`LSTMTrainConfig`."""
    return LSTMTrainConfig(
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        lr=float(params["lr"]),
        epochs=int(params["epochs"]),
        window_len=int(params["window_len"]),
    )


def _train_live(trainer: LSTMTrainer, train_w, val_w, epochs: int) -> LSTMTrainHistory:
    """Loop ``trainer._train_one_epoch`` and refresh a live plotly chart."""
    chart_slot = st.empty()
    progress = st.progress(0.0, text=f"epoch 0 / {epochs}")
    tr: list[float] = []
    va: list[float] = []
    best_epoch, best_val = 0, float("inf")
    for ep in range(epochs):
        tr.append(trainer._train_one_epoch(train_w))
        va.append(trainer._eval_loss(val_w))
        if va[-1] == va[-1] and va[-1] < best_val:  # NaN-safe min
            best_val, best_epoch = va[-1], ep
        snap = LSTMTrainHistory(ep + 1, tuple(tr), tuple(va), best_epoch)
        chart_slot.plotly_chart(
            charts.loss_curve(snap, title="LSTM training loss (live)"),
            use_container_width=True,
        )
        progress.progress((ep + 1) / epochs, text=f"epoch {ep + 1} / {epochs}")
    return LSTMTrainHistory(epochs, tuple(tr), tuple(va), best_epoch)


def _run_training(state: GUIState, params: dict[str, float | int]) -> None:
    """Build dataset + model, then drive the live training loop."""
    # Use the proper (state, action, next_state) trajectory builder — ``build_windows``
    # indexes each triple as ``t[0]/t[1]/t[2]``, so passing raw states would crash with
    # ``TypeError: 'State' object is not subscriptable``.
    trajectory = generate_trajectory(num_days=28, seed=get_sdk().seed)
    windows = build_windows(trajectory, window_len=int(params["window_len"]))
    train_w, val_w = split_train_val(windows, val_days=7)
    if not train_w or not val_w:
        st.error("Trajectory too short for this window_len / val split — shrink window_len.")
        return
    model = LSTMWorldModel(
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        seed=42,
    )
    trainer = LSTMTrainer(model=model, config=_build_config(params), seed=42)
    with st.spinner("Training LSTM world model…"):
        history = _train_live(trainer, train_w, val_w, epochs=int(params["epochs"]))
    state.set("history", history)
    set_last_world_model_history(history)
    st.success("Training complete.")


def _render_results(history: LSTMTrainHistory) -> None:
    """Final metric strip + frozen loss chart after a completed run."""
    st.subheader("Final metrics")
    metric_row(
        {
            "epochs_run": str(history.epochs_run),
            "final_train_loss": f"{history.train_loss[-1]:.4f}",
            "final_val_loss": f"{history.final_val_loss:.4f}",
            "best_epoch": str(history.best_epoch + 1),
        }
    )
    st.plotly_chart(charts.loss_curve(history), use_container_width=True)


def render() -> None:
    """Render the LSTM training page in full."""
    state = GUIState("lstm")
    hero("LSTM world model", "§7.3 — recurrent transition model f_φ", icon="🧠")
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
    params = _sidebar_controls()
    if st.button("Train LSTM", type="primary", use_container_width=True):
        _run_training(state, params)
    history = state.get("history")
    if history is not None:
        _render_results(history)
    page_footer()


render()
