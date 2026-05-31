"""Helpers for §7.3 LSTM page — kept separate so 03_lstm.py stays ≤150 LOC."""

from __future__ import annotations

import streamlit as st

from src.gui import charts
from src.gui.components import metric_row
from src.gui.state import GUIState, get_sdk, set_last_world_model_history
from src.model.dataset import build_windows, split_train_val
from src.model.lstm_trainer import LSTMTrainer
from src.model.lstm_world import LSTMWorldModel
from src.model.trajectory_builder import generate_trajectory
from src.model.types import LSTMTrainConfig, LSTMTrainHistory

HIDDEN_OPTIONS: tuple[int, ...] = (16, 32, 64, 128)
LAYERS_OPTIONS: tuple[int, ...] = (1, 2)

HELP_HIDDEN = "LSTM hidden-state dimension (more = more capacity, slower)."
HELP_LAYERS = "Stacked LSTM depth (1 is usually enough for this dataset)."
HELP_EPOCHS = "How many full passes over the training windows (best @ N annotation shows the best epoch)."
HELP_LR = "Adam learning rate. 1e-3 is the default; smaller = more stable."
HELP_WINDOW = "Number of consecutive days per training window."


def sidebar_controls() -> dict[str, float | int]:
    """Render the sidebar hyperparameter widgets and return their values."""
    sb = st.sidebar
    sb.header("LSTM Hyperparameters")
    hidden_size = sb.select_slider("Hidden size", options=list(HIDDEN_OPTIONS), value=64, help=HELP_HIDDEN)
    num_layers = sb.select_slider("Number of layers", options=list(LAYERS_OPTIONS), value=1, help=HELP_LAYERS)
    epochs = sb.slider("Training epochs", min_value=5, max_value=200, value=15, step=5, help=HELP_EPOCHS)
    lr = sb.select_slider(
        "Learning rate",
        options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        value=1e-3,
        format_func=lambda v: f"{v:.0e}",
        help=HELP_LR,
    )
    window_len = sb.slider(
        "Window length (days)", min_value=3, max_value=14, value=7, step=1, help=HELP_WINDOW
    )
    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "epochs": int(epochs),
        "lr": float(lr),
        "window_len": int(window_len),
    }


def _build_config(params: dict[str, float | int]) -> LSTMTrainConfig:
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


def run_training(state: GUIState, params: dict[str, float | int]) -> None:
    """Build dataset + model, then drive the live training loop."""
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


def render_results(history: LSTMTrainHistory) -> None:
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
