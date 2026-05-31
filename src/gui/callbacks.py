"""§7 Live-training observer callbacks for the Streamlit GUI.

Trainer code is *not* modified. We wrap the SDK's train methods in a
1-step-at-a-time loop and forward per-step metrics to a
:class:`LiveTrainingCallback`; pages plug ``streamlit_progress_callback``
in to stream a progress bar + per-step line chart update.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.model.lstm_trainer import LSTMTrainer
from src.model.types import LSTMTrainHistory, TransitionWindow
from src.sdk.sdk import WorkoutSDK
from src.services.a2c_types import A2CHistory
from src.services.types import REINFORCEHistory


@dataclass
class LiveTrainingCallback:
    """Called once per training step (epoch for LSTM, episode for REINFORCE/A2C)."""

    on_step: Callable[[int, dict], None]
    on_done: Callable[[Any], None]


def streamlit_progress_callback(total: int, chart_placeholder, progress_placeholder) -> LiveTrainingCallback:
    """Build a callback that updates a Streamlit progress bar + per-step line chart."""
    total = max(1, int(total))
    series: dict[str, list[float]] = {}

    def _on_step(step_idx: int, metrics: dict) -> None:
        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            series.setdefault(key, []).append(numeric)
        progress_placeholder.progress(min(1.0, (step_idx + 1) / total), text=f"step {step_idx + 1} / {total}")
        if series:
            chart_placeholder.line_chart(series)

    return LiveTrainingCallback(
        on_step=_on_step,
        on_done=lambda _f: progress_placeholder.progress(1.0, text=f"done · {total} steps"),
    )


def _stream(episodes: int, step_fn, merge_fn, metrics_fn, cb: LiveTrainingCallback):
    """Run ``episodes`` single-step trains, merging history and pushing metrics."""
    merged = None
    for i in range(int(episodes)):
        h = step_fn()
        merged = h if merged is None else merge_fn(merged, h)
        cb.on_step(i, metrics_fn(h))
    cb.on_done(merged)
    return merged


def _merge_reinforce(p: REINFORCEHistory, n: REINFORCEHistory) -> REINFORCEHistory:
    return REINFORCEHistory(
        p.episodes_run + n.episodes_run,
        p.rewards + n.rewards,
        p.losses + n.losses,
        p.baseline + n.baseline,
        p.seed,
    )


def _merge_a2c(p: A2CHistory, n: A2CHistory) -> A2CHistory:
    return A2CHistory(
        p.episodes_run + n.episodes_run,
        p.rewards + n.rewards,
        p.actor_losses + n.actor_losses,
        p.critic_losses + n.critic_losses,
        p.advantages_mean + n.advantages_mean,
        p.seed,
    )


def train_reinforce_live(sdk: WorkoutSDK, episodes: int, callback: LiveTrainingCallback) -> REINFORCEHistory:
    """Wrap ``sdk.train_reinforce`` so per-episode metrics stream live."""
    return _stream(
        episodes,
        lambda: sdk.train_reinforce(episodes=1)[1],
        _merge_reinforce,
        lambda h: {"reward": h.rewards[-1], "loss": h.losses[-1], "baseline": h.baseline[-1]},
        callback,
    )


def train_a2c_live(sdk: WorkoutSDK, episodes: int, callback: LiveTrainingCallback) -> A2CHistory:
    """Wrap ``sdk.train_a2c`` so per-episode metrics stream live."""
    return _stream(
        episodes,
        lambda: sdk.train_a2c(episodes=1)[1],
        _merge_a2c,
        lambda h: {
            "reward": h.rewards[-1],
            "actor_loss": h.actor_losses[-1],
            "critic_loss": h.critic_losses[-1],
        },
        callback,
    )


def train_lstm_live(
    trainer: LSTMTrainer,
    train_windows: list[TransitionWindow],
    val_windows: list[TransitionWindow],
    epochs: int,
    callback: LiveTrainingCallback,
) -> LSTMTrainHistory:
    """Per-epoch wrapper around ``LSTMTrainer`` so metrics stream live."""
    tls, vls, best_ep, best_v = [], [], 0, math.inf
    for epoch in range(int(epochs)):
        tl = trainer._train_one_epoch(train_windows)
        vl = trainer._eval_loss(val_windows)
        tls.append(tl)
        vls.append(vl)
        if math.isfinite(vl) and vl < best_v:
            best_v, best_ep = vl, epoch
        callback.on_step(epoch, {"train_loss": tl, "val_loss": vl})
    history = LSTMTrainHistory(
        epochs_run=int(epochs), train_loss=tuple(tls), val_loss=tuple(vls), best_epoch=best_ep
    )
    callback.on_done(history)
    return history
