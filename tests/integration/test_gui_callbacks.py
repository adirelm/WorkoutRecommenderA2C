"""Tests for src/gui/callbacks.py — §7 live-training observer pattern.

Verifies the `LiveTrainingCallback` shape, the Streamlit placeholder
adapter, and that `train_*_live` wrappers stream per-step metrics +
a final `on_done` event. Trainer / SDK calls are mocked."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.gui.callbacks import (
    LiveTrainingCallback,
    streamlit_progress_callback,
    train_a2c_live,
    train_lstm_live,
    train_reinforce_live,
)
from src.model.types import LSTMTrainHistory
from src.services.a2c_types import A2CHistory
from src.services.types import REINFORCEHistory


def _reinforce_hist(reward: float, loss: float, baseline: float) -> REINFORCEHistory:
    return REINFORCEHistory(episodes_run=1, rewards=(reward,), losses=(loss,), baseline=(baseline,), seed=0)


def _a2c_hist(reward: float, al: float, cl: float) -> A2CHistory:
    return A2CHistory(
        episodes_run=1,
        rewards=(reward,),
        actor_losses=(al,),
        critic_losses=(cl,),
        advantages_mean=(0.0,),
        seed=0,
    )


def test_live_training_callback_is_dataclass_with_step_and_done_fields():
    assert dataclasses.is_dataclass(LiveTrainingCallback)
    field_names = {f.name for f in dataclasses.fields(LiveTrainingCallback)}
    assert field_names == {"on_step", "on_done"}
    cb = LiveTrainingCallback(on_step=lambda i, m: None, on_done=lambda h: None)
    assert callable(cb.on_step) and callable(cb.on_done)


def test_streamlit_progress_callback_updates_placeholders_per_step():
    chart, progress = MagicMock(), MagicMock()
    cb = streamlit_progress_callback(total=3, chart_placeholder=chart, progress_placeholder=progress)
    cb.on_step(0, {"reward": 1.0, "loss": 0.5})
    cb.on_step(1, {"reward": 2.0, "loss": 0.4})
    assert progress.progress.call_count == 2
    # frac argument grows monotonically toward 1.0
    fracs = [c.args[0] for c in progress.progress.call_args_list]
    assert fracs[0] < fracs[1] <= 1.0
    assert chart.line_chart.call_count == 2
    cb.on_done(None)
    progress.progress.assert_called_with(1.0, text="done · 3 steps")


def test_streamlit_progress_callback_ignores_non_numeric_metrics():
    chart, progress = MagicMock(), MagicMock()
    cb = streamlit_progress_callback(total=1, chart_placeholder=chart, progress_placeholder=progress)
    cb.on_step(0, {"reward": 1.0, "note": "skip-me"})
    plotted = chart.line_chart.call_args.args[0]
    assert "reward" in plotted and "note" not in plotted


def test_train_reinforce_live_invokes_on_step_per_episode_and_on_done_once():
    sdk = MagicMock()
    sdk.train_reinforce.side_effect = [
        (MagicMock(), _reinforce_hist(1.0, 0.1, 0.5)),
        (MagicMock(), _reinforce_hist(2.0, 0.2, 0.7)),
        (MagicMock(), _reinforce_hist(3.0, 0.3, 0.9)),
    ]
    on_step, on_done = MagicMock(), MagicMock()
    cb = LiveTrainingCallback(on_step=on_step, on_done=on_done)
    merged = train_reinforce_live(sdk, episodes=3, callback=cb)
    assert sdk.train_reinforce.call_count == 3
    assert on_step.call_count == 3
    assert [c.args[0] for c in on_step.call_args_list] == [0, 1, 2]
    assert on_step.call_args_list[0].args[1] == {"reward": 1.0, "loss": 0.1, "baseline": 0.5}
    on_done.assert_called_once_with(merged)
    assert merged.episodes_run == 3 and merged.rewards == (1.0, 2.0, 3.0)


def test_train_a2c_live_invokes_on_step_per_episode_and_on_done_once():
    sdk = MagicMock()
    sdk.train_a2c.side_effect = [
        (MagicMock(), _a2c_hist(1.0, 0.1, 0.2)),
        (MagicMock(), _a2c_hist(2.0, 0.3, 0.4)),
    ]
    on_step, on_done = MagicMock(), MagicMock()
    cb = LiveTrainingCallback(on_step=on_step, on_done=on_done)
    merged = train_a2c_live(sdk, episodes=2, callback=cb)
    assert on_step.call_count == 2
    assert on_step.call_args_list[1].args == (1, {"reward": 2.0, "actor_loss": 0.3, "critic_loss": 0.4})
    on_done.assert_called_once_with(merged)
    assert merged.rewards == (1.0, 2.0)


def test_train_lstm_live_invokes_on_step_per_epoch_and_on_done_once():
    trainer = SimpleNamespace(
        _train_one_epoch=MagicMock(side_effect=[0.9, 0.5, 0.3]),
        _eval_loss=MagicMock(side_effect=[0.8, 0.4, 0.6]),
    )
    on_step, on_done = MagicMock(), MagicMock()
    cb = LiveTrainingCallback(on_step=on_step, on_done=on_done)
    history = train_lstm_live(trainer, train_windows=[], val_windows=[], epochs=3, callback=cb)
    assert trainer._train_one_epoch.call_count == 3
    assert trainer._eval_loss.call_count == 3
    assert on_step.call_count == 3
    assert on_step.call_args_list[0].args == (0, {"train_loss": 0.9, "val_loss": 0.8})
    assert isinstance(history, LSTMTrainHistory)
    assert history.epochs_run == 3
    assert history.best_epoch == 1  # val_loss minimum is 0.4 at epoch 1
    on_done.assert_called_once_with(history)
