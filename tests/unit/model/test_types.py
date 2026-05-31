"""Tests for src/model/types.py — LSTM config defaults align with config.yaml + frozen invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.model.types import LSTMTrainConfig, LSTMTrainHistory, TransitionWindow

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def test_lstm_train_config_defaults_match_config_yaml():
    cfg = yaml.safe_load(_CONFIG_PATH.read_text())
    lstm_cfg = cfg["lstm"]
    defaults = LSTMTrainConfig()
    assert defaults.hidden_size == lstm_cfg["hidden_size"] == 64
    assert defaults.num_layers == lstm_cfg["num_layers"] == 1
    assert defaults.dropout == lstm_cfg["dropout"] == 0.0
    assert defaults.lr == float(lstm_cfg["lr"]) == 1e-3
    assert defaults.epochs == lstm_cfg["epochs"] == 50
    # config.yaml uses `window` (rolling-window length); LSTMTrainConfig calls it window_len
    assert defaults.window_len == lstm_cfg["window"] == 7
    assert defaults.batch_size == lstm_cfg["batch_size"] == 16
    assert defaults.val_split_days == lstm_cfg["val_split_days"] == 7


def test_lstm_train_history_final_val_loss_property():
    hist = LSTMTrainHistory(
        epochs_run=3,
        train_loss=(0.9, 0.6, 0.4),
        val_loss=(1.0, 0.7, 0.5),
        best_epoch=2,
    )
    assert hist.final_val_loss == 0.5
    empty = LSTMTrainHistory(epochs_run=0, train_loss=(), val_loss=(), best_epoch=-1)
    assert np.isnan(empty.final_val_loss)


def test_transition_window_is_frozen():
    w = TransitionWindow(
        state_seq=np.zeros((7, 12), dtype=np.float32),
        action_seq=np.zeros((7,), dtype=np.int64),
        next_state=np.zeros((12,), dtype=np.float32),
    )
    with pytest.raises(FrozenInstanceError):
        w.next_state = np.ones((12,), dtype=np.float32)  # type: ignore[misc]
