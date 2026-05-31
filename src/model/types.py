"""Typed dataclasses for the model layer (Phase 3+: LSTM world model + policy + critic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionWindow:
    """One supervised training window for the LSTM world model.

    state_seq: shape (window_len, STATE_DIM) — past states
    action_seq: shape (window_len,) — past action ids
    next_state: shape (STATE_DIM,) — target (the state to predict)
    """

    state_seq: np.ndarray
    action_seq: np.ndarray
    next_state: np.ndarray

    def __post_init__(self):
        # Frozen dataclass — can't set in __post_init__, but assert shapes via property checks elsewhere
        pass


@dataclass(frozen=True)
class LSTMTrainConfig:
    """LSTM hyperparameters per config/config.yaml."""

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    epochs: int = 50
    window_len: int = 7
    batch_size: int = 16
    val_split_days: int = 7
    grad_clip_norm: float = 1.0
    action_embed_dim: int = 8


@dataclass(frozen=True)
class LSTMTrainHistory:
    """Per-epoch training trace for the LSTM."""

    epochs_run: int
    train_loss: tuple[float, ...]  # length = epochs_run
    val_loss: tuple[float, ...]
    best_epoch: int  # epoch with lowest val_loss

    @property
    def final_val_loss(self) -> float:
        """Last-epoch validation MSE; NaN if val set was empty."""
        return self.val_loss[-1] if self.val_loss else float("nan")
