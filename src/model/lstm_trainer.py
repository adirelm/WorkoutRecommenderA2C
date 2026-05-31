"""Supervised LSTM training loop (Phase 3, brief §7.3.1).

Standard PyTorch fit loop over `list[TransitionWindow]`:
    for epoch in range(epochs):
        shuffle(train_windows)
        for batch in mini_batches(train_windows, batch_size):
            optimizer.zero_grad()
            pred = model(state_seq, action_seq)
            loss = torch_f.mse_loss(pred, next_state)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        log train + val MSE

History is returned at the end — no early-abort, no checkpoint side-effects.
The seed is funnelled through `set_global_seed` for bit-reproducible runs.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_f

from src.model.lstm_world import LSTMWorldModel
from src.model.types import LSTMTrainConfig, LSTMTrainHistory, TransitionWindow
from src.utils.seeding import set_global_seed


def _stack_batch(
    windows: list[TransitionWindow], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of TransitionWindow into (state_seq, action_seq, next_state) tensors."""
    state_seq = torch.from_numpy(np.stack([w.state_seq for w in windows], axis=0)).to(
        device=device, dtype=torch.float32
    )
    action_seq = torch.from_numpy(np.stack([w.action_seq for w in windows], axis=0)).to(
        device=device, dtype=torch.long
    )
    next_state = torch.from_numpy(np.stack([w.next_state for w in windows], axis=0)).to(
        device=device, dtype=torch.float32
    )
    return state_seq, action_seq, next_state


class LSTMTrainer:
    """Wraps a `LSTMWorldModel` with an Adam + grad-clip supervised loop."""

    def __init__(
        self,
        model: LSTMWorldModel,
        config: LSTMTrainConfig,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        set_global_seed(seed)
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.seed = seed
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # Use a dedicated numpy RNG for deterministic batch shuffling
        # (independent of any later np.random usage in tests).
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------- helpers
    def _eval_loss(self, windows: list[TransitionWindow]) -> float:
        if not windows:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            state_seq, action_seq, next_state = _stack_batch(windows, self.device)
            pred = self.model(state_seq, action_seq)
            return float(torch_f.mse_loss(pred, next_state).item())

    def _train_one_epoch(self, train_windows: list[TransitionWindow]) -> float:
        self.model.train()
        n = len(train_windows)
        if n == 0:
            return float("nan")
        order = self._rng.permutation(n)
        batch_size = max(1, int(self.config.batch_size))
        total_loss = 0.0
        total_count = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = [train_windows[i] for i in idx]
            state_seq, action_seq, next_state = _stack_batch(batch, self.device)
            self.optimizer.zero_grad()
            pred = self.model(state_seq, action_seq)
            loss = torch_f.mse_loss(pred, next_state)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)
        return total_loss / max(1, total_count)

    # --------------------------------------------------------------------- fit
    def fit(
        self,
        train_windows: list[TransitionWindow],
        val_windows: list[TransitionWindow],
    ) -> LSTMTrainHistory:
        """Train for `config.epochs`; return per-epoch losses + best epoch."""
        epochs = int(self.config.epochs)
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = 0
        best_val = math.inf
        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_windows)
            val_loss = self._eval_loss(val_windows)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
        return LSTMTrainHistory(
            epochs_run=epochs,
            train_loss=tuple(train_losses),
            val_loss=tuple(val_losses),
            best_epoch=best_epoch,
        )

    # ----------------------------------------------------------------- predict
    def predict(self, state_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Inference wrapper — model in eval mode, no grad. Returns (batch, STATE_DIM)."""
        self.model.eval()
        with torch.no_grad():
            return self.model(state_seq.to(self.device), action_seq.to(self.device))
