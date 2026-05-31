"""Tests for src/model/lstm_trainer.py — supervised loop over LSTMWorldModel."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.utils as torch_nn_utils

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.lstm_trainer import LSTMTrainer
from src.model.lstm_world import LSTMWorldModel
from src.model.types import LSTMTrainConfig, TransitionWindow
from src.utils.seeding import set_global_seed


def _make_windows(n: int, window_len: int = 5, seed: int = 0) -> list[TransitionWindow]:
    """Synthesize a small toy dataset of TransitionWindows."""
    rng = np.random.default_rng(seed)
    out: list[TransitionWindow] = []
    for _ in range(n):
        state_seq = rng.standard_normal((window_len, STATE_DIM)).astype(np.float32)
        action_seq = rng.integers(0, ACTION_COUNT, size=(window_len,)).astype(np.int64)
        next_state = rng.standard_normal((STATE_DIM,)).astype(np.float32)
        out.append(TransitionWindow(state_seq=state_seq, action_seq=action_seq, next_state=next_state))
    return out


def _trainer(epochs: int = 3, seed: int = 42, batch_size: int = 4) -> LSTMTrainer:
    cfg = LSTMTrainConfig(
        hidden_size=16,
        num_layers=1,
        lr=1e-2,
        epochs=epochs,
        batch_size=batch_size,
        grad_clip_norm=1.0,
    )
    # Seed BEFORE constructing the model so weight init is also deterministic.
    set_global_seed(seed)
    model = LSTMWorldModel(hidden_size=16, num_layers=1)
    return LSTMTrainer(model=model, config=cfg, device="cpu", seed=seed)


def test_fit_returns_history_with_epoch_count_epochs():
    trainer = _trainer(epochs=3)
    train = _make_windows(12, seed=1)
    val = _make_windows(4, seed=2)
    history = trainer.fit(train, val)
    assert history.epochs_run == 3
    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3


def test_fit_train_loss_monotone_average():
    trainer = _trainer(epochs=5)
    train = _make_windows(20, seed=11)
    val = _make_windows(4, seed=12)
    history = trainer.fit(train, val)
    first_two = history.train_loss[:2]
    last_two = history.train_loss[-2:]
    assert sum(last_two) / 2.0 < sum(first_two) / 2.0, (
        f"loss did not decrease on average: first={first_two} last={last_two}"
    )


def test_fit_val_loss_finite_no_nan():
    trainer = _trainer(epochs=3)
    train = _make_windows(10, seed=21)
    val = _make_windows(5, seed=22)
    history = trainer.fit(train, val)
    for v in history.val_loss:
        assert math.isfinite(v), f"val_loss has non-finite value: {history.val_loss}"


def test_predict_returns_correct_shape():
    trainer = _trainer(epochs=1)
    batch, seq = 3, 5
    state_seq = torch.randn(batch, seq, STATE_DIM, dtype=torch.float32)
    action_seq = torch.randint(0, ACTION_COUNT, size=(batch, seq), dtype=torch.long)
    pred = trainer.predict(state_seq, action_seq)
    assert pred.shape == (batch, STATE_DIM)


def test_seeded_trainer_deterministic():
    train = _make_windows(8, seed=31)
    val = _make_windows(3, seed=32)
    t1 = _trainer(epochs=3, seed=123)
    h1 = t1.fit(train, val)
    t2 = _trainer(epochs=3, seed=123)
    h2 = t2.fit(train, val)
    assert h1.train_loss == h2.train_loss, f"{h1.train_loss} != {h2.train_loss}"


def test_grad_clip_applied(monkeypatch):
    trainer = _trainer(epochs=2, batch_size=4)
    # 8 windows / batch_size=4 = 2 batches per epoch, 2 epochs = 4 calls expected.
    train = _make_windows(8, seed=41)
    val = _make_windows(2, seed=42)

    calls = {"n": 0}
    real_clip = torch_nn_utils.clip_grad_norm_

    def _spy(parameters, max_norm, *a, **kw):
        calls["n"] += 1
        return real_clip(parameters, max_norm, *a, **kw)

    monkeypatch.setattr("src.model.lstm_trainer.nn.utils.clip_grad_norm_", _spy)
    trainer.fit(train, val)
    assert calls["n"] == 4, f"expected 4 grad-clip calls (one per batch), got {calls['n']}"


def test_fit_handles_empty_val_gracefully():
    trainer = _trainer(epochs=2)
    train = _make_windows(8, seed=51)
    history = trainer.fit(train, val_windows=[])
    assert len(history.val_loss) == 2
    for v in history.val_loss:
        assert math.isnan(v), f"empty val should produce NaN, got {history.val_loss}"
