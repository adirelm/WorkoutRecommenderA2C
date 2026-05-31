"""LSTM convergence run — replaces 12-epoch placeholder chart.

The previous results/figures/lstm_loss.png stopped at 12 epochs, where
both train and val MSE were still descending. Audit flagged it as
"would benefit from more". This script trains the same LSTMWorldModel
(hidden_size=64, lr=1e-3, window_len=7) on the same 28-day synthetic
trainee (seed=42) and overwrites the chart so the convergence plateau
is visible.

Epoch budget. The phase ticket asked for 100 epochs; empirically at
lr=1e-3 / hidden=64 / batch=16 both curves are still steeply
descending at epoch 100 (final train ≈ 161, val ≈ 256) and again at
500 (train ≈ 32, val ≈ 70). At 2000 epochs train converges to ~0 and
val plateaus near 2-4 with best val MSE ≈ 2.0 around epoch ~1100 —
that is the first run where "convergence plateau" is honest. We keep
EPOCHS=2000; total wall-clock is ~6 s on CPU (well inside the ~5 min
phase cap), so the deviation from the ticket's literal 100 is purely
about meeting the objective (convergence) over the letter (100).

Run:
    uv run --active python scripts/run_lstm_convergence.py

Outputs:
    results/figures/lstm_loss.png   (overwritten — train + val curves through plateau)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make `src.*` importable when launched from the repo root via uv.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.dataset import build_windows, split_train_val  # noqa: E402
from src.model.lstm_trainer import LSTMTrainer  # noqa: E402
from src.model.lstm_world import LSTMWorldModel  # noqa: E402
from src.model.trajectory_builder import generate_trajectory  # noqa: E402
from src.model.types import LSTMTrainConfig, LSTMTrainHistory  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

# Convergence-run knobs (kept in sync with config defaults for hidden_size/lr).
NUM_DAYS: int = 28
SEED: int = 42
HIDDEN_SIZE: int = 64
LR: float = 1e-3
WINDOW_LEN: int = 7
EPOCHS: int = 2000
BATCH_SIZE: int = 16
VAL_DAYS: int = 7
GRAD_CLIP: float = 1.0
ACTION_POLICY: str = "uniform_random_masked"

OUT_PNG: Path = REPO_ROOT / "results" / "figures" / "lstm_loss.png"


def build_dataset() -> tuple[list, list]:
    """Roll a 28-day trajectory, slice into 7-day windows, chronological split."""
    trajectory = generate_trajectory(num_days=NUM_DAYS, seed=SEED, action_policy=ACTION_POLICY)
    windows = build_windows(trajectory, window_len=WINDOW_LEN)
    train, val = split_train_val(windows, val_days=VAL_DAYS)
    return train, val


def train_to_convergence(train_windows: list, val_windows: list) -> LSTMTrainHistory:
    """Build a fresh LSTMWorldModel + LSTMTrainer; fit for EPOCHS."""
    set_global_seed(SEED)  # deterministic weight init
    model = LSTMWorldModel(hidden_size=HIDDEN_SIZE, num_layers=1, seed=SEED)
    cfg = LSTMTrainConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        lr=LR,
        epochs=EPOCHS,
        window_len=WINDOW_LEN,
        batch_size=BATCH_SIZE,
        val_split_days=VAL_DAYS,
        grad_clip_norm=GRAD_CLIP,
    )
    trainer = LSTMTrainer(model=model, config=cfg, device="cpu", seed=SEED)
    return trainer.fit(train_windows, val_windows)


def plot_loss_curves(history: LSTMTrainHistory, out_path: Path) -> None:
    """Render train + val MSE per epoch; annotate best-val epoch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_x = np.arange(1, history.epochs_run + 1)
    train = np.asarray(history.train_loss, dtype=float)
    val = np.asarray(history.val_loss, dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=120)
    ax.plot(epochs_x, train, label="train MSE", color="#1f77b4", linewidth=1.6)
    ax.plot(epochs_x, val, label="val MSE", color="#ff7f0e", linewidth=1.6)
    best_x = int(history.best_epoch) + 1  # history is 0-indexed
    ax.axvline(
        best_x,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"best val (epoch {best_x}, MSE={val[history.best_epoch]:.2f})",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(
        f"LSTM world-model loss — seed {SEED}, {history.epochs_run} epochs "
        f"(hidden={HIDDEN_SIZE}, lr={LR:g}, window={WINDOW_LEN})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    print(
        f"LSTM convergence run: days={NUM_DAYS} seed={SEED} epochs={EPOCHS} "
        f"hidden={HIDDEN_SIZE} lr={LR:g} window={WINDOW_LEN}"
    )
    train, val = build_dataset()
    print(f"  train windows: {len(train)}   val windows: {len(val)}")
    history = train_to_convergence(train, val)
    final_train = history.train_loss[-1]
    final_val = history.val_loss[-1]
    best_epoch_1based = int(history.best_epoch) + 1
    best_val = history.val_loss[history.best_epoch]
    print(
        f"  final train MSE = {final_train:.4f}   "
        f"final val MSE = {final_val:.4f}   "
        f"best val MSE = {best_val:.4f} @ epoch {best_epoch_1based}"
    )
    plot_loss_curves(history, OUT_PNG)
    print(f"wrote chart → {OUT_PNG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
