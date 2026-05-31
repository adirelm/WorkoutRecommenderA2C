"""REINFORCE learning-rate sensitivity sweep — Phase 10 results-optimization pass.

Sweeps the Adam learning rate over 5 textbook decades (1e-4, 3e-4, 1e-3,
3e-3, 1e-2) × 3 seeds; for each (lr, seed) trains REINFORCE for
SHORT_EPISODES and records the mean reward over the *final WINDOW*
episodes. Plots mean ± 1σ across seeds versus lr (log-x), and saves the
raw per-(lr, seed) tail-means to JSON for downstream reuse.

This is an evidence-of-methodology pilot, NOT a publication-grade sweep:
short horizon, 3 seeds. The README-level claim being substantiated is
that the codebase's default lr (3e-4, "OpenAI default") sits at-or-near
the empirical sweet spot for this environment / horizon.

Run:
    uv run --active python scripts/run_lr_sweep.py

Outputs:
    results/figures/reinforce_lr_sweep.png   (mean ± 1σ line plot, log-x)
    results/lr_sweep.json                    (raw matrix + summary)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.workout_env import WorkoutEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.reinforce_trainer import REINFORCETrainer  # noqa: E402
from src.services.types import REINFORCEConfig  # noqa: E402

LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
SEEDS: tuple[int, ...] = (42, 123, 2026)
SHORT_EPISODES: int = 30
WINDOW: int = 5

OUT_PNG: Path = REPO_ROOT / "results" / "figures" / "reinforce_lr_sweep.png"
OUT_JSON: Path = REPO_ROOT / "results" / "lr_sweep.json"


def train_one(lr: float, seed: int) -> float:
    """Train REINFORCE once with the given lr/seed; return mean of last WINDOW rewards."""
    env = WorkoutEnv(seed=seed)
    policy = PolicyNet(seed=seed)
    cfg = REINFORCEConfig(lr=float(lr), episodes=SHORT_EPISODES)
    history = REINFORCETrainer(policy, env, cfg, seed=seed).train(episodes=SHORT_EPISODES)
    tail = history.rewards[-WINDOW:] if len(history.rewards) >= WINDOW else history.rewards
    return float(np.mean(tail))


def run_sweep() -> np.ndarray:
    """Iterate the LR_GRID × SEEDS grid; return matrix shape (len(LR_GRID), len(SEEDS))."""
    matrix = np.zeros((len(LR_GRID), len(SEEDS)), dtype=float)
    for i, lr in enumerate(LR_GRID):
        for j, seed in enumerate(SEEDS):
            t0 = time.perf_counter()
            tail_mean = train_one(lr, seed)
            elapsed = time.perf_counter() - t0
            matrix[i, j] = tail_mean
            print(
                f"  lr={lr:.0e}  seed={seed:>4d}  →  mean(last {WINDOW}) = "
                f"{tail_mean:+.3f}   [{elapsed:5.1f}s]"
            )
    return matrix


def plot_curve(matrix: np.ndarray, out_path: Path) -> None:
    """Render mean ± 1σ across seeds per lr on a log-x line plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
    ax.errorbar(
        LR_GRID,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=1.6,
        color="#1f77b4",
        ecolor="#1f77b4",
        label=f"mean ± 1σ across {len(SEEDS)} seeds",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Adam learning rate (log scale)")
    ax.set_ylabel(f"mean reward over last {WINDOW} of {SHORT_EPISODES} episodes")
    ax.set_title(
        f"REINFORCE lr sensitivity — {len(LR_GRID)} lrs × {len(SEEDS)} seeds × "
        f"{SHORT_EPISODES} eps"
    )
    ax.axvline(3e-4, linestyle="--", linewidth=1.0, color="#888888", label="default lr=3e-4")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def dump_json(matrix: np.ndarray, out_path: Path) -> None:
    """Persist per-(lr, seed) tail-means plus summary stats for downstream reuse."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lr_grid": list(LR_GRID),
        "seeds": list(SEEDS),
        "episodes": SHORT_EPISODES,
        "window": WINDOW,
        "tail_means": matrix.tolist(),
        "mean_per_lr": matrix.mean(axis=1).tolist(),
        "std_per_lr": matrix.std(axis=1).tolist(),
        "best_lr": float(LR_GRID[int(np.argmax(matrix.mean(axis=1)))]),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    print(
        f"REINFORCE lr sweep: {len(LR_GRID)} lrs × {len(SEEDS)} seeds × "
        f"{SHORT_EPISODES} eps = {len(LR_GRID) * len(SEEDS)} runs"
    )
    matrix = run_sweep()
    plot_curve(matrix, OUT_PNG)
    dump_json(matrix, OUT_JSON)
    print(f"wrote chart → {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"wrote json  → {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"mean per lr: {matrix.mean(axis=1)}")
    print(f"best lr    : {LR_GRID[int(np.argmax(matrix.mean(axis=1)))]:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
