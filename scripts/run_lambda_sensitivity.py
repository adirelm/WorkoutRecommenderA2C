"""Lambda sensitivity sweep — closes V3 §9.1 "no λ_1×λ_2 sensitivity analysis".

Sweeps the eq.-15 reward weights λ_1 (overload) × λ_2 (imbalance) over a
3×3 grid; for each (λ_1, λ_2) pair, trains REINFORCE for SHORT_EPISODES
under a fixed seed and records the mean episodic reward over the *last
window* of episodes. The 3×3 final-mean matrix is written as a heatmap to
results/figures/lambda_sensitivity.png.

This is an evidence-of-methodology pilot, NOT a publication-grade sweep:
single seed, single short run per cell, no statistical bands. The README
of EXPERIMENTS.md flags it as such. The goal is to show that *we know
how to do the sweep* and that λ values measurably move the policy.

Run:
    uv run --active python scripts/run_lambda_sensitivity.py

Outputs:
    results/figures/lambda_sensitivity.png   (3×3 heatmap)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Make `src.*` importable when launched from the repo root via uv.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.reward import RewardConfig  # noqa: E402
from src.env.workout_env import WorkoutEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.reinforce_trainer import REINFORCETrainer  # noqa: E402
from src.services.types import REINFORCEConfig  # noqa: E402

# Sweep grid (V3 §9.1). 3×3 = 9 short runs; defaults (λ_1=2.0, λ_2=1.0) are inside.
LAMBDA_1_GRID: tuple[float, ...] = (1.0, 2.0, 3.0)
LAMBDA_2_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)

SHORT_EPISODES: int = 20  # pilot run — reduce to 10 if wall-clock > 5 min
WINDOW: int = 5  # mean of last-N episode rewards (smooths late variance)
SEED: int = 42

OUT_PNG: Path = REPO_ROOT / "results" / "figures" / "lambda_sensitivity.png"
CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"


def load_reward_defaults() -> dict:
    """Load reward.* from config/config.yaml so the pilot mirrors the live env."""
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg.get("rewards", {}))


def build_reward_config(defaults: dict, lambda_1: float, lambda_2: float) -> RewardConfig:
    """Construct a RewardConfig with sweep λs but config defaults elsewhere."""
    return RewardConfig(
        lambda_1=float(lambda_1),
        lambda_2=float(lambda_2),
        w_progress=float(defaults.get("w_progress", 0.7)),
        w_variety=float(defaults.get("w_variety", 0.3)),
        overload_threshold_mult=float(defaults.get("overload_threshold_multiplier", 1.2)),
        overload_exponent=float(defaults.get("overload_exponent", 1.5)),
    )


def train_one_cell(lambda_1: float, lambda_2: float, defaults: dict) -> float:
    """Train REINFORCE once with the given λ pair; return mean of last WINDOW rewards."""
    reward_cfg = build_reward_config(defaults, lambda_1, lambda_2)
    env = WorkoutEnv(reward_config=reward_cfg, seed=SEED)
    policy = PolicyNet(seed=SEED)
    trainer_cfg = REINFORCEConfig(episodes=SHORT_EPISODES)
    history = REINFORCETrainer(policy, env, trainer_cfg, seed=SEED).train(episodes=SHORT_EPISODES)
    tail = history.rewards[-WINDOW:] if len(history.rewards) >= WINDOW else history.rewards
    return float(np.mean(tail))


def run_sweep() -> np.ndarray:
    """Iterate the 3×3 grid, return a matrix shaped (len(L1), len(L2)) of mean rewards."""
    defaults = load_reward_defaults()
    matrix = np.zeros((len(LAMBDA_1_GRID), len(LAMBDA_2_GRID)), dtype=float)
    for i, l1 in enumerate(LAMBDA_1_GRID):
        for j, l2 in enumerate(LAMBDA_2_GRID):
            mean_reward = train_one_cell(l1, l2, defaults)
            matrix[i, j] = mean_reward
            print(f"  λ_1={l1:.2f}  λ_2={l2:.2f}  →  mean(last {WINDOW}) = {mean_reward:+.3f}")
    return matrix


def plot_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    """Render the 3×3 heatmap with annotated cells; save as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=120)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(LAMBDA_2_GRID)), [f"{v:g}" for v in LAMBDA_2_GRID])
    ax.set_yticks(range(len(LAMBDA_1_GRID)), [f"{v:g}" for v in LAMBDA_1_GRID])
    ax.set_xlabel(r"$\lambda_2$ (imbalance weight)")
    ax.set_ylabel(r"$\lambda_1$ (overload weight)")
    ax.set_title(
        f"λ sensitivity — REINFORCE mean reward (last {WINDOW} of {SHORT_EPISODES} eps, seed={SEED})"
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < matrix.mean() else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, label="mean episodic reward")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    print(
        f"Running λ sensitivity sweep: "
        f"{len(LAMBDA_1_GRID)}×{len(LAMBDA_2_GRID)}={len(LAMBDA_1_GRID) * len(LAMBDA_2_GRID)} "
        f"cells × {SHORT_EPISODES} episodes, seed={SEED}"
    )
    matrix = run_sweep()
    plot_heatmap(matrix, OUT_PNG)
    print(f"wrote heatmap → {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"matrix:\n{matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
