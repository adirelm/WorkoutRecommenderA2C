"""Lambda sensitivity sweep — V3 §9.1 expansion (5×5 grid, 3 seeds/cell).

Sweeps the eq.-15 reward weights λ_1 (overload) × λ_2 (imbalance) over a
5×5 grid; for each (λ_1, λ_2) pair, trains REINFORCE for SHORT_EPISODES
under THREE seeds, then averages the per-seed mean of the last WINDOW
episode-rewards. The 5×5 final-mean matrix is rendered as a viridis
heatmap to results/figures/lambda_sensitivity.png.

This replaces the earlier 3×3 single-seed pilot. Three seeds per cell
gives a small variance estimate, though we still report only the mean
(no confidence bands on the heatmap — that's left for a future sweep).

Run:
    uv run --active python scripts/run_lambda_sensitivity.py

Outputs:
    results/figures/lambda_sensitivity.png   (5×5 heatmap, 3-seed mean)
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

# Sweep grid (V3 §9.1 expansion). 5×5 = 25 cells × 3 seeds = 75 short runs.
LAMBDA_1_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
LAMBDA_2_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)

SHORT_EPISODES: int = 20  # episodes per (cell, seed) run
WINDOW: int = 5  # mean of last-N episode rewards (smooths late variance)
SEEDS: tuple[int, ...] = (42, 43, 44)  # 3 seeds per cell

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


def train_one_run(lambda_1: float, lambda_2: float, defaults: dict, seed: int) -> float:
    """Train REINFORCE once with the given λ pair and seed; return last-WINDOW mean."""
    reward_cfg = build_reward_config(defaults, lambda_1, lambda_2)
    env = WorkoutEnv(reward_config=reward_cfg, seed=seed)
    policy = PolicyNet(seed=seed)
    trainer_cfg = REINFORCEConfig(episodes=SHORT_EPISODES)
    history = REINFORCETrainer(policy, env, trainer_cfg, seed=seed).train(episodes=SHORT_EPISODES)
    tail = history.rewards[-WINDOW:] if len(history.rewards) >= WINDOW else history.rewards
    return float(np.mean(tail))


def run_sweep() -> np.ndarray:
    """Iterate the 5×5×3 grid; return (len(L1), len(L2)) matrix of seed-averaged means."""
    defaults = load_reward_defaults()
    matrix = np.zeros((len(LAMBDA_1_GRID), len(LAMBDA_2_GRID)), dtype=float)
    for i, l1 in enumerate(LAMBDA_1_GRID):
        for j, l2 in enumerate(LAMBDA_2_GRID):
            per_seed = [train_one_run(l1, l2, defaults, s) for s in SEEDS]
            cell_mean = float(np.mean(per_seed))
            matrix[i, j] = cell_mean
            print(
                f"  λ_1={l1:.2f}  λ_2={l2:.2f}  →  "
                f"seed-means={[f'{v:+.2f}' for v in per_seed]}  "
                f"cell mean={cell_mean:+.3f}"
            )
    return matrix


def plot_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    """Render the 5×5 heatmap with annotated cells; save as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(LAMBDA_2_GRID)), [f"{v:g}" for v in LAMBDA_2_GRID])
    ax.set_yticks(range(len(LAMBDA_1_GRID)), [f"{v:g}" for v in LAMBDA_1_GRID])
    ax.set_xlabel(r"$\lambda_2$ (imbalance weight)")
    ax.set_ylabel(r"$\lambda_1$ (overload weight)")
    ax.set_title(
        "λ sensitivity — REINFORCE mean reward "
        f"(last {WINDOW} of {SHORT_EPISODES} eps, {len(SEEDS)} seeds/cell)"
    )
    mean_val = float(matrix.mean())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < mean_val else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="mean episodic reward (3-seed avg)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    n_cells = len(LAMBDA_1_GRID) * len(LAMBDA_2_GRID)
    print(
        f"Running λ sensitivity sweep: "
        f"{len(LAMBDA_1_GRID)}×{len(LAMBDA_2_GRID)}={n_cells} cells × "
        f"{len(SEEDS)} seeds × {SHORT_EPISODES} episodes "
        f"= {n_cells * len(SEEDS)} REINFORCE runs"
    )
    matrix = run_sweep()
    plot_heatmap(matrix, OUT_PNG)
    print(f"wrote heatmap → {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"matrix:\n{matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
