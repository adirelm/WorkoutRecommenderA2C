"""10-seed REINFORCE vs A2C comparison — Phase 10 results-optimization pass.

The prior results/figures/comparison.png used 3 seeds — directional only.
This script re-runs §7.6 at 10 seeds × 50 episodes per algo so the mean
± 1σ bands have enough samples for a Welch t-test on end-of-training
reward. Outputs (overwrite the prior 3-seed artifact):

    results/figures/comparison.png   — overlayed mean ± 1σ bands
    results/comparison_seeded.json   — raw per-(algo, seed, episode) rewards

Trim policy: if 10 × 50 exceeds the ~10 min wall-clock budget, drop to
N_SEEDS=7 / N_EPISODES=30 below. Dry run measured ~2 min, no trim needed.

Run: uv run --active python scripts/run_seeded_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.workout_env import WorkoutEnv  # noqa: E402
from src.model.actor_critic import ActorCriticNet  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.a2c_trainer import A2CTrainer  # noqa: E402
from src.services.a2c_types import A2CConfig  # noqa: E402
from src.services.reinforce_trainer import REINFORCETrainer  # noqa: E402
from src.services.types import REINFORCEConfig  # noqa: E402

N_SEEDS: int = 10
N_EPISODES: int = 50
BASE_SEED: int = 42

OUT_PNG: Path = REPO_ROOT / "results" / "figures" / "comparison.png"
OUT_JSON: Path = REPO_ROOT / "results" / "comparison_seeded.json"

REINFORCE_COLOR = "#1f77b4"  # blue
A2C_COLOR = "#ff7f0e"  # orange


def train_reinforce_seed(seed: int) -> np.ndarray:
    """Train REINFORCE for N_EPISODES at the given seed; return per-episode rewards."""
    env = WorkoutEnv(seed=seed)
    policy = PolicyNet(seed=seed)
    cfg = REINFORCEConfig(episodes=N_EPISODES)
    hist = REINFORCETrainer(policy, env, cfg, seed=seed).train(episodes=N_EPISODES)
    return np.asarray(hist.rewards, dtype=np.float32)


def train_a2c_seed(seed: int) -> np.ndarray:
    """Train A2C for N_EPISODES at the given seed; return per-episode rewards."""
    env = WorkoutEnv(seed=seed)
    ac = ActorCriticNet(seed=seed)
    cfg = A2CConfig(episodes=N_EPISODES)
    hist = A2CTrainer(ac, env, cfg, seed=seed).train(episodes=N_EPISODES)
    return np.asarray(hist.rewards, dtype=np.float32)


def run_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Train both algos × N_SEEDS × N_EPISODES; return (reinforce, a2c) stacks."""
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    r_stack = np.zeros((N_SEEDS, N_EPISODES), dtype=np.float32)
    a_stack = np.zeros((N_SEEDS, N_EPISODES), dtype=np.float32)
    for i, seed in enumerate(seeds):
        t0 = time.perf_counter()
        r_stack[i] = train_reinforce_seed(seed)
        t_r = time.perf_counter() - t0
        t0 = time.perf_counter()
        a_stack[i] = train_a2c_seed(seed)
        t_a = time.perf_counter() - t0
        print(
            f"  seed={seed:>4d}  reinforce_tail={r_stack[i, -5:].mean():+.2f}  "
            f"a2c_tail={a_stack[i, -5:].mean():+.2f}  [{t_r + t_a:.1f}s]"
        )
    return r_stack, a_stack


def plot_bands(r_stack: np.ndarray, a_stack: np.ndarray, out_path: Path) -> None:
    """Render overlayed mean ± 1σ bands for REINFORCE (blue) and A2C (orange)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, N_EPISODES + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    for stack, label, color in (
        (r_stack, "REINFORCE", REINFORCE_COLOR),
        (a_stack, "A2C", A2C_COLOR),
    ):
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.0, label=f"{label} mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, label=f"{label} ±1σ")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title(f"REINFORCE vs A2C — {N_SEEDS} seeds × {N_EPISODES} episodes (mean ± 1σ)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def dump_json(r_stack: np.ndarray, a_stack: np.ndarray, out_path: Path) -> None:
    """Persist raw per-(seed, episode) reward arrays for reproducibility."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seeds": N_SEEDS,
        "n_episodes": N_EPISODES,
        "base_seed": BASE_SEED,
        "seeds": [BASE_SEED + i for i in range(N_SEEDS)],
        "reinforce_rewards": r_stack.tolist(),
        "a2c_rewards": a_stack.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def report_significance(r_stack: np.ndarray, a_stack: np.ndarray) -> None:
    """Welch t-test on end-of-training (last-5-episode mean per seed)."""
    r_tail = r_stack[:, -5:].mean(axis=1)
    a_tail = a_stack[:, -5:].mean(axis=1)
    t, p = stats.ttest_ind(a_tail, r_tail, equal_var=False)
    r_band = (r_tail.mean() - r_tail.std(), r_tail.mean() + r_tail.std())
    a_band = (a_tail.mean() - a_tail.std(), a_tail.mean() + a_tail.std())
    overlap = not (a_band[0] > r_band[1] or r_band[0] > a_band[1])
    print("\nFinal-5-episode reward (per-seed mean):")
    print(f"  REINFORCE  mean={r_tail.mean():+.3f}  std={r_tail.std():.3f}  band={r_band}")
    print(f"  A2C        mean={a_tail.mean():+.3f}  std={a_tail.std():.3f}  band={a_band}")
    print(f"Welch t-test (A2C vs REINFORCE):  t={t:+.3f}  p={p:.4f}")
    print(f"±1σ bands overlap at tail: {overlap}")
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"Verdict: difference is {verdict} at α=0.05.")


def main() -> int:
    print(f"Seeded comparison: {N_SEEDS} seeds × {N_EPISODES} episodes × 2 algos")
    t0 = time.perf_counter()
    r_stack, a_stack = run_sweep()
    elapsed = time.perf_counter() - t0
    print(f"\nsweep complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    plot_bands(r_stack, a_stack, OUT_PNG)
    dump_json(r_stack, a_stack, OUT_JSON)
    print(f"wrote chart → {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"wrote json  → {OUT_JSON.relative_to(REPO_ROOT)}")
    report_significance(r_stack, a_stack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
