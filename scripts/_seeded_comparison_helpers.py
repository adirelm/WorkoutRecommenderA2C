"""Helpers for run_seeded_comparison.py — kept separate so script stays ≤150 LOC."""

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

from src.model.actor_critic import ActorCriticNet  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.model.world_model_builder import build_lstm_env, train_program_world_model  # noqa: E402
from src.services.a2c_trainer import A2CTrainer  # noqa: E402
from src.services.a2c_types import A2CConfig  # noqa: E402
from src.services.reinforce_trainer import REINFORCETrainer  # noqa: E402
from src.services.types import REINFORCEConfig  # noqa: E402

REINFORCE_COLOR = "#1f77b4"  # blue
A2C_COLOR = "#ff7f0e"  # orange
RANDOM_COLOR = "#7f7f7f"  # grey — uniform-random floor


def eval_random_seed(seed: int, n_episodes: int, model) -> np.ndarray:
    """Roll a uniform-random masked policy over the LSTM env (no training); per-episode rewards.

    Anchors the absolute reward scale: any learned policy should beat this floor.
    """
    env = build_lstm_env(seed=seed, model=model)
    rng = np.random.default_rng(seed)
    rewards = np.zeros(n_episodes, dtype=np.float32)
    for ep in range(n_episodes):
        env.reset()
        done, total = False, 0.0
        while not done:
            _s, r, done, _i = env.step(int(rng.choice(np.flatnonzero(env.action_mask()))))
            total += float(r)
        rewards[ep] = total
    return rewards


def train_reinforce_seed(seed: int, n_episodes: int, model) -> np.ndarray:
    """Train REINFORCE over the frozen LSTM env (real PHUL); return per-episode rewards."""
    env = build_lstm_env(seed=seed, model=model)
    policy = PolicyNet(seed=seed)
    cfg = REINFORCEConfig(episodes=n_episodes)
    hist = REINFORCETrainer(policy, env, cfg, seed=seed).train(episodes=n_episodes)
    return np.asarray(hist.rewards, dtype=np.float32)


def train_a2c_seed(seed: int, n_episodes: int, model) -> np.ndarray:
    """Train A2C over the frozen LSTM env (real PHUL); return per-episode rewards."""
    env = build_lstm_env(seed=seed, model=model)
    ac = ActorCriticNet(seed=seed)
    cfg = A2CConfig(episodes=n_episodes)
    hist = A2CTrainer(ac, env, cfg, seed=seed).train(episodes=n_episodes)
    return np.asarray(hist.rewards, dtype=np.float32)


def run_sweep(n_seeds: int, n_episodes: int, base_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """REINFORCE + A2C + random over n_seeds × n_episodes on the frozen LSTM env; return stacks."""
    seeds = [base_seed + i for i in range(n_seeds)]
    r_stack = np.zeros((n_seeds, n_episodes), dtype=np.float32)
    a_stack = np.zeros((n_seeds, n_episodes), dtype=np.float32)
    rnd_stack = np.zeros((n_seeds, n_episodes), dtype=np.float32)
    model = train_program_world_model(seed=base_seed)  # fixed env; only RL seed varies
    for i, seed in enumerate(seeds):
        t0 = time.perf_counter()
        r_stack[i] = train_reinforce_seed(seed, n_episodes, model)
        a_stack[i] = train_a2c_seed(seed, n_episodes, model)
        rnd_stack[i] = eval_random_seed(seed, n_episodes, model)
        print(
            f"  seed={seed:>4d}  reinforce_tail={r_stack[i, -5:].mean():+.2f}  "
            f"a2c_tail={a_stack[i, -5:].mean():+.2f}  random_tail={rnd_stack[i, -5:].mean():+.2f}  "
            f"[{time.perf_counter() - t0:.1f}s]"
        )
    return r_stack, a_stack, rnd_stack


def plot_bands(
    r_stack: np.ndarray,
    a_stack: np.ndarray,
    rnd_stack: np.ndarray,
    out_path: Path,
    n_seeds: int,
    n_episodes: int,
) -> None:
    """Mean ± 1σ bands: REINFORCE (blue), A2C (orange), uniform-random floor (grey)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, n_episodes + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    for stack, label, color in (
        (r_stack, "REINFORCE", REINFORCE_COLOR),
        (a_stack, "A2C", A2C_COLOR),
        (rnd_stack, "Random", RANDOM_COLOR),
    ):
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.0, label=f"{label} mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, label=f"{label} ±1σ")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title(f"REINFORCE vs A2C — {n_seeds} seeds × {n_episodes} episodes (mean ± 1σ)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def dump_json(
    r_stack: np.ndarray,
    a_stack: np.ndarray,
    rnd_stack: np.ndarray,
    out_path: Path,
    n_seeds: int,
    n_episodes: int,
    base_seed: int,
) -> None:
    """Persist raw per-(seed, episode) reward arrays for reproducibility."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "base_seed": base_seed,
        "seeds": [base_seed + i for i in range(n_seeds)],
        "reinforce_rewards": r_stack.tolist(),
        "a2c_rewards": a_stack.tolist(),
        "random_rewards": rnd_stack.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tail(stack: np.ndarray) -> np.ndarray:
    """Per-seed mean of the last-5 episodes."""
    return stack[:, -5:].mean(axis=1)


def _vs_random(rl: np.ndarray, rnd: np.ndarray) -> str:
    """Directional verdict of an RL policy against the masked-random floor (Welch t)."""
    _, p = stats.ttest_ind(rl, rnd, equal_var=False)
    if p >= 0.05:
        return f"p={p:.4f} -> NOT distinguishable from random"
    return f"p={p:.4f} -> {'BEATS' if rl.mean() > rnd.mean() else 'BELOW'} random (significant)"


def report_significance(r_stack: np.ndarray, a_stack: np.ndarray, rnd_stack: np.ndarray) -> None:
    """Welch t-tests on last-5-episode per-seed means: A2C vs REINFORCE, and each RL vs random."""
    r, a, rnd = _tail(r_stack), _tail(a_stack), _tail(rnd_stack)
    t_ar, p_ar = stats.ttest_ind(a, r, equal_var=False)
    print("\nFinal-5-episode reward (per-seed mean ± std):")
    print(f"  REINFORCE  {r.mean():+.3f} ± {r.std():.3f}")
    print(f"  A2C        {a.mean():+.3f} ± {a.std():.3f}")
    print(f"  Random     {rnd.mean():+.3f} ± {rnd.std():.3f}")
    print(f"A2C vs REINFORCE:  t={t_ar:+.3f}  p={p_ar:.4f}  -> {'sig' if p_ar < 0.05 else 'NOT sig'}")
    print(f"REINFORCE vs Random:  {_vs_random(r, rnd)}")
    print(f"A2C vs Random:  {_vs_random(a, rnd)}")
