"""Tests for A2CTrainer — short, seeded, deterministic runs."""

from __future__ import annotations

import math

from src.env.workout_env import WorkoutEnv
from src.model.actor_critic import ActorCriticNet
from src.services.a2c_trainer import A2CTrainer
from src.services.a2c_types import A2CConfig


def _tiny_trainer(seed: int = 42) -> A2CTrainer:
    env = WorkoutEnv(seed=seed)
    ac = ActorCriticNet(actor_hidden=8, critic_hidden=8, seed=seed)
    config = A2CConfig(actor_hidden=8, critic_hidden=8, actor_lr=1e-3, critic_lr=1e-3, episodes=4)
    return A2CTrainer(ac, env, config, seed=seed)


def test_train_returns_history_with_correct_length():
    trainer = _tiny_trainer()
    history = trainer.train()
    assert history.episodes_run == 4
    assert len(history.rewards) == 4
    assert len(history.actor_losses) == 4
    assert len(history.critic_losses) == 4
    assert len(history.advantages_mean) == 4


def test_history_actor_and_critic_losses_finite():
    trainer = _tiny_trainer()
    history = trainer.train()
    assert all(math.isfinite(loss) for loss in history.actor_losses)
    assert all(math.isfinite(loss) for loss in history.critic_losses)


def test_seeded_trainer_deterministic():
    h1 = _tiny_trainer(seed=42).train()
    h2 = _tiny_trainer(seed=42).train()
    assert h1.rewards == h2.rewards


def test_separate_actor_critic_optimizers_exist():
    trainer = _tiny_trainer()
    assert trainer.actor_optim is not trainer.critic_optim
    assert trainer.actor_optim.param_groups[0]["lr"] == 1e-3
    assert trainer.critic_optim.param_groups[0]["lr"] == 1e-3


def test_advantages_mean_recorded():
    trainer = _tiny_trainer()
    history = trainer.train()
    assert all(math.isfinite(a) for a in history.advantages_mean)


def test_train_handles_zero_episodes():
    """Passing episodes=0 still returns a valid history."""
    trainer = _tiny_trainer()
    history = trainer.train(episodes=0)
    assert history.episodes_run == 0
    assert history.rewards == ()


def test_episodes_explore_different_trajectories():
    """Regression guard for the per-episode seed-reuse bug (v1p5 P0).

    If run_episode() re-seeded the env with the same seed every episode, every
    rollout would produce an identical (state, action, reward) trajectory and
    A2C would never learn from variance. Assert that two consecutive episodes
    in the SAME training run produce different reward totals (extremely unlikely
    to coincide by chance under any non-trivial stochastic policy + transition)."""
    trainer = _tiny_trainer(seed=42)
    h = trainer.train(episodes=5)
    # All 5 episode rewards being IDENTICAL would mean the env never advanced —
    # the bug. Allow 2 to coincide by sheer luck, but not all 5.
    unique_rewards = set(h.rewards)
    assert len(unique_rewards) >= 2, (
        f"All episodes produced identical reward {h.rewards} — per-episode seed reuse bug regressed"
    )
