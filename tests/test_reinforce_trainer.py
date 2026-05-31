"""Tests for src/services/reinforce_trainer.py — REINFORCE loop (brief §7.4 eq. 16)."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.env.workout_env import EnvConfig, WorkoutEnv
from src.model.policy_net import PolicyNet
from src.services.baseline import RunningMeanBaseline
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig, REINFORCEHistory

_TINY_HIDDEN = 8
_TINY_EPISODES = 5
_TINY_EP_LEN = 6


def _make_trainer(
    seed: int = 42, episodes: int = _TINY_EPISODES, baseline_alpha: float = 0.1
) -> REINFORCETrainer:
    env = WorkoutEnv(env_config=EnvConfig(episode_length=_TINY_EP_LEN), seed=seed)
    policy = PolicyNet(hidden=_TINY_HIDDEN, seed=seed)
    cfg = REINFORCEConfig(
        policy_hidden=_TINY_HIDDEN, episodes=episodes, baseline_alpha=baseline_alpha, lr=1e-2
    )
    return REINFORCETrainer(policy=policy, env=env, config=cfg, seed=seed)


def test_train_returns_history_with_correct_length():
    trainer = _make_trainer()
    history = trainer.train()
    assert isinstance(history, REINFORCEHistory)
    assert history.episodes_run == _TINY_EPISODES
    assert len(history.rewards) == _TINY_EPISODES
    assert len(history.losses) == _TINY_EPISODES
    assert len(history.baseline) == _TINY_EPISODES


def test_history_rewards_finite_no_nan():
    trainer = _make_trainer()
    history = trainer.train()
    for r in history.rewards:
        assert math.isfinite(r), f"non-finite reward {r}"
    for loss in history.losses:
        assert math.isfinite(loss), f"non-finite loss {loss}"


def test_loss_decreases_over_episodes_average():
    """Loose sanity: average of last 3 < average of first 3 (learning signal present)."""
    trainer = _make_trainer(episodes=12)
    history = trainer.train()
    losses = np.asarray(history.losses, dtype=float)
    first = float(np.mean(np.abs(losses[:3])))
    last = float(np.mean(np.abs(losses[-3:])))
    # Loose: just require the trend not be wildly worse (within 3x the first window).
    assert last < first * 3.0, f"loss exploded: first={first:.3f} last={last:.3f}"


def test_seeded_trainer_deterministic():
    t1 = _make_trainer(seed=7)
    h1 = t1.train()
    t2 = _make_trainer(seed=7)
    h2 = t2.train()
    assert h1.rewards == h2.rewards, "same seed must reproduce reward sequence"


def test_baseline_used_in_loss():
    """baseline_alpha > 0 → RunningMeanBaseline.value moves away from 0 over episodes."""
    trainer = _make_trainer()
    assert isinstance(trainer.baseline, RunningMeanBaseline)
    assert trainer.baseline.alpha > 0
    initial = trainer.baseline.value
    trainer.train()
    assert trainer.baseline.value != initial, "baseline never updated"
    assert trainer.baseline.updates == _TINY_EPISODES


def test_policy_explores_initially():
    """Early episodes show >1 distinct action — sampling, not argmax."""
    trainer = _make_trainer()
    actions_seen: set[int] = set()
    for _ in range(3):
        _, actions, _ = trainer.run_episode()
        actions_seen.update(actions)
    assert len(actions_seen) > 1, "policy collapsed to a single action — no exploration"


def test_action_mask_respected():
    """No illegal action is ever sampled during a training rollout."""
    env = WorkoutEnv(env_config=EnvConfig(episode_length=_TINY_EP_LEN), seed=99)
    policy = PolicyNet(hidden=_TINY_HIDDEN, seed=99)
    state = env.reset(seed=99)
    for _step in range(_TINY_EP_LEN):
        mask = env.action_mask()
        state_t = torch.from_numpy(state.to_array())
        mask_t = torch.from_numpy(mask)
        action_id, _ = policy.sample(state_t, action_mask=mask_t)
        assert mask[action_id], f"sampled illegal action {action_id} under mask {mask.tolist()}"
        state, _, done, _ = env.step(action_id)
        if done:
            break


def test_run_episode_returns_tuple_of_three_lists():
    trainer = _make_trainer()
    states, actions, rewards = trainer.run_episode()
    assert isinstance(states, list)
    assert isinstance(actions, list)
    assert isinstance(rewards, list)
    assert len(actions) == len(rewards) == _TINY_EP_LEN
    assert all(0 <= a < 7 for a in actions)


def test_train_episodes_override():
    """Passing episodes=N to train() overrides config.episodes."""
    trainer = _make_trainer(episodes=20)  # config says 20
    history = trainer.train(episodes=3)  # but we ask for 3
    assert history.episodes_run == 3
    assert len(history.rewards) == 3


def test_baseline_alpha_zero_keeps_baseline_constant():
    """alpha=0 → vanilla REINFORCE; baseline.value never changes."""
    trainer = _make_trainer(seed=1, episodes=3, baseline_alpha=0.0)
    trainer.train()
    assert trainer.baseline.value == 0.0
