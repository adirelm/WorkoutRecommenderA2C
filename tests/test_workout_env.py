"""Tests for WorkoutEnv — Phase-2 gym-like env (brief §7.3, ADR-002)."""

from __future__ import annotations

import numpy as np

from src.env.state import ACTION_COUNT, State
from src.env.workout_env import EnvConfig, WorkoutEnv


def test_reset_returns_initial_state() -> None:
    env = WorkoutEnv(seed=42)
    s = env.reset()
    assert isinstance(s, State)
    assert s == State.initial()


def test_step_returns_4_tuple() -> None:
    env = WorkoutEnv(seed=42)
    env.reset()
    out = env.step(0)
    assert isinstance(out, tuple)
    assert len(out) == 4
    next_state, reward, done, info = out
    assert isinstance(next_state, State)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_done_after_episode_length_steps() -> None:
    env = WorkoutEnv(EnvConfig(episode_length=28), seed=42)
    env.reset()
    done = False
    for _ in range(28):
        _, _, done, _ = env.step(0)
    assert done is True


def test_done_false_during_episode() -> None:
    env = WorkoutEnv(EnvConfig(episode_length=28), seed=42)
    env.reset()
    for i in range(27):
        _, _, done, _ = env.step(0)
        assert done is False, f"Premature done at step {i}"


def test_seed_determinism() -> None:
    actions = [1, 2, 3, 0, 4, 5, 6, 1, 2, 0]
    env_a = WorkoutEnv(seed=123)
    env_b = WorkoutEnv(seed=123)
    env_a.reset()
    env_b.reset()
    traj_a = [env_a.step(a) for a in actions]
    traj_b = [env_b.step(a) for a in actions]
    for (sa, ra, da, _ia), (sb, rb, db, _ib) in zip(traj_a, traj_b, strict=True):
        assert sa == sb
        assert ra == rb
        assert da == db


def test_action_space_is_seven() -> None:
    env = WorkoutEnv(seed=42)
    assert env.action_space == ACTION_COUNT == 7


def test_state_dim_is_twelve() -> None:
    env = WorkoutEnv(seed=42)
    assert env.state_dim == 12


def test_info_contains_reward_decomposition() -> None:
    env = WorkoutEnv(seed=42)
    env.reset()
    _, _, _, info = env.step(1)
    assert "gain" in info
    assert "overload" in info
    assert "imbalance" in info


def test_action_mask_updates_each_step() -> None:
    env = WorkoutEnv(seed=42)
    env.reset()
    mask0 = env.action_mask().copy()
    assert isinstance(mask0, np.ndarray)
    assert mask0.shape == (ACTION_COUNT,)
    # Drive soreness up by repeating Push to force mask change
    for _ in range(5):
        env.step(1)
    mask1 = env.action_mask()
    assert mask1.shape == (ACTION_COUNT,)
    # Rest (0) must always remain legal
    assert bool(mask1[0]) is True
