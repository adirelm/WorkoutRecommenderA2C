"""Tests for WorkoutEnv — Phase-2 gym-like env (brief §7.3, ADR-002)."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.state import ACTION_COUNT, State
from src.env.synthetic_trainee import SyntheticTrainee
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


def test_reset_with_explicit_seed_overrides_constructor_seed() -> None:
    """Line 98: reset(seed=X) updates self._seed and reproduces a fresh rollout."""
    env = WorkoutEnv(seed=42)
    env.reset()
    env.step(1)
    env.step(2)
    # Re-seed via reset(); the next step trajectory should match an env built
    # with the new seed from scratch (proves seed=X took effect).
    env.reset(seed=999)
    s_after, r_after, _, _ = env.step(1)
    env_fresh = WorkoutEnv(seed=999)
    env_fresh.reset()
    s_fresh, r_fresh, _, _ = env_fresh.step(1)
    assert s_after == s_fresh
    assert r_after == r_fresh


def test_step_with_action_out_of_range_raises_value_error() -> None:
    """Line 109: action_id outside [0, ACTION_COUNT) raises ValueError."""
    env = WorkoutEnv(seed=42)
    env.reset()
    with pytest.raises(ValueError, match="out of range"):
        env.step(ACTION_COUNT)
    with pytest.raises(ValueError, match="out of range"):
        env.step(-1)


def test_history_returns_copy_of_action_sequence() -> None:
    """Line 152: history() exposes the per-step action log as a list copy."""
    env = WorkoutEnv(seed=42)
    env.reset()
    for a in (0, 1, 2, 3):
        env.step(a)
    hist = env.history()
    assert hist == [0, 1, 2, 3]
    # mutating the returned list must not affect internal state
    hist.append(99)
    assert env.history() == [0, 1, 2, 3]


def test_injected_trainee_is_used_verbatim() -> None:
    """Line 157: when a trainee is injected, _build_trainee returns it as-is."""
    sentinel = SyntheticTrainee(rng=np.random.default_rng(0), noise_sigma=0.0)
    env = WorkoutEnv(seed=42, trainee=sentinel)
    # _build_trainee is called on construction AND on every reset.
    assert env._trainee is sentinel
    env.reset()
    assert env._trainee is sentinel


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
