"""Env invariants — properties that must hold across many random rollouts."""

from __future__ import annotations

import numpy as np
import pytest

from src.env.state import ACTION_COUNT, STATE_CHANNEL_NAMES

try:
    from src.env.workout_env import WorkoutEnv
except ImportError:  # pragma: no cover - env not yet implemented
    WorkoutEnv = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(WorkoutEnv is None, reason="WorkoutEnv not yet implemented (ADR-004).")

EPISODE_LEN = 28  # brief §7.4 — 28-day episode


def _declared_bounds() -> dict[str, tuple[float, float]]:
    """Per-channel declared min/max from ADR-002 (State docstring)."""
    return {
        "fatigue": (0.0, 1.0),
        "soreness_push": (0.0, 1.0),
        "soreness_pull": (0.0, 1.0),
        "soreness_legs": (0.0, 1.0),
        "soreness_core": (0.0, 1.0),
        "readiness": (0.0, 1.0),
        "rolling_7d_volume": (0.0, np.inf),
        "streak_days_trained": (0.0, np.inf),
        "days_since_last_rest": (0.0, np.inf),
        "muscle_balance_push_vs_pull": (-1.0, 1.0),
        "adherence_signal": (-1.0, 1.0),
        "weekly_progress": (0.0, 1.2),
    }


def test_state_stays_in_bounds_over_1000_steps() -> None:
    """Across 1000 mixed random steps, no state channel goes out of its declared range."""
    env = WorkoutEnv(seed=0)
    state = env.reset()
    rng = np.random.default_rng(0)
    bounds = _declared_bounds()
    for _ in range(1000):
        action = int(rng.integers(0, ACTION_COUNT))
        state, _reward, done, _info = env.step(action)
        arr = state.to_array() if hasattr(state, "to_array") else np.asarray(state)
        for i, name in enumerate(STATE_CHANNEL_NAMES):
            lo, hi = bounds[name]
            assert lo - 1e-6 <= arr[i] <= hi + 1e-6, f"{name}={arr[i]} out of [{lo},{hi}]"
        if done:
            state = env.reset()


def test_reward_is_finite_and_bounded() -> None:
    """No NaN, no inf, magnitude reasonable for ~100 random episodes."""
    env = WorkoutEnv(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        env.reset()
        for _ in range(EPISODE_LEN):
            action = int(rng.integers(0, ACTION_COUNT))
            _state, reward, done, _info = env.step(action)
            assert np.isfinite(reward), "reward must be finite"
            assert abs(reward) < 1e3, f"reward magnitude {reward} unreasonable"
            if done:
                break


def test_seed_reproduces_byte_exact_rollouts() -> None:
    """Two envs with same seed produce identical state arrays + rewards step-by-step."""
    env_a, env_b = WorkoutEnv(seed=42), WorkoutEnv(seed=42)
    s_a, s_b = env_a.reset(), env_b.reset()
    np.testing.assert_array_equal(s_a.to_array(), s_b.to_array())
    actions = np.random.default_rng(42).integers(0, ACTION_COUNT, size=EPISODE_LEN)
    for a in actions:
        s_a, r_a, d_a, _ = env_a.step(int(a))
        s_b, r_b, d_b, _ = env_b.step(int(a))
        np.testing.assert_array_equal(s_a.to_array(), s_b.to_array())
        assert r_a == r_b and d_a == d_b


def test_mask_never_zeros_out_every_action() -> None:
    """At least one action is always legal — Mobility per ADR-004."""
    env = WorkoutEnv(seed=2)
    env.reset()
    rng = np.random.default_rng(2)
    for _ in range(EPISODE_LEN * 10):
        mask = env.action_mask()
        assert np.any(mask), "action mask must always allow at least one action (Mobility)"
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal))
        _s, _r, done, _i = env.step(action)
        if done:
            env.reset()


def test_episode_terminates_at_length_28() -> None:
    """done flips True exactly at step 28 (1-indexed) / step 27 (0-indexed)."""
    env = WorkoutEnv(seed=3)
    env.reset()
    for step_idx in range(EPISODE_LEN):
        _s, _r, done, _i = env.step(0)  # Rest, always legal
        if step_idx < EPISODE_LEN - 1:
            assert not done, f"premature termination at 0-indexed step {step_idx}"
        else:
            assert done, f"failed to terminate at 0-indexed step {step_idx} (= step 28)"
