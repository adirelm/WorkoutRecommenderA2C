"""Phase 2 integration: WorkoutEnv full episode + reward decomposition + masking."""

import math

import numpy as np
import pytest

from src.env.state import ACTION_COUNT, STATE_DIM
from src.env.workout_env import WorkoutEnv

# Regression fixture: pinned total_reward for seed=42 env + rng=42 masked policy.
# Computed locally once; any drift here means the env or reward shape changed.
_SEED42_TOTAL_REWARD = 6.4212110632159


def test_full_episode_28_days_uniform_policy():
    """Run 28 days of seeded masked policy; assert episode terminates,
    rewards are finite + tightly bounded, state ends in valid ranges,
    reward decomposition keys are present, and the total reward matches
    the pinned regression fixture for (env seed=42, policy rng=42)."""
    env = WorkoutEnv(seed=42)
    s = env.reset()
    policy_rng = np.random.default_rng(42)
    total_reward = 0.0
    episode_length = env.cfg.episode_length
    for t in range(episode_length):
        mask = env.action_mask()
        legal = [a for a in range(ACTION_COUNT) if mask[a]]
        assert legal, f"no legal actions at t={t}"
        action = int(policy_rng.choice(legal))
        s, r, done, info = env.step(action)
        total_reward += r
        assert not math.isnan(r)  # NaN check
        assert abs(r) < 10  # tight realistic bound for our config (max observed ~0.3)
        assert s.to_array().shape == (STATE_DIM,)
        # decomposition keys present
        for k in ("gain", "overload", "imbalance"):
            assert k in info
        if t == episode_length - 1:
            assert done is True
        else:
            assert done is False
    assert isinstance(total_reward, float)
    # final state ranges
    arr = s.to_array()
    assert 0.0 <= arr[0] <= 1.0  # fatigue
    assert 0.0 <= arr[5] <= 1.0  # readiness
    # Regression fixture: pinned total reward for this seed pair.
    assert total_reward == pytest.approx(_SEED42_TOTAL_REWARD, abs=1e-3)


def test_two_envs_same_seed_same_trajectory():
    """Determinism end-to-end."""
    env_a = WorkoutEnv(seed=7)
    env_b = WorkoutEnv(seed=7)
    env_a.reset()
    env_b.reset()
    rewards_a, rewards_b = [], []
    for t in range(env_a.cfg.episode_length):
        mask_a = env_a.action_mask()
        mask_b = env_b.action_mask()
        assert (mask_a == mask_b).all()
        legal = [a for a in range(ACTION_COUNT) if mask_a[a]]
        action = legal[t % len(legal)]
        sa, ra, _, _ = env_a.step(action)
        sb, rb, _, _ = env_b.step(action)
        rewards_a.append(ra)
        rewards_b.append(rb)
        assert (sa.to_array() == sb.to_array()).all()
    assert rewards_a == rewards_b
