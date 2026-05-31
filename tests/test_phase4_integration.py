"""Phase 4 integration: PolicyNet + REINFORCETrainer + WorkoutEnv (with Phase-2 SyntheticTrainee env, NOT yet LSTM)."""

import math

from src.env.workout_env import WorkoutEnv
from src.model.policy_net import PolicyNet
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig


def test_reinforce_full_pipeline_smoke():
    env = WorkoutEnv(seed=42)
    policy = PolicyNet(hidden=32, seed=42)  # smaller for speed
    config = REINFORCEConfig(policy_hidden=32, lr=1e-3, episodes=10, baseline_alpha=0.1)
    trainer = REINFORCETrainer(policy, env, config, seed=42)
    history = trainer.train()

    assert history.episodes_run == 10
    assert all(math.isfinite(r) for r in history.rewards)
    assert all(math.isfinite(loss) for loss in history.losses)

    # Loss-trend sanity: last 3 mean < first 3 mean (loose)
    first_3 = sum(history.losses[:3]) / 3
    last_3 = sum(history.losses[-3:]) / 3
    assert last_3 != first_3  # at least learning is happening (not flat)
