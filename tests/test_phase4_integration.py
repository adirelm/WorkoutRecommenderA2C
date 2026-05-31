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
    # Real learning assertion (P0 v4p4): reward trend must rise OR loss must decrease across episodes.
    first_3_reward = sum(history.rewards[:3]) / 3
    last_3_reward = sum(history.rewards[-3:]) / 3
    # Either rewards improve OR losses shrink — REINFORCE is high-variance, so we OR the conditions.
    reward_improved = last_3_reward > first_3_reward
    loss_decreased = last_3 < first_3
    assert reward_improved or loss_decreased, (
        f"neither reward trend ({first_3_reward:.3f} -> {last_3_reward:.3f}) "
        f"nor loss trend ({first_3:.3f} -> {last_3:.3f}) shows learning"
    )
    # Determinism: a second run with the same seed must produce identical rewards.
    env2 = WorkoutEnv(seed=42)
    policy2 = PolicyNet(hidden=32, seed=42)
    trainer2 = REINFORCETrainer(policy2, env2, config, seed=42)
    history2 = trainer2.train()
    assert history.rewards == history2.rewards, "same seed must produce identical reward trajectory"
