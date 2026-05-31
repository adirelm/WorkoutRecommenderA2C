"""Phase 5 integration: A2CTrainer end-to-end + REINFORCE-vs-A2C comparison."""

import math

from src.model.actor_critic import ActorCriticNet
from src.services.a2c_trainer import A2CTrainer
from src.services.comparator import compare

from src.env.workout_env import WorkoutEnv
from src.model.policy_net import PolicyNet
from src.services.a2c_types import A2CConfig
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig


def test_a2c_full_pipeline_smoke():
    env = WorkoutEnv(seed=42)
    ac = ActorCriticNet(actor_hidden=16, critic_hidden=16, seed=42)
    config = A2CConfig(actor_hidden=16, critic_hidden=16, actor_lr=1e-3, critic_lr=1e-3, episodes=8)
    trainer = A2CTrainer(ac, env, config, seed=42)
    history = trainer.train()
    assert history.episodes_run == 8
    assert all(math.isfinite(r) for r in history.rewards)
    assert all(math.isfinite(loss) for loss in history.actor_losses)
    assert all(math.isfinite(loss) for loss in history.critic_losses)


def test_compare_reinforce_vs_a2c_smoke():
    # 2 seeds, 5 episodes each for both algos
    reinforce_histories = []
    a2c_histories = []
    for seed in [42, 43]:
        env_r = WorkoutEnv(seed=seed)
        policy = PolicyNet(hidden=16, seed=seed)
        rc = REINFORCEConfig(policy_hidden=16, episodes=5)
        r_hist = REINFORCETrainer(policy, env_r, rc, seed=seed).train()
        reinforce_histories.append(r_hist)

        env_a = WorkoutEnv(seed=seed)
        ac = ActorCriticNet(actor_hidden=16, critic_hidden=16, seed=seed)
        ac_config = A2CConfig(actor_hidden=16, critic_hidden=16, episodes=5)
        a_hist = A2CTrainer(ac, env_a, ac_config, seed=seed).train()
        a2c_histories.append(a_hist)

    result = compare(reinforce_histories, a2c_histories)
    assert result.episode_count == 5
    assert result.seed_count == 2
    assert result.reinforce_mean_reward.shape == (5,)
    assert result.a2c_mean_reward.shape == (5,)
