"""REINFORCE training loop (brief §7.4 eq. 16).

One Monte-Carlo policy-gradient update per episode:

    1. roll out one episode under π_θ  →  (states, actions, rewards, log_probs)
    2. compute discounted returns G_t (compute_returns)
    3. update the running-mean baseline b with the episodic total
    4. loss = - mean_t [ log π_θ(a_t | s_t) · (G_t - b) ]   (reinforce_loss)
    5. optimizer step (Adam, with optional grad-norm clip)

REINFORCE-specific update logic lives here; shared rollout / sampling /
seeding belongs to :class:`src.services.base_trainer.BaseTrainer` (V3 §4
no-duplication, §12 open-closed extension point).
"""

from __future__ import annotations

import torch
from torch import nn

from src.env.workout_env import WorkoutEnv
from src.model.policy_net import PolicyNet
from src.services.base_trainer import BaseTrainer
from src.services.baseline import RunningMeanBaseline
from src.services.reinforce_helpers import compute_returns, reinforce_loss
from src.services.types import REINFORCEConfig, REINFORCEHistory


class REINFORCETrainer(BaseTrainer):
    """Monte-Carlo REINFORCE with running-mean baseline.

    Input:
      state_t (torch.Tensor) — (STATE_DIM,) per-step state into forward_step().
      mask_t (torch.Tensor) — bool (ACTION_COUNT,) legal-action mask.
      episodes (int | None) — override config.episodes for one train() call.

    Output:
      REINFORCEHistory — train() returns episodes_run + per-episode rewards,
      losses, baseline trace, and seed (immutable tuples).

    Setup:
      policy (PolicyNet) — actor network whose params Adam updates.
      env (WorkoutEnv) — episodic environment to roll out against.
      config (REINFORCEConfig | None) — lr, gamma, episodes, baseline_alpha,
        entropy_coef, grad_clip_norm (defaults to REINFORCEConfig()).
      seed (int) — passed to BaseTrainer for global RNG seeding.
    """

    def __init__(
        self,
        policy: PolicyNet,
        env: WorkoutEnv,
        config: REINFORCEConfig | None = None,
        seed: int = 42,
    ) -> None:
        cfg = config if config is not None else REINFORCEConfig()
        super().__init__(env=env, config=cfg, seed=seed)
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.baseline = RunningMeanBaseline(alpha=cfg.baseline_alpha)

    # ----------------------------------------------------------- registry hook
    @classmethod
    def build(cls, env: WorkoutEnv, seed: int, episodes: int) -> REINFORCETrainer:
        """SDK-facing factory (V3 §12): wires PolicyNet + config from config.yaml [reinforce]."""
        cfg = REINFORCEConfig.from_yaml(episodes=int(episodes))
        policy = PolicyNet(hidden=cfg.policy_hidden, seed=seed)
        return cls(policy=policy, env=env, config=cfg, seed=seed)

    @property
    def net(self) -> PolicyNet:
        """Uniform accessor used by SDK to cache the trained network."""
        return self.policy

    # ----------------------------------------------------------- forward hook
    def forward_step(
        self,
        state_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """REINFORCE forward: policy logits → masked Categorical → sample."""
        logits = self.policy.forward(state_t)
        action_id, log_prob, entropy, _dist = self.sample_action(logits, mask_t)
        return action_id, log_prob, entropy, None

    # ----------------------------------------------------------- legacy rollout
    def run_episode(self) -> tuple[list, list, list]:
        """Compat shim used by tests; returns (states, actions, rewards)."""
        states, actions, rewards, _ = self._rollout_with_log_probs()
        return states, actions, rewards

    def _rollout_with_log_probs(
        self,
    ) -> tuple[list, list[int], list[float], list[torch.Tensor]]:
        records = self.rollout_episode()
        # Reconstruct the legacy states list: pre-step states + the post-loop final state.
        states: list = [r.state_t for r in records]
        actions = [r.action_id for r in records]
        rewards = [r.reward for r in records]
        log_probs = [r.log_prob for r in records]
        return states, actions, rewards, log_probs

    # -------------------------------------------------------------------- train
    def train(self, episodes: int | None = None) -> REINFORCEHistory:
        """Episode loop with Adam + grad-clip; returns rewards/losses/baseline trace. Brief §7.4."""
        n_episodes = int(episodes) if episodes is not None else int(self.config.episodes)
        ep_rewards: list[float] = []
        ep_losses: list[float] = []
        ep_baseline: list[float] = []
        for _ in range(n_episodes):
            _states, _actions, rewards, log_probs = self._rollout_with_log_probs()
            total_return = float(sum(rewards))
            returns = compute_returns(rewards, self.config.gamma)
            self.baseline.update(total_return)
            loss = reinforce_loss(log_probs, returns, baseline=self.baseline.value)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            ep_rewards.append(total_return)
            ep_losses.append(float(loss.item()))
            ep_baseline.append(float(self.baseline.value))
        return REINFORCEHistory(
            episodes_run=n_episodes,
            rewards=tuple(ep_rewards),
            losses=tuple(ep_losses),
            baseline=tuple(ep_baseline),
            seed=self.seed,
        )
