"""REINFORCE training loop (brief §7.4 eq. 16).

One Monte-Carlo policy-gradient update per episode:

    1. roll out one episode under π_θ  →  (states, actions, rewards, log_probs)
    2. compute discounted returns G_t (compute_returns)
    3. update the running-mean baseline b with the episodic total
    4. loss = - mean_t [ log π_θ(a_t | s_t) · (G_t - b) ]   (reinforce_loss)
    5. optimizer step (Adam, with optional grad-norm clip)

The trainer owns the env, policy, optimiser, and baseline; rollouts use the
action mask exposed by the env so illegal actions get -inf logits before
sampling. Per CLAUDE.md size cap (150 LOC) the loop is delegated to small
helper modules: :mod:`src.services.baseline` and
:mod:`src.services.reinforce_helpers`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.env.workout_env import WorkoutEnv
from src.model.policy_net import PolicyNet
from src.services.baseline import RunningMeanBaseline
from src.services.reinforce_helpers import compute_returns, reinforce_loss
from src.services.types import REINFORCEConfig, REINFORCEHistory
from src.utils.seeding import set_global_seed


class REINFORCETrainer:
    """Monte-Carlo REINFORCE with running-mean baseline."""

    def __init__(
        self,
        policy: PolicyNet,
        env: WorkoutEnv,
        config: REINFORCEConfig | None = None,
        seed: int = 42,
    ) -> None:
        set_global_seed(int(seed))
        self.policy = policy
        self.env = env
        self.config = config if config is not None else REINFORCEConfig()
        self.seed = int(seed)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.baseline = RunningMeanBaseline(alpha=self.config.baseline_alpha)

    # ------------------------------------------------------------------ episode
    def run_episode(self) -> tuple[list, list, list]:
        """One episode rollout under the *current* policy (no gradient step).

        Returns (states, actions, rewards) — used by tests and by the
        per-update path (which needs log_probs and so calls the internal
        :meth:`_rollout_with_log_probs` instead).
        """
        states, actions, rewards, _ = self._rollout_with_log_probs()
        return states, actions, rewards

    def _rollout_with_log_probs(
        self,
    ) -> tuple[list, list[int], list[float], list[torch.Tensor]]:
        state = self.env.reset(seed=self.seed)
        states: list = [state]
        actions: list[int] = []
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        done = False
        while not done:
            state_t = torch.from_numpy(state.to_array())
            mask = self.env.action_mask()
            mask_t = torch.from_numpy(mask)
            logits = self.policy.forward(state_t)
            masked = self.policy._apply_mask(logits, mask_t)
            dist = torch.distributions.Categorical(logits=masked)
            action_t = dist.sample()
            log_p = dist.log_prob(action_t)
            action_id = int(action_t.item())
            next_state, reward, done, _ = self.env.step(action_id)
            actions.append(action_id)
            rewards.append(float(reward))
            log_probs.append(log_p)
            states.append(next_state)
            state = next_state
        return states, actions, rewards, log_probs

    # -------------------------------------------------------------------- train
    def train(self, episodes: int | None = None) -> REINFORCEHistory:
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
