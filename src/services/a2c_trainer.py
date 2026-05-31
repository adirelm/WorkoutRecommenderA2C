"""A2C training loop (brief §7.5).

A2C-specific update lives here; shared rollout / sampling / seeding
belongs to :class:`src.services.base_trainer.BaseTrainer` (V3 §4
no-duplication, §12 open-closed extension point).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import optim

from src.env.workout_env import WorkoutEnv
from src.model.actor_critic import ActorCriticNet
from src.services.a2c_helpers import actor_loss, compute_advantages_td, critic_loss
from src.services.a2c_types import A2CConfig, A2CHistory
from src.services.base_trainer import BaseTrainer


class A2CTrainer(BaseTrainer):
    """Synchronous A2C with separate actor + critic Adams."""

    def __init__(
        self,
        ac_net: ActorCriticNet,
        env: WorkoutEnv,
        config: A2CConfig | None = None,
        seed: int = 42,
    ) -> None:
        cfg = config or A2CConfig()
        super().__init__(env=env, config=cfg, seed=seed)
        self.ac_net = ac_net
        self.actor_optim = optim.Adam(
            [*self.ac_net.actor_fc.parameters(), *self.ac_net.actor_head.parameters()],
            lr=cfg.actor_lr,
        )
        self.critic_optim = optim.Adam(
            [*self.ac_net.critic_fc.parameters(), *self.ac_net.critic_head.parameters()],
            lr=cfg.critic_lr,
        )

    # ----------------------------------------------------------- forward hook
    def forward_step(
        self,
        state_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """A2C forward: actor-critic logits+value → masked Categorical → sample."""
        logits, value = self.ac_net.forward(state_t)
        action_id, log_prob, entropy, _dist = self.sample_action(logits, mask_t)
        return action_id, log_prob, entropy, value

    # ----------------------------------------------------------- legacy rollout
    def run_episode(self) -> tuple[list, list, list, list, list, list, list]:
        """Roll out one episode.

        Returns (states, actions, rewards, log_probs, values, dones, entropies).
        Env is re-seeded ONCE in __init__; subsequent resets advance the env RNG
        naturally so each episode samples a fresh trajectory.
        """
        records = self.rollout_episode()
        states = [r.state_t for r in records]
        actions = [r.action_id for r in records]
        rewards = [r.reward for r in records]
        log_probs = [r.log_prob for r in records]
        values = [r.value for r in records]
        dones = [r.done for r in records]
        entropies = [r.entropy for r in records]
        return states, actions, rewards, log_probs, values, dones, entropies

    def train(self, episodes: int | None = None) -> A2CHistory:
        """Episode loop with separate actor+critic Adams + TD-advantage; brief §7.5."""
        episodes = int(episodes if episodes is not None else self.config.episodes)
        rewards_hist: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        adv_means: list[float] = []
        for _ in range(episodes):
            _states, _, rewards, log_probs, values, dones, entropies = self.run_episode()
            # next_values: shift values left by 1; last is 0 (terminal absorbing)
            value_floats = [float(v.detach().item()) for v in values]
            next_values = [*value_floats[1:], 0.0]
            advantages = compute_advantages_td(
                rewards, value_floats, next_values, dones, gamma=self.config.gamma
            )
            targets = [adv + v for adv, v in zip(advantages, value_floats, strict=True)]
            a_loss = actor_loss(
                log_probs, advantages, entropy_coef=self.config.entropy_coef, entropies=entropies
            )
            c_loss = critic_loss(values, targets)
            actor_params = [
                *self.ac_net.actor_fc.parameters(),
                *self.ac_net.actor_head.parameters(),
            ]
            critic_params = [
                *self.ac_net.critic_fc.parameters(),
                *self.ac_net.critic_head.parameters(),
            ]
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            a_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(actor_params, self.config.grad_clip_norm)
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, self.config.grad_clip_norm)
            self.critic_optim.step()
            rewards_hist.append(float(np.sum(rewards)))
            actor_losses.append(float(a_loss.detach().item()))
            critic_losses.append(float(c_loss.detach().item()))
            adv_means.append(float(np.mean(advantages)) if advantages else 0.0)
        return A2CHistory(
            episodes_run=episodes,
            rewards=tuple(rewards_hist),
            actor_losses=tuple(actor_losses),
            critic_losses=tuple(critic_losses),
            advantages_mean=tuple(adv_means),
            seed=self.seed,
        )
