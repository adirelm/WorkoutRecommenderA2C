"""A2C training loop (brief §7.5)."""

from __future__ import annotations

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

from src.env.workout_env import WorkoutEnv
from src.model.actor_critic import ActorCriticNet
from src.services.a2c_helpers import actor_loss, compute_advantages_td, critic_loss
from src.services.a2c_types import A2CConfig, A2CHistory
from src.utils.seeding import set_global_seed


class A2CTrainer:
    """Synchronous A2C with separate actor + critic Adams."""

    def __init__(
        self,
        ac_net: ActorCriticNet,
        env: WorkoutEnv,
        config: A2CConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.ac_net = ac_net
        self.env = env
        self.config = config or A2CConfig()
        self.seed = int(seed)
        set_global_seed(self.seed)
        self.actor_optim = optim.Adam(
            [*self.ac_net.actor_fc.parameters(), *self.ac_net.actor_head.parameters()],
            lr=self.config.actor_lr,
        )
        self.critic_optim = optim.Adam(
            [*self.ac_net.critic_fc.parameters(), *self.ac_net.critic_head.parameters()],
            lr=self.config.critic_lr,
        )

    def run_episode(self) -> tuple[list, list, list, list, list, list]:
        """Roll out one episode. Returns (states, actions, rewards, log_probs, values, dones)."""
        self.env.reset(seed=self.seed)
        states: list = []
        actions: list[int] = []
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        dones: list[bool] = []
        done = False
        s = self.env.reset(seed=self.seed)
        while not done:
            state_t = torch.as_tensor(s.to_array(), dtype=torch.float32)
            mask = torch.as_tensor(self.env.action_mask(), dtype=torch.bool)
            logits, value = self.ac_net.forward(state_t)
            masked = self.ac_net._apply_mask(logits, mask)
            dist = Categorical(logits=masked)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            s, r, done, _ = self.env.step(int(action.item()))
            states.append(state_t)
            actions.append(int(action.item()))
            rewards.append(float(r))
            log_probs.append(log_prob)
            values.append(value)
            dones.append(bool(done))
        return states, actions, rewards, log_probs, values, dones

    def train(self, episodes: int | None = None) -> A2CHistory:
        episodes = int(episodes if episodes is not None else self.config.episodes)
        rewards_hist: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        adv_means: list[float] = []
        for _ in range(episodes):
            _states, _, rewards, log_probs, values, dones = self.run_episode()
            # next_values: shift values left by 1; last is 0 (terminal absorbing)
            value_floats = [float(v.detach().item()) for v in values]
            next_values = [*value_floats[1:], 0.0]
            advantages = compute_advantages_td(
                rewards, value_floats, next_values, dones, gamma=self.config.gamma
            )
            targets = [adv + v for adv, v in zip(advantages, value_floats, strict=True)]
            # entropy estimate: H(pi) = -sum_a p log p; cheap proxy via -log_prob.mean()
            entropies = [-lp.detach() for lp in log_probs]
            a_loss = actor_loss(
                log_probs, advantages, entropy_coef=self.config.entropy_coef, entropies=entropies
            )
            c_loss = critic_loss(values, targets)
            self.actor_optim.zero_grad()
            a_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.config.grad_clip_norm)
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.config.grad_clip_norm)
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
