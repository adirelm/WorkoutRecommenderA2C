"""BaseTrainer — abstract policy-gradient trainer (brief §16 "building blocks").

V3 §4 / §12 mandate: every on-policy trainer in this codebase shares the
same skeleton — seed the world, reset the env, sample masked actions from
a ``Categorical(logits=masked)``, log per-step (state, action, reward, log
π, entropy, [value]), then run an algorithm-specific update. REINFORCE
and A2C used to duplicate that skeleton; this ABC hoists it once and lets
subclasses override only the bits that actually differ between algorithms.

Extension points (open-closed per V3 §12):

* :meth:`forward_step` — abstract; takes the (state_tensor, mask_tensor)
  pair and returns ``(log_prob, entropy, value_or_None)`` plus the
  sampled ``action_id`` (Python int). The base class never inspects the
  network, so subclasses can plug in any architecture.
* :meth:`train` — abstract; the per-algorithm update logic
  (REINFORCE = single Adam step on log_prob · (G_t - b);
  A2C = TD-advantage + separate actor/critic Adams + entropy bonus).

Adding a third algorithm (e.g. PPO) becomes:

    class PPOTrainer(BaseTrainer):
        def forward_step(self, state_t, mask_t): ...
        def train(self, episodes): ...
    # then register in sdk.WorkoutSDK._TRAINER_REGISTRY — zero SDK edits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from src.env.workout_env import WorkoutEnv
from src.utils.seeding import set_global_seed


@dataclass
class StepRecord:
    """One env-step's tensors + scalars (used by :meth:`rollout_episode`)."""

    state_t: torch.Tensor
    action_id: int
    reward: float
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor | None
    done: bool


class BaseTrainer(ABC):
    """Abstract on-policy trainer; subclasses implement forward_step + train."""

    def __init__(self, env: WorkoutEnv, config: Any, seed: int = 42) -> None:
        """Seed everything, then store env + config. Subclasses build optimisers."""
        self.env = env
        self.config = config
        self.seed = int(seed)
        set_global_seed(self.seed)

    # ----------------------------------------------------------- abstract hooks
    @abstractmethod
    def forward_step(
        self,
        state_t: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample one action under the current policy.

        Args:
            state_t: state tensor of shape ``(STATE_DIM,)``.
            mask_t: bool tensor of shape ``(ACTION_COUNT,)``.

        Returns:
            ``(action_id, log_prob, entropy, value_or_None)`` — log_prob and
            entropy must be differentiable scalar tensors connected to the
            policy parameters; value is None for value-free algorithms
            (REINFORCE) and a scalar critic prediction for actor-critic
            algorithms (A2C).
        """

    @abstractmethod
    def train(self, episodes: int | None = None) -> Any:
        """Episode loop with per-algorithm update; returns a History dataclass."""

    # ----------------------------------------------------------- shared helpers
    def sample_action(
        self,
        logits: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, Categorical]:
        """Mask logits → Categorical → sample. Shared by every subclass."""
        if mask_t.dtype != torch.bool:
            mask_t = mask_t.to(dtype=torch.bool)
        neg_inf = torch.full_like(logits, float("-inf"))
        masked = torch.where(mask_t, logits, neg_inf)
        dist = Categorical(logits=masked)
        action_t = dist.sample()
        return int(action_t.item()), dist.log_prob(action_t), dist.entropy(), dist

    def rollout_episode(self) -> list[StepRecord]:
        """Roll out one episode under the *current* policy. No gradient step.

        Delegates the forward pass to :meth:`forward_step`. Subclasses get a
        clean per-step record stream they can fold into their own update.
        """
        state = self.env.reset()
        records: list[StepRecord] = []
        done = False
        while not done:
            state_t = torch.as_tensor(state.to_array(), dtype=torch.float32)
            mask_t = torch.as_tensor(self.env.action_mask(), dtype=torch.bool)
            action_id, log_prob, entropy, value = self.forward_step(state_t, mask_t)
            next_state, reward, done, _info = self.env.step(action_id)
            records.append(
                StepRecord(
                    state_t=state_t,
                    action_id=action_id,
                    reward=float(reward),
                    log_prob=log_prob,
                    entropy=entropy,
                    value=value,
                    done=bool(done),
                )
            )
            state = next_state
        return records
