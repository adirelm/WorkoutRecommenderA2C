"""Actor-Critic network for A2C (brief §7.5).

Two separate hidden-layer-1 heads (no shared trunk) to match brief §5.2:
* Actor: state(12) -> Linear(actor_hidden) -> ReLU -> Linear(ACTION_COUNT) -> softmax logits
* Critic: state(12) -> Linear(critic_hidden) -> ReLU -> Linear(1) -> scalar V(s)

Sampling is Categorical (stochastic, NOT argmax), with action-mask via -inf
logit substitution (ADR-004 / Huang & Ontañón 2022).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from src.env.state import ACTION_COUNT, STATE_DIM
from src.model.masking import apply_mask

_EXPECTED_NDIM_SINGLE = 1
_EXPECTED_NDIM_BATCH = 2


class ActorCriticNet(nn.Module):
    """Actor + Critic with disjoint parameter sets (brief §5.2)."""

    def __init__(
        self,
        actor_hidden: int = 128,
        critic_hidden: int = 128,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.actor_hidden = int(actor_hidden)
        self.critic_hidden = int(critic_hidden)
        self.actor_fc = nn.Linear(STATE_DIM, self.actor_hidden)
        self.actor_head = nn.Linear(self.actor_hidden, ACTION_COUNT)
        self.critic_fc = nn.Linear(STATE_DIM, self.critic_hidden)
        self.critic_head = nn.Linear(self.critic_hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value). value squeezed to scalar if batch is 0-d / 1-d."""
        if state.dim() not in (_EXPECTED_NDIM_SINGLE, _EXPECTED_NDIM_BATCH):
            raise ValueError(f"state must be 1-D or 2-D; got {tuple(state.shape)}")
        if state.size(-1) != STATE_DIM:
            raise ValueError(f"state last dim must be {STATE_DIM}; got {state.size(-1)}")
        a = F.relu(self.actor_fc(state))
        logits = self.actor_head(a)
        c = F.relu(self.critic_fc(state))
        value = self.critic_head(c).squeeze(-1)
        return logits, value

    @staticmethod
    def _apply_mask(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        """Validate + delegate to shared :func:`src.model.masking.apply_mask`."""
        if action_mask is None:
            return logits
        if action_mask.shape[-1] != ACTION_COUNT:
            raise ValueError(f"action_mask last dim must be {ACTION_COUNT}; got {action_mask.shape[-1]}")
        return apply_mask(logits, action_mask)

    def sample(
        self, state: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[int, float, float]:
        """Returns (action_id, log_prob_value, critic_value) — primitives for the trainer."""
        logits, value = self.forward(state)
        masked = self._apply_mask(logits, action_mask)
        dist = Categorical(logits=masked)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def log_prob_and_value(
        self, state: torch.Tensor, action_id: int, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable (log π(a|s), V(s)) — used in actor + critic losses."""
        if not 0 <= int(action_id) < ACTION_COUNT:
            raise ValueError(f"action_id must be in [0, {ACTION_COUNT}); got {action_id}")
        logits, value = self.forward(state)
        masked = self._apply_mask(logits, action_mask)
        dist = Categorical(logits=masked)
        return dist.log_prob(torch.tensor(int(action_id))), value

    @property
    def num_parameters(self) -> int:
        """Total parameter count (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
