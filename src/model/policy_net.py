"""Stochastic policy network: π_θ(a|s) over 7 discrete actions (brief §7.4 + TR5).

Architecture (TR5 grading anchor):
    Linear(STATE_DIM, hidden=128) → ReLU → Linear(128, ACTION_COUNT)

Outputs are *logits*; softmax happens at sample/log_prob time so action
masking (ADR-004 / Huang & Ontañón 2022) can set illegal positions to
-inf *before* normalisation. Sampling uses ``torch.distributions.Categorical``
— NOT argmax — so REINFORCE/A2C retain the exploration their on-policy
estimators require (brief §7.4, TR5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from src.env.state import ACTION_COUNT, STATE_DIM

_DEFAULT_HIDDEN = 128
_EXPECTED_INPUT_NDIM_SINGLE = 1
_EXPECTED_INPUT_NDIM_BATCH = 2


class PolicyNet(nn.Module):
    """1-layer FC stochastic policy. Inputs: state (STATE_DIM,). Outputs: logits over actions."""

    def __init__(self, hidden: int = _DEFAULT_HIDDEN, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.hidden = int(hidden)
        self.fc = nn.Linear(STATE_DIM, self.hidden)
        self.head = nn.Linear(self.hidden, ACTION_COUNT)

    # ----------------------------------------------------------------- forward
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (STATE_DIM,) or (batch, STATE_DIM) -> logits same leading shape x ACTION_COUNT."""
        if state.dim() not in (_EXPECTED_INPUT_NDIM_SINGLE, _EXPECTED_INPUT_NDIM_BATCH):
            raise ValueError(
                f"state must be 1-D (STATE_DIM,) or 2-D (batch, STATE_DIM); got {tuple(state.shape)}"
            )
        if state.size(-1) != STATE_DIM:
            raise ValueError(f"state last dim must be {STATE_DIM}; got {state.size(-1)}")
        h = F.relu(self.fc(state))
        return self.head(h)

    # ---------------------------------------------------- masked logit helper
    @staticmethod
    def _apply_mask(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        if action_mask.shape[-1] != ACTION_COUNT:
            raise ValueError(f"action_mask last dim must be {ACTION_COUNT}; got {action_mask.shape[-1]}")
        mask_bool = action_mask.to(dtype=torch.bool)
        neg_inf = torch.full_like(logits, float("-inf"))
        return torch.where(mask_bool, logits, neg_inf)

    # ----------------------------------------------------------------- sample
    def sample(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[int, float]:
        """Sample one action from Categorical(softmax(masked_logits)).

        Returns ``(action_id, log_prob)`` as Python primitives so callers
        (trainer/agent) do not need to unwrap tensors. Use ``log_prob()``
        for the differentiable path used inside the loss.
        """
        logits = self.forward(state)
        masked = self._apply_mask(logits, action_mask)
        dist = Categorical(logits=masked)
        action = dist.sample()
        log_p = dist.log_prob(action)
        return int(action.item()), float(log_p.item())

    # --------------------------------------------------------------- log_prob
    def log_prob(
        self,
        state: torch.Tensor,
        action_id: int,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable log π(a | s) at the supplied action — used in the policy-gradient loss."""
        if not 0 <= int(action_id) < ACTION_COUNT:
            raise ValueError(f"action_id must be in [0, {ACTION_COUNT}); got {action_id}")
        logits = self.forward(state)
        masked = self._apply_mask(logits, action_mask)
        dist = Categorical(logits=masked)
        action_tensor = torch.tensor(int(action_id), dtype=torch.long, device=logits.device)
        return dist.log_prob(action_tensor)

    # ------------------------------------------------------------- properties
    @property
    def num_parameters(self) -> int:
        """Total parameter count (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
