"""A2C helpers — TD-error advantage + actor loss + critic MSE (brief §7.5)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812


def compute_advantages_td(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    dones: Sequence[bool],
    gamma: float = 0.99,
) -> list[float]:
    """Brief eq. 9 / 17: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1]; got {gamma}")
    n = len(rewards)
    if len(values) != n or len(next_values) != n or len(dones) != n:
        raise ValueError("rewards/values/next_values/dones must have equal lengths")
    advantages: list[float] = []
    for t in range(n):
        bootstrap = 0.0 if dones[t] else float(gamma) * float(next_values[t])
        delta = float(rewards[t]) + bootstrap - float(values[t])
        advantages.append(delta)
    return advantages


def actor_loss(
    log_probs: Sequence[torch.Tensor],
    advantages: Sequence[float],
    entropy_coef: float = 0.01,
    entropies: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Brief eq. 10: L_actor = -mean(log pi * adv.detach()) - entropy_coef * mean(H(pi))."""
    if len(log_probs) == 0:
        raise ValueError("actor_loss called with empty trajectory")
    if len(log_probs) != len(advantages):
        raise ValueError("log_probs and advantages length mismatch")
    # adv is built from Python floats → detached-by-construction; explicit cast for clarity
    adv = torch.tensor(list(advantages), dtype=torch.float32, device=log_probs[0].device)
    stacked = torch.stack(list(log_probs))
    pg_loss = -(stacked * adv).mean()
    if entropy_coef > 0.0 and entropies is not None and len(entropies) > 0:
        h_stacked = torch.stack(list(entropies))
        return pg_loss - entropy_coef * h_stacked.mean()
    return pg_loss


def critic_loss(
    values: Sequence[torch.Tensor],
    targets: Sequence[float],
) -> torch.Tensor:
    """Brief eq. 12: L_critic = mean((target - V(s))^2). Targets detached."""
    if len(values) == 0:
        raise ValueError("critic_loss called with empty trajectory")
    if len(values) != len(targets):
        raise ValueError("values and targets length mismatch")
    v_stacked = torch.stack(list(values))
    t_tensor = torch.tensor(list(targets), dtype=torch.float32, device=v_stacked.device)
    return F.mse_loss(v_stacked, t_tensor)
