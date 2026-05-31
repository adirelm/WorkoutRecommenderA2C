"""Algorithmic helpers for REINFORCE (brief §7.4 eq. 16).

Two pure functions kept out of the trainer module so each piece stays
testable in isolation:

* :func:`compute_returns` — discounted return G_t = sum_{k=0..T-t} gamma^k r_{t+k}
  computed backwards in O(T).
* :func:`reinforce_loss` — REINFORCE policy-gradient surrogate loss with
  optional running-mean baseline subtraction:
      L = - mean_t [ log π_θ(a_t | s_t) · (G_t - b) ]
  The negative sign converts the *ascent* on E[R] into the *descent*
  PyTorch optimisers expect.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    """Backward-accumulated discounted returns. Returns len == len(rewards)."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1]; got {gamma}")
    g = 0.0
    out: list[float] = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g = float(rewards[t]) + float(gamma) * g
        out[t] = g
    return out


def reinforce_loss(
    log_probs: Sequence[torch.Tensor],
    returns: Sequence[float],
    baseline: float = 0.0,
) -> torch.Tensor:
    """REINFORCE policy-gradient loss:  L = - mean_t [ log π · (G_t - b) ].

    Args:
        log_probs: per-timestep log π_θ(a_t | s_t) — must be differentiable
            scalar tensors connected to the policy parameters.
        returns: per-timestep discounted returns (Python floats).
        baseline: scalar subtracted from every return; defaults to 0.0
            (vanilla REINFORCE).

    Returns:
        Scalar 0-d tensor — ``.backward()``-ready.
    """
    if len(log_probs) != len(returns):
        raise ValueError(f"log_probs ({len(log_probs)}) and returns ({len(returns)}) length mismatch")
    if len(log_probs) == 0:
        raise ValueError("reinforce_loss called with empty trajectory")
    advantages = torch.tensor(
        [float(g) - float(baseline) for g in returns],
        dtype=torch.float32,
        device=log_probs[0].device,
    )
    stacked = torch.stack(list(log_probs))
    return -(stacked * advantages).mean()
