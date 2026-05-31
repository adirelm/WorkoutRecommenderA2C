"""Helpers for WorkoutSDK — kept separate so sdk.py stays ≤150 LOC (CLAUDE.md §1)."""

from __future__ import annotations

import numpy as np
import torch

from src.env.state import State
from src.model.actor_critic import ActorCriticNet
from src.model.policy_net import PolicyNet
from src.sdk.types import PolicyHandle, WorkoutRecommendation


def build_policy_handle(
    algorithm: str,
    rewards: tuple[float, ...],
    episodes_run: int,
) -> PolicyHandle:
    """Wrap a finished training run in an opaque PolicyHandle."""
    final = float(rewards[-1]) if rewards else 0.0
    return PolicyHandle(
        algorithm=str(algorithm),
        final_reward=final,
        episodes_trained=int(episodes_run),
    )


def _logits_and_value(
    net: PolicyNet | ActorCriticNet,
    state_t: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Return (logits, scalar V(s)) — V=0.0 for REINFORCE PolicyNet (no critic)."""
    if isinstance(net, ActorCriticNet):
        logits, value = net.forward(state_t)
        return logits, float(value.detach().item())
    logits = net.forward(state_t)
    return logits, 0.0


def recommend_from_net(
    net: PolicyNet | ActorCriticNet,
    state: State,
    mask: np.ndarray,
    action_names: tuple[str, ...],
    action_count: int,
) -> WorkoutRecommendation:
    """Greedy-argmax recommendation from a trained policy (mask-aware).

    Returns probabilities (softmax over masked logits, sums to 1), the picked
    action's name, an expected reward proxy (critic value for A2C, 0.0 for
    REINFORCE), and the *current* state as a stand-in for the predicted next
    state (the SDK does not own the LSTM rollout — that's Phase 3 territory).
    """
    state_arr = state.to_array()
    state_t = torch.as_tensor(state_arr, dtype=torch.float32)
    with torch.no_grad():
        logits, value = _logits_and_value(net, state_t)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        masked = net._apply_mask(logits, mask_t)
        probs_t = torch.softmax(masked, dim=-1)
        probs = tuple(float(p) for p in probs_t.tolist())
        action_id = int(torch.argmax(masked).item())
    if not 0 <= action_id < action_count:
        raise RuntimeError(f"argmax produced out-of-range action_id={action_id}")
    return WorkoutRecommendation(
        action_id=action_id,
        action_name=action_names[action_id],
        probs=probs,
        next_state_predicted=tuple(float(x) for x in state_arr.tolist()),
        expected_reward=value,
    )
