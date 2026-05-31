"""Helpers for WorkoutSDK — kept separate so sdk.py stays ≤150 LOC (CLAUDE.md §1)."""

from __future__ import annotations

import numpy as np
import torch

from src.env.state import STATE_DIM, State
from src.env.workout_env import WorkoutEnv
from src.model.actor_critic import ActorCriticNet
from src.model.policy_net import PolicyNet
from src.sdk.types import UNKNOWN_REWARD, PolicyHandle, WorkoutRecommendation
from src.services.a2c_trainer import A2CTrainer
from src.services.a2c_types import A2CConfig, A2CHistory
from src.services.comparator import ComparisonResult, compare
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig, REINFORCEHistory


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
    """Return (logits, scalar V(s)) — V=UNKNOWN_REWARD for REINFORCE PolicyNet (no critic)."""
    if isinstance(net, ActorCriticNet):
        logits, value = net.forward(state_t)
        return logits, float(value.detach().item())
    logits = net.forward(state_t)
    return logits, UNKNOWN_REWARD


def _mask_logits(logits: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Set logits at illegal positions to -inf so softmax zeros them out.

    Inlined here (rather than reaching into ``PolicyNet._apply_mask``) so the
    SDK layer does not depend on a private attribute of the model layer — the
    masking rule is one line and is part of the SDK's public recommendation
    contract, not an internal model detail.
    """
    neg_inf = torch.full_like(logits, float("-inf"))
    return torch.where(mask_t, logits, neg_inf)


def recommend_from_net(
    net: PolicyNet | ActorCriticNet,
    state: State,
    mask: np.ndarray,
    action_names: tuple[str, ...],
    action_count: int,
) -> WorkoutRecommendation:
    """Greedy-argmax recommendation from a trained policy (mask-aware).

    Returns softmax probabilities over masked logits (sums to 1), the picked
    action's name, an honest expected-reward proxy (critic value for A2C,
    :data:`UNKNOWN_REWARD` for REINFORCE), and a zero-vector for
    ``next_state_predicted`` — the SDK's ``recommend()`` deliberately does not
    wire a frozen world-model into this path. See
    :class:`WorkoutRecommendation` for the full contract.
    """
    state_arr = state.to_array()
    state_t = torch.as_tensor(state_arr, dtype=torch.float32)
    with torch.no_grad():
        logits, value = _logits_and_value(net, state_t)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        masked = _mask_logits(logits, mask_t)
        probs_t = torch.softmax(masked, dim=-1)
        probs = tuple(float(p) for p in probs_t.tolist())
        action_id = int(torch.argmax(masked).item())
    if not 0 <= action_id < action_count:
        raise RuntimeError(f"argmax produced out-of-range action_id={action_id}")
    return WorkoutRecommendation(
        action_id=action_id,
        action_name=action_names[action_id],
        probs=probs,
        next_state_predicted=(0.0,) * STATE_DIM,
        expected_reward=value,
    )


def run_compare_sweep(
    base_seed: int, seeds: int, episodes: int
) -> tuple[ComparisonResult, ActorCriticNet, PolicyHandle]:
    """Run REINFORCE + A2C over N seeds x E episodes; return result + last A2C net/handle."""
    r_hists: list[REINFORCEHistory] = []
    a_hists: list[A2CHistory] = []
    a2c_nets: list[ActorCriticNet] = []
    a2c_handles: list[PolicyHandle] = []
    for s in range(int(seeds)):
        seed = int(base_seed) + s
        env_r = WorkoutEnv(seed=seed)
        policy = PolicyNet(seed=seed)
        r_hists.append(
            REINFORCETrainer(policy, env_r, REINFORCEConfig(episodes=int(episodes)), seed=seed).train(
                episodes=int(episodes)
            )
        )
        env_a = WorkoutEnv(seed=seed)
        ac = ActorCriticNet(seed=seed)
        a_hist = A2CTrainer(ac, env_a, A2CConfig(episodes=int(episodes)), seed=seed).train(
            episodes=int(episodes)
        )
        a_hists.append(a_hist)
        a2c_nets.append(ac)
        a2c_handles.append(build_policy_handle("A2C", a_hist.rewards, a_hist.episodes_run))
    return compare(r_hists, a_hists), a2c_nets[-1], a2c_handles[-1]
