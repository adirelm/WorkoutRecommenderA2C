"""REINFORCE vs A2C side-by-side comparison (brief §7.6, DA4, TR1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    reinforce_mean_reward: np.ndarray
    reinforce_std_reward: np.ndarray
    a2c_mean_reward: np.ndarray
    a2c_std_reward: np.ndarray
    episode_count: int
    seed_count: int


def _stack(histories: list[Any]) -> np.ndarray:
    """histories[i].rewards is a tuple of floats; stack into (N, E)."""
    arrays = [np.asarray(h.rewards, dtype=np.float32) for h in histories]
    return np.stack(arrays, axis=0)


def compare(reinforce_histories: list[Any], a2c_histories: list[Any]) -> ComparisonResult:
    if len(reinforce_histories) == 0 or len(a2c_histories) == 0:
        raise ValueError("histories must be non-empty")
    if len(reinforce_histories) != len(a2c_histories):
        raise ValueError(
            f"seed-count mismatch: reinforce={len(reinforce_histories)} a2c={len(a2c_histories)}"
        )
    r_stack = _stack(reinforce_histories)
    a_stack = _stack(a2c_histories)
    if r_stack.shape[1] != a_stack.shape[1]:
        raise ValueError(f"episode-count mismatch: reinforce={r_stack.shape[1]} a2c={a_stack.shape[1]}")
    return ComparisonResult(
        reinforce_mean_reward=r_stack.mean(axis=0),
        reinforce_std_reward=r_stack.std(axis=0),
        a2c_mean_reward=a_stack.mean(axis=0),
        a2c_std_reward=a_stack.std(axis=0),
        episode_count=int(r_stack.shape[1]),
        seed_count=int(r_stack.shape[0]),
    )
