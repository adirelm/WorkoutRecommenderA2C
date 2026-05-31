"""Typed dataclasses for the services layer (Phase 4: REINFORCE, Phase 5: A2C)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class REINFORCEConfig:
    """REINFORCE hyperparameters per config/config.yaml reinforce.* keys."""

    policy_hidden: int = 128  # TR5: brief lecture spec — 1-layer FC, 128 neurons
    lr: float = 3.0e-4
    gamma: float = 0.99
    episodes: int = 500
    baseline_alpha: float = 0.05
    entropy_coef: float = 0.0
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class EpisodeResult:
    """One episode's trace — used by trainers + analysis notebook."""

    episode: int
    total_reward: float
    length: int
    actions: tuple[int, ...]
    states: tuple  # tuple[State, ...] but avoid circular import; trainer constructs
    rewards: tuple[float, ...]
    loss: float


@dataclass(frozen=True)
class REINFORCEHistory:
    """Per-episode log returned by REINFORCETrainer.train()."""

    episodes_run: int
    rewards: tuple[float, ...]  # length = episodes_run; total reward per episode
    losses: tuple[float, ...]  # length = episodes_run; per-episode loss
    baseline: tuple[float, ...]  # running mean estimate per episode
    seed: int
