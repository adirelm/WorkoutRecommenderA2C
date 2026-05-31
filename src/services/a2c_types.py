"""Typed dataclasses for A2C (Phase 5, brief §7.5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class A2CConfig:
    """A2C hyperparameters per config/config.yaml a2c.* keys."""

    actor_hidden: int = 128
    critic_hidden: int = 128
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    episodes: int = 500
    grad_clip_norm: float = 0.5


@dataclass(frozen=True)
class A2CHistory:
    """Per-episode log returned by A2CTrainer.train()."""

    episodes_run: int
    rewards: tuple[float, ...]
    actor_losses: tuple[float, ...]
    critic_losses: tuple[float, ...]
    advantages_mean: tuple[float, ...]  # per-episode mean(advantage) for diagnostics
    seed: int
