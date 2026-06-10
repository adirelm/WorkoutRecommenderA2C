"""Typed dataclasses for A2C (Phase 5, brief §7.5)."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config_loader import load_config


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

    @classmethod
    def from_yaml(cls, episodes: int | None = None) -> A2CConfig:
        """Build from config.yaml [a2c] — the runtime source of truth (V3 §7.2)."""
        c = load_config()["a2c"]
        return cls(
            actor_hidden=int(c["actor_hidden"]),
            critic_hidden=int(c["critic_hidden"]),
            actor_lr=float(c["actor_lr"]),
            critic_lr=float(c["critic_lr"]),
            gamma=float(c["gamma"]),
            entropy_coef=float(c["entropy_coef"]),
            episodes=int(episodes if episodes is not None else c["episodes"]),
            grad_clip_norm=float(c["grad_clip_norm"]),
        )


@dataclass(frozen=True)
class A2CHistory:
    """Per-episode log returned by A2CTrainer.train()."""

    episodes_run: int
    rewards: tuple[float, ...]
    actor_losses: tuple[float, ...]
    critic_losses: tuple[float, ...]
    advantages_mean: tuple[float, ...]  # per-episode mean(advantage) for diagnostics
    seed: int
