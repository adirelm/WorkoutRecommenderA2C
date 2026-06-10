"""Typed dataclasses for the services layer (Phase 4: REINFORCE, Phase 5: A2C)."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config_loader import load_config


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

    @classmethod
    def from_yaml(cls, episodes: int | None = None) -> REINFORCEConfig:
        """Build from config.yaml [reinforce] — the runtime source of truth (V3 §7.2)."""
        c = load_config()["reinforce"]
        return cls(
            policy_hidden=int(c["policy_hidden"]),
            lr=float(c["lr"]),
            gamma=float(c["gamma"]),
            episodes=int(episodes if episodes is not None else c["episodes"]),
            baseline_alpha=float(c["baseline_alpha"]),
            entropy_coef=float(c["entropy_coef"]),
            grad_clip_norm=float(c["grad_clip_norm"]),
        )


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
