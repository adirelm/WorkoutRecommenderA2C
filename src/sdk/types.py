"""Public dataclasses for the WorkoutSDK facade (Phase 6 / PRD §6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogbookHandle:
    """Opaque handle for the synthetic trainee dataset (loaded program + daily trajectory)."""

    program_name: str
    n_days: int
    state_dim: int


@dataclass(frozen=True)
class WorldModelHandle:
    """Opaque handle wrapping a trained, frozen LSTMWorldModel."""

    n_params: int
    val_loss_final: float
    epochs_trained: int


@dataclass(frozen=True)
class PolicyHandle:
    """Opaque handle wrapping a trained PolicyNet (REINFORCE) or ActorCriticNet (A2C)."""

    algorithm: str  # "REINFORCE" or "A2C"
    final_reward: float
    episodes_trained: int


@dataclass(frozen=True)
class WorkoutRecommendation:
    """Output of TrainingSDK.recommend(state)."""

    action_id: int
    action_name: str
    probs: tuple[float, ...]  # length ACTION_COUNT = 7
    next_state_predicted: tuple[float, ...]  # length STATE_DIM = 12
    expected_reward: float
