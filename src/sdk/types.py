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


# Sentinel used by WorkoutRecommendation.expected_reward when the active
# policy has no critic (e.g. REINFORCE). Keeping it as a named constant
# makes the "0.0 means 'unknown', not 'predicted zero reward'" contract
# greppable from callers (notebooks, CLI, tests).
UNKNOWN_REWARD: float = 0.0


@dataclass(frozen=True)
class WorkoutRecommendation:
    """Output of TrainingSDK.recommend(state).

    Fields:
        action_id / action_name: greedy-argmax pick over the masked softmax.
        probs: softmax probabilities over the 7 actions (sums to 1.0; illegal
            actions are masked to 0 *before* normalisation).
        next_state_predicted: length-STATE_DIM tuple. **In Phase 6 the SDK's
            ``recommend()`` does not wire a frozen world-model into the
            recommendation path**, so this field is populated with a zero
            vector ``(0.0,) * STATE_DIM`` as an explicit "unknown — no
            world-model rollout was performed here" sentinel. The Phase 7
            notebook drives the LSTM world-model rollout separately for the
            analysis, so the SDK does not duplicate that responsibility.
        expected_reward: value-estimate-if-available. For A2C this is the
            critic value V(s). For REINFORCE there is no critic, so this is
            set to :data:`UNKNOWN_REWARD` (= 0.0) — the value 0.0 here means
            "unknown", *not* "the policy predicts a zero-reward action".
    """

    action_id: int
    action_name: str
    probs: tuple[float, ...]  # length ACTION_COUNT = 7
    next_state_predicted: tuple[float, ...]  # length STATE_DIM = 12, zero-vector sentinel in Phase 6
    expected_reward: float  # critic V(s) for A2C; UNKNOWN_REWARD (0.0) for REINFORCE
