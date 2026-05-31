"""LSTM-backed transition function for WorkoutEnv (brief §7.3).

Phase-4 replacement for SyntheticTrainee: wraps a trained, FROZEN
LSTMWorldModel + a rolling state/action history buffer to expose the
same ``next_state(state, action_id, prescribed_volume, prescribed_muscles)``
contract. The prescribed_* args are accepted (API parity) but ignored —
the LSTM is trained to predict (state, action) → next_state.

Output values are clamped to declared State ranges (see ADR-002):
- fatigue / soreness_* / readiness / weekly_progress → [0, 1]  (weekly_progress
  capped at 1.2 per the State docstring)
- muscle_balance_push_vs_pull / adherence_signal → [-1, 1]
- rolling_7d_volume → ≥ 0
- streak_days_trained / days_since_last_rest → non-negative int (round + clamp)
"""

from __future__ import annotations

import numpy as np
import torch

from src.env.state import ACTION_COUNT, STATE_DIM, State
from src.model.lstm_world import LSTMWorldModel


class LSTMEnvAdapter:
    """Wraps a frozen LSTMWorldModel + history buffer; mimics SyntheticTrainee."""

    def __init__(self, model: LSTMWorldModel, window_len: int = 7) -> None:
        if window_len <= 0:
            raise ValueError(f"window_len must be positive, got {window_len}")
        if not model.is_frozen():
            raise RuntimeError(
                "Call model.freeze() before constructing LSTMEnvAdapter "
                "(brief §7.3 — LSTM frozen during RL phase)."
            )
        self.model = model
        self.window_len = int(window_len)
        self._state_history: list[State] = []
        self._action_history: list[int] = []

    def reset(self, initial_state: State) -> None:
        """Seed history with ``window_len`` copies of ``initial_state``.

        Action history is seeded with Rest (0) — the only action that asserts
        nothing about the trainee's prior week."""
        self._state_history = [initial_state] * self.window_len
        self._action_history = [0] * self.window_len

    @property
    def history_len(self) -> int:
        """Current size of the (state, action) history buffer."""
        return len(self._state_history)

    def next_state(
        self,
        state: State,
        action_id: int,
        prescribed_volume: float = 0.0,
        prescribed_muscles: dict[str, float] | None = None,
    ) -> State:
        """Forward (history + new (state, action)) through the LSTM; return next State.

        ``prescribed_volume`` and ``prescribed_muscles`` are accepted for API
        parity with :class:`SyntheticTrainee` and intentionally ignored — the
        LSTM is trained to predict (state, action) → next_state."""
        del prescribed_volume, prescribed_muscles  # API parity, not consumed here
        if not self.model.is_frozen():
            raise RuntimeError("LSTMEnvAdapter requires a frozen model (call model.freeze()).")
        if not 0 <= int(action_id) < ACTION_COUNT:
            raise ValueError(f"action_id={action_id} outside [0, {ACTION_COUNT})")

        # Append current step then roll the buffer back to window_len.
        self._state_history.append(state)
        self._action_history.append(int(action_id))
        if len(self._state_history) > self.window_len:
            self._state_history = self._state_history[-self.window_len :]
            self._action_history = self._action_history[-self.window_len :]

        state_arr = np.stack([s.to_array() for s in self._state_history], axis=0)
        state_tensor = torch.from_numpy(state_arr).unsqueeze(0).float()  # (1, W, STATE_DIM)
        action_tensor = torch.tensor(self._action_history, dtype=torch.int64).unsqueeze(0)  # (1, W)

        with torch.no_grad():
            pred = self.model(state_tensor, action_tensor)  # (1, STATE_DIM)
        vec = pred.squeeze(0).cpu().numpy()
        assert vec.shape == (STATE_DIM,)
        return _vector_to_state(vec)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _clip_signed(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _nonneg_int(x: float) -> int:
    return int(max(0, round(float(x))))


def _vector_to_state(vec: np.ndarray) -> State:
    """Map an LSTM 12-d output vector to a valid State (clamped per channel)."""
    return State(
        fatigue=_clip01(vec[0]),
        soreness_push=_clip01(vec[1]),
        soreness_pull=_clip01(vec[2]),
        soreness_legs=_clip01(vec[3]),
        soreness_core=_clip01(vec[4]),
        readiness=_clip01(vec[5]),
        rolling_7d_volume=float(max(0.0, vec[6])),
        streak_days_trained=_nonneg_int(vec[7]),
        days_since_last_rest=_nonneg_int(vec[8]),
        muscle_balance_push_vs_pull=_clip_signed(vec[9]),
        adherence_signal=_clip_signed(vec[10]),
        weekly_progress=float(np.clip(vec[11], 0.0, 1.2)),
    )
