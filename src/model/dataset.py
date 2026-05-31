"""Sliding-window supervised dataset builder for the LSTM world model.

Given a trajectory list[(State, action_id, next_State)] of length T,
emit (T - window_len + 1) TransitionWindow training examples (one per
index i in [window_len-1, T-1]).
"""

from __future__ import annotations

import numpy as np

from src.env.state import STATE_DIM, State
from src.model.types import TransitionWindow


def build_windows(trajectory: list[tuple[State, int, State]], window_len: int = 7) -> list[TransitionWindow]:
    """Build sliding windows from a trajectory.

    For each index i in [window_len-1, T-1], emit a TransitionWindow with:
      - state_seq: states[i-window_len+1 : i+1]  (window_len states ending with state at i)
      - action_seq: actions[i-window_len+1 : i+1]
      - next_state: trajectory[i].next_state

    If T < window_len the trajectory is too short and an empty list is returned.
    """
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    T = len(trajectory)  # noqa: N806 — T matches mathematical notation in docstring
    if window_len > T:
        return []

    # Pre-extract arrays once (avoids re-allocating inside the loop).
    states = np.stack([t[0].to_array() for t in trajectory], axis=0).astype(np.float32)
    actions = np.asarray([int(t[1]) for t in trajectory], dtype=np.int64)
    next_states = np.stack([t[2].to_array() for t in trajectory], axis=0).astype(np.float32)

    assert states.shape == (T, STATE_DIM), f"states shape {states.shape} != ({T}, {STATE_DIM})"

    windows: list[TransitionWindow] = []
    for i in range(window_len - 1, T):
        start = i - window_len + 1
        end = i + 1
        windows.append(
            TransitionWindow(
                state_seq=states[start:end].copy(),
                action_seq=actions[start:end].copy(),
                next_state=next_states[i].copy(),
            )
        )
    return windows


def split_train_val(
    windows: list[TransitionWindow], val_days: int = 7
) -> tuple[list[TransitionWindow], list[TransitionWindow]]:
    """Chronological split — last `val_days` entries become validation.

    If val_days >= len(windows), everything goes to val and train is empty.
    If val_days <= 0, everything goes to train and val is empty.
    """
    if val_days <= 0:
        return list(windows), []
    if val_days >= len(windows):
        return [], list(windows)
    cut = len(windows) - val_days
    return list(windows[:cut]), list(windows[cut:])
