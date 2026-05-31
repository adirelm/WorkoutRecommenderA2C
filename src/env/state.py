"""State vector for the env layer (ADR-002 — C2_moderate_12d, brief §7.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 12
STATE_CHANNEL_NAMES: tuple[str, ...] = (
    "fatigue",
    "soreness_push",
    "soreness_pull",
    "soreness_legs",
    "soreness_core",
    "readiness",
    "rolling_7d_volume",
    "streak_days_trained",
    "days_since_last_rest",
    "muscle_balance_push_vs_pull",
    "adherence_signal",
    "weekly_progress",
)
assert len(STATE_CHANNEL_NAMES) == STATE_DIM


@dataclass(frozen=True)
class State:
    """Immutable 12-channel trainee state. Values in normalised ranges (see ADR-002)."""

    fatigue: float  # [0, 1]
    soreness_push: float  # [0, 1]
    soreness_pull: float  # [0, 1]
    soreness_legs: float  # [0, 1]
    soreness_core: float  # [0, 1]
    readiness: float  # [0, 1]
    rolling_7d_volume: float  # raw volume (un-normalised)
    streak_days_trained: int  # 0..N
    days_since_last_rest: int  # 0..N
    muscle_balance_push_vs_pull: float  # [-1, 1]
    adherence_signal: float  # [-1, 1]
    weekly_progress: float  # [0, 1.2] capped

    def to_array(self) -> np.ndarray:
        """Pack the 12 state channels into a float32 ndarray in STATE_CHANNEL_NAMES order."""
        return np.asarray(
            [
                self.fatigue,
                self.soreness_push,
                self.soreness_pull,
                self.soreness_legs,
                self.soreness_core,
                self.readiness,
                self.rolling_7d_volume,
                float(self.streak_days_trained),
                float(self.days_since_last_rest),
                self.muscle_balance_push_vs_pull,
                self.adherence_signal,
                self.weekly_progress,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Inverse of to_array — round-trip: State.from_array(s.to_array()) == s."""
        if arr.shape != (STATE_DIM,):
            raise ValueError(f"expected shape ({STATE_DIM},), got {arr.shape}")
        return cls(
            fatigue=float(arr[0]),
            soreness_push=float(arr[1]),
            soreness_pull=float(arr[2]),
            soreness_legs=float(arr[3]),
            soreness_core=float(arr[4]),
            readiness=float(arr[5]),
            rolling_7d_volume=float(arr[6]),
            streak_days_trained=int(arr[7]),
            days_since_last_rest=int(arr[8]),
            muscle_balance_push_vs_pull=float(arr[9]),
            adherence_signal=float(arr[10]),
            weekly_progress=float(arr[11]),
        )

    @classmethod
    def initial(cls) -> State:
        """A rested baseline: zero fatigue/soreness/streak, readiness=1.0."""
        return cls(
            fatigue=0.0,
            soreness_push=0.0,
            soreness_pull=0.0,
            soreness_legs=0.0,
            soreness_core=0.0,
            readiness=1.0,
            rolling_7d_volume=0.0,
            streak_days_trained=0,
            days_since_last_rest=0,
            muscle_balance_push_vs_pull=0.0,
            adherence_signal=0.0,
            weekly_progress=0.0,
        )


ACTION_NAMES: tuple[str, ...] = ("Rest", "Push", "Pull", "Legs", "FullBody", "Conditioning", "Mobility")
EXPECTED_ACTION_COUNT = 7
ACTION_COUNT = len(ACTION_NAMES)
assert ACTION_COUNT == EXPECTED_ACTION_COUNT
