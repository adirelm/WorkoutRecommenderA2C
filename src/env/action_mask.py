"""Action masking service (ADR-004, brief §7.6.1).

Applies hard constraints to the policy's action distribution by setting
logits of illegal actions to -inf so softmax assigns them zero probability.

Rules (ADR-004):
    1. Rest (id=0) masked when last 3 days were all Rest — avoids degenerate
       low-load loops.
    2. Legs (id=3) masked when state.soreness_legs > 0.8.
    3. Conditioning (id=5) masked when an overload signal exceeds the
       configured threshold (rolling_7d_volume used as proxy in Phase 2).
    4. Mobility (id=6) is NEVER masked — always a safe fallback.
"""

from __future__ import annotations

import numpy as np

from src.env.state import ACTION_COUNT, State

# Action ID constants (mirrors ACTION_NAMES order in src/env/state.py).
_REST_ID = 0
_LEGS_ID = 3
_CONDITIONING_ID = 5
_MOBILITY_ID = 6


class ActionMaskService:
    """Builds boolean legality masks and applies them to policy logits."""

    def __init__(
        self,
        rest_streak_threshold: int = 3,
        legs_soreness_threshold: float = 0.8,
        conditioning_overload_threshold: float = 1.0,
    ) -> None:
        self._rest_streak_threshold = int(rest_streak_threshold)
        self._legs_soreness_threshold = float(legs_soreness_threshold)
        self._conditioning_overload_threshold = float(conditioning_overload_threshold)

    # ------------------------------------------------------------------ mask
    def mask(self, state: State, history: list[int]) -> np.ndarray:
        """Return a boolean array of shape (ACTION_COUNT,).

        True = action is legal, False = action is masked.
        """
        mask = np.ones(ACTION_COUNT, dtype=bool)

        # Rule 1: Rest streak — block another Rest if last K days were all Rest.
        k = self._rest_streak_threshold
        if k > 0 and len(history) >= k:
            recent = history[-k:]
            if all(a == _REST_ID for a in recent):
                mask[_REST_ID] = False

        # Rule 2: Legs masked when soreness_legs above threshold.
        if state.soreness_legs > self._legs_soreness_threshold:
            mask[_LEGS_ID] = False

        # Rule 3: Conditioning masked when overload signal exceeds threshold.
        # Phase-2 proxy: rolling_7d_volume vs the configured threshold.
        if state.rolling_7d_volume > self._conditioning_overload_threshold:
            mask[_CONDITIONING_ID] = False

        # Rule 4: Mobility is always legal — re-assert in case other rules
        # mutate it later (defensive; current ruleset never touches it).
        mask[_MOBILITY_ID] = True

        return mask

    # -------------------------------------------------------- apply_to_logits
    def apply_to_logits(self, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Set logits[~mask] = -inf so softmax assigns zero probability.

        Returns a new array — does not mutate the input. The output keeps
        the input dtype; for integer inputs the result is promoted to float
        so -inf is representable.
        """
        if logits.shape != mask.shape:
            raise ValueError(f"logits shape {logits.shape} != mask shape {mask.shape}")
        out_dtype = logits.dtype if np.issubdtype(logits.dtype, np.floating) else np.float32
        out = logits.astype(out_dtype, copy=True)
        out[~mask.astype(bool)] = -np.inf
        return out
