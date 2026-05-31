"""Reward function — eq. 15 in the brief (ADR-003, §7.4).

    r_t = gain_t - lambda_1 * overload_t - lambda_2 * imbalance_t

`gain` blends within-target progress and dietary-style variety; `overload`
penalises super-linear excess over the rolling baseline; `imbalance`
penalises muscle-group skew. Decomposition is returned for analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

try:  # scipy is a declared dep; fall back to manual JS for safety in slim envs.
    from scipy.spatial.distance import jensenshannon as _scipy_js  # type: ignore
except ImportError:  # pragma: no cover - exercised only in stripped envs
    _scipy_js = None

from src.env.state import State


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and shape parameters (ADR-003 defaults).

    The 7-day baseline window is fixed by THEORY and supplied via the
    ``baseline_7d_volume`` argument to :meth:`RewardFunction.compute` — there
    is no config knob because changing it would invalidate the eq. 15 contract.
    """

    lambda_1: float = 2.0  # overload weight
    lambda_2: float = 1.0  # imbalance weight
    w_progress: float = 0.7  # within-gain progress weight
    w_variety: float = 0.3  # within-gain variety weight
    overload_threshold_mult: float = 1.2
    overload_exponent: float = 1.5
    progress_clip_ceiling: float = 1.2  # max single-step progress delta


def _js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in [0, 1] (base-2). Keys must align."""
    keys = sorted(set(p) | set(q))
    p_vec = np.asarray([p.get(k, 0.0) for k in keys], dtype=float)
    q_vec = np.asarray([q.get(k, 0.0) for k in keys], dtype=float)
    # Normalise to proper distributions; guard against all-zero inputs.
    if p_vec.sum() > 0:
        p_vec = p_vec / p_vec.sum()
    if q_vec.sum() > 0:
        q_vec = q_vec / q_vec.sum()
    if _scipy_js is not None:
        dist = float(_scipy_js(p_vec, q_vec, base=2.0))
        # scipy returns distance (sqrt of divergence); square to get divergence.
        if np.isnan(dist):
            return 0.0
        return float(dist**2)
    # Manual fallback: JS divergence base-2.
    m = 0.5 * (p_vec + q_vec)
    return float(0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


class RewardFunction:
    """Compute eq. 15 reward with full decomposition for analysis."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config if config is not None else RewardConfig()

    def compute(  # noqa: PLR0913 — eq. 15 inputs are fixed by the brief contract
        self,
        state: State,
        next_state: State,
        action_id: int,
        weekly_target: float,
        baseline_7d_volume: float,
        muscle_share_14d: dict[str, float],
        target_muscle_dist: dict[str, float],
    ) -> dict:
        """Eq. 15 reward.

        ``weekly_target`` gates progress to 0 when the trainee has no weekly
        target; ``next_state.weekly_progress`` is assumed already normalised
        upstream (``weekly_so_far / target``).

        ``target_muscle_dist`` is used only by :meth:`_variety` (JS-divergence
        from a goal distribution). The :meth:`_imbalance` penalty looks at
        ``muscle_share_14d`` alone — it punishes any skew regardless of goal.
        """
        cfg = self.config
        progress = self._progress(state, next_state, weekly_target)
        variety = self._variety(muscle_share_14d, target_muscle_dist)
        gain = cfg.w_progress * progress + cfg.w_variety * variety
        overload = self._overload(next_state.rolling_7d_volume, baseline_7d_volume)
        imbalance = self._imbalance(muscle_share_14d)
        reward = gain - cfg.lambda_1 * overload - cfg.lambda_2 * imbalance
        return {
            "reward": float(reward),
            "gain": float(gain),
            "overload": float(overload),
            "imbalance": float(imbalance),
            "progress": float(progress),
            "variety": float(variety),
        }

    # ---- components --------------------------------------------------------

    def _progress(self, state: State, next_state: State, weekly_target: float) -> float:
        if weekly_target <= 0:
            return 0.0
        prev = state.weekly_progress
        # next_state.weekly_progress is already normalised by target upstream;
        # treat it as the "weekly_so_far / target" signal at the new step.
        delta = next_state.weekly_progress - prev
        return float(np.clip(delta, 0.0, self.config.progress_clip_ceiling))

    def _variety(self, share: Mapping[str, float], target: Mapping[str, float]) -> float:
        return float(1.0 - _js_divergence(share, target))

    def _overload(self, rolling_7d: float, baseline: float) -> float:
        cfg = self.config
        if baseline <= 0:
            return 0.0
        threshold = cfg.overload_threshold_mult * baseline
        excess = rolling_7d - threshold
        if excess <= 0:
            return 0.0
        return float((excess / baseline) ** cfg.overload_exponent)

    @staticmethod
    def _imbalance(share: Mapping[str, float]) -> float:
        if not share:
            return 0.0
        values = np.asarray(list(share.values()), dtype=float)
        return float(np.var(values))
