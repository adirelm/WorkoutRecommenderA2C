"""Running-mean baseline for REINFORCE variance reduction (brief §7.4).

The vanilla REINFORCE gradient
    g_t = ∇_θ log π_θ(a_t | s_t) · G_t
suffers from high variance because G_t is the *full* episodic return.
Subtracting a state-independent baseline b — the running mean of past
episodic returns — leaves the gradient unbiased (E[∇log π · b] = 0)
while shrinking variance toward var(G_t - b). This is the standard
"REINFORCE with baseline" recipe (Sutton & Barto §13.4).

The estimator here is an exponential moving average with rate
``alpha`` (default 0.05 per config.yaml ``reinforce.baseline_alpha``):

    b_{k+1} = (1 - alpha) * b_k + alpha * G_k       (b_0 = 0.0)

``alpha == 0`` disables the baseline (b stays at the initial value)
which makes the trainer falls back to vanilla REINFORCE — useful for
ablations comparing "with vs without baseline".
"""

from __future__ import annotations


# NOTE: We use a state-INDEPENDENT scalar EMA over episodic totals
# (Williams 1992 "reinforcement comparison"). Sutton-Barto §13.4
# prefers a learned V(s_t) for tighter variance reduction — both
# are unbiased; ours is simpler and adequate for a 12-dim, 28-day
# toy environment. See ADR-008-lstm-world-model-simplifications.md
# for the broader "simplifications, justified" philosophy.
class RunningMeanBaseline:
    """Exponential-moving-average estimator of E[G] used by REINFORCE."""

    def __init__(self, alpha: float = 0.05, initial: float = 0.0) -> None:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self._alpha = float(alpha)
        self._value = float(initial)
        self._updates = 0

    @property
    def value(self) -> float:
        """Current baseline estimate (subtracted from G_t inside the loss)."""
        return self._value

    @property
    def alpha(self) -> float:
        """EMA rate; alpha=0 disables the baseline (ablation lever)."""
        return self._alpha

    @property
    def updates(self) -> int:
        """Number of update() calls so far — useful for tests."""
        return self._updates

    def update(self, total_return: float) -> float:
        """EMA update; returns the *new* baseline value.

        ``alpha == 0`` is a no-op (baseline stays frozen at its initial value),
        which is the documented way to disable the baseline for ablations.
        """
        if self._alpha > 0.0:
            self._value = (1.0 - self._alpha) * self._value + self._alpha * float(total_return)
        self._updates += 1
        return self._value

    def reset(self, initial: float = 0.0) -> None:
        """Reset the estimator — used between independent training runs."""
        self._value = float(initial)
        self._updates = 0
