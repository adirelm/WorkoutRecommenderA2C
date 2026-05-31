"""Generate a synthetic-trainee transition trajectory for LSTM training (brief §7.3).

Bridge between Phase-1 daily aggregates and Phase-2 WorkoutEnv: roll the env
under a fixed baseline policy for `num_days` steps to produce the
list[(state, action, next_state)] trajectory the LSTM world model consumes
(via src/model/dataset.build_windows). Two baseline policies are exposed:

  * ``uniform_random_masked`` — uniform over the legal actions at each step.
  * ``rotate_chains``         — deterministic PPL-style cycle, with Mobility
                                forced on streak-day-7 to break overload.

Both policies are seedable; ``set_global_seed`` is called up front so that
runs with the same seed produce bit-identical trajectories.
"""

from __future__ import annotations

import numpy as np

from src.env.state import State
from src.env.workout_env import WorkoutEnv
from src.utils.seeding import set_global_seed

# Deterministic PPL-style cycle. Order chosen so a 7-day window covers every
# action exactly once: Rest, Push, Pull, Legs, FullBody, Conditioning, Mobility.
# Index 6 (Mobility) is the streak-break slot; the loop forces it when
# streak_days_trained reaches 7 (see _rotate_chains_action).
_ROTATE_CYCLE: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
_MOBILITY_ID: int = 6
_STREAK_BREAK_THRESHOLD: int = 7

_VALID_POLICIES: frozenset[str] = frozenset({"uniform_random_masked", "rotate_chains"})


def generate_trajectory(
    num_days: int = 28,
    seed: int = 42,
    action_policy: str = "uniform_random_masked",
) -> list[tuple[State, int, State]]:
    """Roll out WorkoutEnv for ``num_days`` steps under a baseline policy.

    Args:
        num_days: Number of env steps to take (== trajectory length).
        seed: Forwarded to ``set_global_seed`` and ``WorkoutEnv``; identical
            seeds produce identical trajectories.
        action_policy: ``"uniform_random_masked"`` samples uniformly from the
            current legal-action mask; ``"rotate_chains"`` follows a fixed
            7-day PPL-Mobility cycle.

    Returns:
        List of (state, action_id, next_state) triples, length = num_days.

    Raises:
        ValueError: If ``action_policy`` is not one of the supported names.
    """
    if action_policy not in _VALID_POLICIES:
        raise ValueError(f"action_policy must be one of {sorted(_VALID_POLICIES)}, got {action_policy!r}")
    if num_days <= 0:
        return []

    set_global_seed(seed)
    # Local RNG so policy sampling is independent of any global state the
    # env's internal RNG drains during step().
    rng = np.random.default_rng(seed)
    env = WorkoutEnv(seed=seed)
    state = env.reset()
    trajectory: list[tuple[State, int, State]] = []

    for day_index in range(num_days):
        if action_policy == "uniform_random_masked":
            action = _uniform_random_masked_action(env, rng)
        else:  # action_policy == "rotate_chains"
            action = _rotate_chains_action(day_index, state)
        next_state, _reward, _done, _info = env.step(action)
        trajectory.append((state, action, next_state))
        state = next_state

    return trajectory


def _uniform_random_masked_action(env: WorkoutEnv, rng: np.random.Generator) -> int:
    """Sample uniformly from the env's current legal-action set."""
    mask = env.action_mask()
    legal_ids = np.flatnonzero(mask)
    if legal_ids.size == 0:
        # Defensive: action_mask guarantees Mobility (id=6) is always legal,
        # so this branch should be unreachable. Fall back to Mobility.
        return _MOBILITY_ID
    return int(rng.choice(legal_ids))


def _rotate_chains_action(day_index: int, state: State) -> int:
    """Deterministic PPL cycle. Force Mobility on streak day 7 to break overload."""
    if state.streak_days_trained >= _STREAK_BREAK_THRESHOLD:
        return _MOBILITY_ID
    return _ROTATE_CYCLE[day_index % len(_ROTATE_CYCLE)]
