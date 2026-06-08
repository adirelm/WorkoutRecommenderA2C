"""Build the LSTM training trajectory from the chosen real Kaggle program.

This is the bridge that connects the data layer (``src/data``) to the world
model (closes audit finding F-2). The chosen PHUL program's daily sessions are
mapped to the env's discrete action space and prescribed volumes, then a
*deterministic* SyntheticTrainee (noise sigma=0) is rolled along that real-program
schedule to emit the ``list[(State, action, next_state)]`` the LSTM consumes
via :func:`src.model.dataset.build_windows`.

The trainee still supplies the *physiological* response (fatigue / soreness /
readiness) — the brief (§7.6 Q4) is explicit that the program data carries no
biological outcomes — but the action sequence and per-day volumes now come from
the real Kaggle program rather than a free-floating simulator.
"""

from __future__ import annotations

import numpy as np

from src.data.aggregator import daily_aggregate, insert_rest_days
from src.data.program_loader import load_chosen_program
from src.data.types import DailyEntry
from src.env.state import State
from src.env.synthetic_trainee import SyntheticTrainee
from src.utils.config_loader import load_config

# Action ids (mirror src.env.state ACTION_NAMES order):
# 0 Rest, 1 Push, 2 Pull, 3 Legs, 4 FullBody, 5 Conditioning, 6 Mobility.
_DOMINANT_ACTION: dict[str, int] = {
    "push": 1,
    "pull": 2,
    "legs": 3,
    "core": 4,  # core is trained compound → folded into FullBody
    "cardio": 5,
    "mobility": 6,
}
# Default min push & pull share for a day to count as a PHUL "upper" day → FullBody.
# Config-driven via dataset.upper_day_share; this constant is the fallback default.
_UPPER_DAY_SHARE: float = 0.25


def entry_to_action(entry: DailyEntry, upper_day_share: float = _UPPER_DAY_SHARE) -> int:
    """Map one aggregated daily session to a discrete env action id."""
    if entry.is_rest_day:
        return 0
    dist = entry.muscle_distribution
    if dist.get("push", 0.0) >= upper_day_share and dist.get("pull", 0.0) >= upper_day_share:
        return 4
    dominant = max(dist, key=lambda k: dist[k]) if dist else "push"
    return _DOMINANT_ACTION.get(dominant, 4)


def generate_program_trajectory(
    config: dict | None = None,
    num_days: int | None = None,
    seed: int = 42,
) -> list[tuple[State, int, State]]:
    """Roll a deterministic trainee along the real PHUL daily schedule.

    Args:
        config: Parsed config dict (loaded from disk when ``None``).
        num_days: Trajectory length; defaults to ``dataset.program_cycle_days``.
        seed: Trainee RNG seed (sigma=0, so transitions are deterministic anyway).

    Returns:
        ``list[(state, action_id, next_state)]`` of length ``num_days``.
    """
    cfg = config if config is not None else load_config()
    cycle_days = int(num_days if num_days is not None else cfg["dataset"]["program_cycle_days"])
    scale = float(cfg["dataset"]["program_volume_scale"])
    upper_share = float(cfg["dataset"]["upper_day_share"])

    _name, cleaned = load_chosen_program(cfg)
    entries = insert_rest_days(daily_aggregate(cleaned), cycle_days)

    trainee = SyntheticTrainee(rng=np.random.default_rng(seed), noise_sigma=0.0)
    state = State.initial()
    trajectory: list[tuple[State, int, State]] = []
    for entry in entries:
        action = entry_to_action(entry, upper_share)
        next_state = trainee.next_state(
            state,
            action_id=action,
            prescribed_volume=entry.total_volume * scale,
            prescribed_muscles=entry.muscle_distribution,
        )
        trajectory.append((state, action, next_state))
        state = next_state
    return trajectory
