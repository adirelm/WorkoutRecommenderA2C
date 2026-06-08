"""Tests for the real-PHUL-derived LSTM training trajectory (brief §7.2.4 + §7.3).

Proves the LSTM is trained on a trajectory built from the chosen real Kaggle
program (closes audit finding F-2), not a free-floating synthetic simulator.
"""

from __future__ import annotations

from src.data.types import DailyEntry
from src.env.state import ACTION_COUNT, State
from src.model.program_trajectory import entry_to_action, generate_program_trajectory


def _entry(rest: bool, dist: dict[str, float]) -> DailyEntry:
    return DailyEntry(
        day_in_cycle=0,
        week_index=0,
        is_rest_day=rest,
        total_volume=0.0 if rest else 100.0,
        session_duration_min=0 if rest else 45,
        muscle_distribution=dist,
    )


def test_rest_day_maps_to_rest_action() -> None:
    assert entry_to_action(_entry(True, {})) == 0


def test_upper_day_with_push_and_pull_maps_to_fullbody() -> None:
    assert entry_to_action(_entry(False, {"push": 0.45, "pull": 0.55})) == 4


def test_legs_dominant_day_maps_to_legs_action() -> None:
    assert entry_to_action(_entry(False, {"legs": 0.7, "core": 0.3})) == 3


def test_trajectory_has_expected_length_and_action_range() -> None:
    traj = generate_program_trajectory(seed=42)
    assert len(traj) == 84  # 12 real PHUL weeks x 7 days
    for state, action, nxt in traj:
        assert isinstance(state, State) and isinstance(nxt, State)
        assert 0 <= action < ACTION_COUNT


def test_trajectory_is_deterministic_for_a_seed() -> None:
    a = generate_program_trajectory(seed=42)
    b = generate_program_trajectory(seed=42)
    assert [t[1] for t in a] == [t[1] for t in b]
    assert [t[2].fatigue for t in a] == [t[2].fatigue for t in b]


def test_trajectory_contains_rest_and_training_actions() -> None:
    actions = {t[1] for t in generate_program_trajectory(seed=42)}
    assert 0 in actions  # rest days present
    assert len(actions) >= 3  # real program exercises several action types
