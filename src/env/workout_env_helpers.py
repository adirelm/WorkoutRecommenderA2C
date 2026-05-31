"""Pure helpers for WorkoutEnv — extracted to keep workout_env.py ≤150 raw lines.

Holds the (volume, muscle-group) action table, the group→share-key projection,
and stateless reducers for the 14-day muscle-volume share dict. No class state,
no imports from workout_env.py — strictly one-way dependency.
"""

from __future__ import annotations

# Per-action (prescribed volume, dominant muscle bucket) for share tracking.
_ACTIONS: tuple[tuple[float, str], ...] = (
    (0.0, "rest"),
    (10.0, "push"),
    (10.0, "pull"),
    (12.0, "legs"),
    (14.0, "full"),
    (6.0, "cardio"),
    (4.0, "mobility"),
)
ACTION_VOLUME: dict[int, float] = {i: v for i, (v, _) in enumerate(_ACTIONS)}
ACTION_GROUP: dict[int, str] = {i: g for i, (_, g) in enumerate(_ACTIONS)}
GROUP_TO_SHARE_KEY: dict[str, str] = {
    "push": "push",
    "pull": "pull",
    "legs": "legs",
    "core": "core",
    "full": "push",
    "cardio": "core",
    "mobility": "core",
    "rest": "core",
}


def zero_share() -> dict[str, float]:
    """Return a fresh zero-initialised muscle-share accumulator."""
    return {"push": 0.0, "pull": 0.0, "legs": 0.0, "core": 0.0}


def update_muscle_share(
    share: dict[str, float], group: str, volume_delta: float
) -> None:
    """Mutate ``share`` in place: add ``volume_delta`` to the bucket of ``group``."""
    key = GROUP_TO_SHARE_KEY.get(group, "core")
    share[key] = share.get(key, 0.0) + float(volume_delta)


def normalised_share(share: dict[str, float]) -> dict[str, float]:
    """Return share normalised to sum 1.0; uniform 0.25 fallback when total≤0."""
    total = sum(share.values())
    if total <= 0:
        return dict.fromkeys(share, 0.25)
    return {k: v / total for k, v in share.items()}


def build_step_info(
    decomp: dict[str, float],
    volume_delta: float,
    group: str,
    step_count: int,
) -> dict:
    """Assemble the per-step info dict surfaced through env.step()."""
    return {
        "gain": decomp["gain"],
        "overload": decomp["overload"],
        "imbalance": decomp["imbalance"],
        "progress": decomp.get("progress", 0.0),
        "variety": decomp.get("variety", 0.0),
        "volume_delta": float(volume_delta),
        "muscle_group": group,
        "step": step_count,
    }
