"""§7.3/§7.6 helpers that adapt SDK outputs for the Streamlit GUI.

Every function here is a *pure* adapter: it takes SDK-shaped inputs
(``WorkoutSDK``, ``REINFORCEHistory``, etc.) and returns plain Python
dicts/lists ready for ``st.metric`` / Plotly / Altair. Pages never
massage SDK objects directly — they call into this module so the
business-logic facade contract from CLAUDE.md §3 holds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src.env.state import ACTION_NAMES, State
from src.sdk.sdk import WorkoutSDK


def get_initial_trajectory(sdk: WorkoutSDK, days: int = 28) -> list[State]:
    """Roll a fresh env forward with the masked-uniform policy for ``days`` steps.

    Used by the Data & Environment page to draw the 12-channel state
    sparkline *before* any policy is trained. We sample uniformly over
    the current legal mask so the trajectory respects ADR-004 rules
    without depending on REINFORCE / A2C being initialised.
    """
    if days <= 0:
        raise ValueError(f"days must be > 0, got {days}")
    # _ensure_env is private but prepare_data is the documented public entry.
    sdk.prepare_data()
    env = sdk._ensure_env()
    state = env.reset()
    trajectory: list[State] = [state]
    rng = np.random.default_rng(sdk.seed)
    for _ in range(days):
        mask = env.action_mask()
        legal = np.flatnonzero(mask > 0)
        if legal.size == 0:
            break
        action = int(rng.choice(legal))
        state, _reward, done, _info = env.step(action)
        trajectory.append(state)
        if done:
            break
    return trajectory


def compute_action_histogram(
    rewards_actions_pairs: Iterable[tuple[float, int]],
) -> dict[str, int]:
    """Aggregate ``(reward, action_id)`` pairs into ``{action_name: count}``.

    Used by the Recommendation + Comparison pages to show *what the
    policy actually picked*, not just average reward. Unknown action
    ids are silently dropped — a malformed trace shouldn't crash the
    GUI mid-rerun.
    """
    counts: dict[str, int] = dict.fromkeys(ACTION_NAMES, 0)
    for _reward, action_id in rewards_actions_pairs:
        if 0 <= action_id < len(ACTION_NAMES):
            counts[ACTION_NAMES[action_id]] += 1
    return counts


def format_history_summary(history: Any) -> dict[str, str]:
    """Project a REINFORCE / A2C history dataclass into a ``metric_row`` dict.

    Tolerates both ``REINFORCEHistory`` and ``A2CHistory`` (and any
    future dataclass exposing ``rewards`` + ``episodes_run``). Values
    are formatted strings — the caller is a ``st.metric`` row, not a
    chart, so we don't return raw floats.
    """
    rewards = getattr(history, "rewards", ())
    episodes = int(getattr(history, "episodes_run", len(rewards)))
    if not rewards:
        return {"Episodes": str(episodes), "Final reward": "—", "Mean reward": "—"}
    final = float(rewards[-1])
    mean = float(sum(rewards) / len(rewards))
    best = float(max(rewards))
    summary = {
        "Episodes": str(episodes),
        "Final reward": f"{final:+.2f}",
        "Mean reward": f"{mean:+.2f}",
        "Best episode": f"{best:+.2f}",
    }
    # A2C-only diagnostics: surface mean advantage if present.
    adv = getattr(history, "advantages_mean", None)
    if adv:
        summary["Mean advantage"] = f"{float(sum(adv) / len(adv)):+.3f}"
    return summary
