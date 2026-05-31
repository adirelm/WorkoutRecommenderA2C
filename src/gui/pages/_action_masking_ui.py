"""Sidebar + info-block + softmax helpers for the action-masking demo page.

Split out of ``src/gui/pages/08_action_masking.py`` to keep that file
under the 150-LOC CLAUDE.md §1 limit. Re-imported by the page only.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.env.action_mask import ActionMaskService
from src.env.state import STATE_CHANNEL_NAMES, State
from src.gui.components import info_card, reference_callout
from src.gui.labels import STATE_CHANNEL_HELP, STATE_CHANNEL_LABEL

_REST_ID = 0
_LEGS_ID = 3
_COND_ID = 5
_REST_STREAK_K = 3
_LEGS_SORENESS_TH = 0.8
_COND_OVERLOAD_TH = 1.2
_INT_CHANNELS = {"streak_days_trained", "days_since_last_rest"}
_SIGNED_CHANNELS = {"muscle_balance_push_vs_pull", "adherence_signal"}


def info_block() -> None:
    """Render the §7.6.1 theory card + Huang & Ontañón citation + ADR link."""
    info_card(
        "Action masking (brief §7.6.1)",
        "Invalid-action masking sets logits of illegal moves to <code>−∞</code> so "
        "<code>softmax</code> assigns them zero probability and the policy gradient "
        "ignores them. Four hard rules (rest-streak, legs-soreness, conditioning "
        "overload, mobility-always) encode trainer-safety constraints the network "
        "should never have to relearn from reward signal alone.",
    )
    reference_callout(
        "Reference — Huang & Ontañón (2022)",
        "<em>A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.</em> "
        "FLAIRS-35. Shows masking is equivalent to a state-dependent policy and that "
        "the unmasked logits' gradient is the correct estimator. "
        "See also <a href='https://github.com/AdirElmakyes/Assignment3-WorkoutRecommenderA2C/"
        "blob/main/docs/adr/ADR-004-action-masking.md'>ADR-004</a>.",
    )


def slider_for(name: str, default: float, reset: bool) -> float:
    """Per-channel slider with the right range - falls back to State.initial() on reset."""
    label = STATE_CHANNEL_LABEL.get(name, name)
    help_text = STATE_CHANNEL_HELP.get(name)
    if name in _INT_CHANNELS:
        return float(st.sidebar.slider(label, 0, 14, 0 if reset else int(default), 1, help=help_text))
    if name in _SIGNED_CHANNELS:
        return st.sidebar.slider(label, -1.0, 1.0, 0.0 if reset else default, 0.05, help=help_text)
    if name == "rolling_7d_volume":
        return st.sidebar.slider(label, 0.0, 2.0, 0.0 if reset else default, 0.05, help=help_text)
    hi = 1.2 if name == "weekly_progress" else 1.0
    base = (1.0 if name == "readiness" else 0.0) if reset else default
    return st.sidebar.slider(label, 0.0, hi, base, 0.05, help=help_text)


def state_sliders() -> State:
    """Render 12 channel sliders (with an Initial-reset toggle) and return a State."""
    base = State.initial()
    if st.sidebar.button("Reset to State.initial()"):
        st.session_state["mask_state_reset"] = True
    reset = st.session_state.pop("mask_state_reset", False)
    st.sidebar.header("12-channel state")
    vals = {n: slider_for(n, float(getattr(base, n)), reset) for n in STATE_CHANNEL_NAMES}
    vals["streak_days_trained"] = int(vals["streak_days_trained"])
    vals["days_since_last_rest"] = int(vals["days_since_last_rest"])
    return State(**vals)  # type: ignore[arg-type]


def rule_toggles() -> dict[str, bool]:
    """Four mask-rule on/off toggles (rule disabled = threshold pushed out of reach)."""
    st.sidebar.header("Mask rules")
    return {
        "rest_streak": st.sidebar.toggle(
            "rest_streak (block Rest after K rests)",
            True,
            help="If the last 3 actions were all Rest, mask Rest (logit → −∞).",
        ),
        "legs_soreness": st.sidebar.toggle(
            "legs_soreness (block Legs if sore)",
            True,
            help="If soreness_legs > 0.8, mask the Legs action (logit → −∞).",
        ),
        "conditioning_overload": st.sidebar.toggle(
            "conditioning_overload (block Conditioning over 1.2x baseline)",
            True,
            help="If rolling_7d_volume > 1.2 × baseline, mask Conditioning (logit → −∞).",
        ),
        "mobility_always": st.sidebar.toggle(
            "mobility_always (Mobility never masked)",
            True,
            help="Mobility is always legal — a safe fallback action that is never masked.",
        ),
    }


def build_service(t: dict[str, bool]) -> ActionMaskService:
    """Disabled rules become unreachable thresholds (still a real ActionMaskService)."""
    return ActionMaskService(
        3 if t["rest_streak"] else 10_000,
        0.8 if t["legs_soreness"] else 10.0,
        1.2 if t["conditioning_overload"] else 1e9,
    )


def softmax(logits: np.ndarray) -> tuple[float, ...]:
    """Numerically stable softmax - masked (−∞) logits collapse to 0 cleanly."""
    finite = np.where(np.isfinite(logits), logits, -np.inf)
    shifted = finite - np.max(finite[np.isfinite(finite)], initial=0.0)
    exp = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    total = float(exp.sum())
    return tuple((exp / total).tolist()) if total > 0 else tuple([0.0] * len(logits))


def rule_reasons(state: State, history: list[int], toggles: dict[str, bool]) -> dict[int, str]:
    """Return ``{action_id: reason}`` for each rule that fires under current inputs."""
    reasons: dict[int, str] = {}
    if (
        toggles["rest_streak"]
        and len(history) >= _REST_STREAK_K
        and all(a == _REST_ID for a in history[-_REST_STREAK_K:])
    ):
        reasons[_REST_ID] = f"rest_streak (last {_REST_STREAK_K} days were all Rest)"
    if toggles["legs_soreness"] and state.soreness_legs > _LEGS_SORENESS_TH:
        reasons[_LEGS_ID] = f"legs_soreness ({state.soreness_legs:.2f} > {_LEGS_SORENESS_TH:.2f})"
    if toggles["conditioning_overload"] and state.rolling_7d_volume > _COND_OVERLOAD_TH:
        reasons[_COND_ID] = f"conditioning_overload ({state.rolling_7d_volume:.2f} > {_COND_OVERLOAD_TH:.2f})"
    return reasons


REST_ID = _REST_ID

__all__ = [
    "REST_ID",
    "build_service",
    "info_block",
    "rule_reasons",
    "rule_toggles",
    "slider_for",
    "softmax",
    "state_sliders",
]
