"""Sidebar + theory block helpers for the REINFORCE Streamlit page.

Split out of ``src/gui/pages/04_reinforce.py`` to keep that page file
under the 150-LOC CLAUDE.md §1 limit. The page imports these two
helpers and stays a thin orchestrator.
"""

from __future__ import annotations

import streamlit as st

from src.gui.components import info_card, latex_block


def sidebar() -> dict[str, float | int | bool]:
    """Render the REINFORCE hyperparameter sliders and return their values."""
    st.sidebar.header("REINFORCE hyperparameters")
    episodes = st.sidebar.slider(
        "Training episodes",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
        help="Number of Monte-Carlo policy-gradient updates (one update per episode).",
    )
    policy_hidden = st.sidebar.select_slider(
        "Policy hidden units",
        options=[32, 64, 128, 256],
        value=128,
        help="Width of the 1-layer FC policy network.",
    )
    lr = st.sidebar.slider(
        "Learning rate (Adam)",
        min_value=1e-4,
        max_value=1e-2,
        value=3e-4,
        step=1e-4,
        format="%.4f",
        help="Step size for Adam. Default 3e-4.",
    )
    baseline_alpha = st.sidebar.slider(
        "Baseline EMA coefficient",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Exponential-moving-average smoothing for the running-mean baseline b (eq. 4).",
    )
    gamma = st.sidebar.slider(
        "Discount factor γ",
        min_value=0.9,
        max_value=0.999,
        value=0.99,
        step=0.001,
        format="%.3f",
        help="Reward discount in G_t = Σ γ^k r_{t+k}.",
    )
    show_progress = st.sidebar.toggle(
        "Show progress",
        value=True,
        help="Show live reward curve during training (off = only show final chart).",
    )
    return {
        "episodes": int(episodes),
        "policy_hidden": int(policy_hidden),
        "lr": float(lr),
        "baseline_alpha": float(baseline_alpha),
        "gamma": float(gamma),
        "show_progress": bool(show_progress),
    }


def info_block() -> None:
    """Render the §7.4 theory card with eq. (4) and eq. (16)."""
    info_card(
        "REINFORCE (brief §7.4)",
        "Monte-Carlo policy gradient with a running-mean baseline b. "
        "One full episode is rolled out under policy, returns G_t are "
        "computed, the baseline is updated, and a single Adam step is taken.",
    )
    latex_block(
        r"\nabla_\theta J(\theta) \;=\; "
        r"\mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} "
        r"\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,(G_t - b)\right]",
        label="Eq. (4) - policy-gradient theorem with baseline",
    )
    latex_block(
        r"\mathcal{L}(\theta) \;=\; -\,\frac{1}{T}\sum_{t=0}^{T} "
        r"\log \pi_\theta(a_t \mid s_t)\,(G_t - b)",
        label="Eq. (16) - REINFORCE surrogate loss minimised by Adam",
    )


__all__ = ["info_block", "sidebar"]
