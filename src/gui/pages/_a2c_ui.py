"""Sidebar + theory block helpers for the A2C Streamlit page.

Split out of ``src/gui/pages/05_a2c.py`` so that page stays under the
150-LOC CLAUDE.md §1 limit. Imported by ``05_a2c.py``; no other consumers.
"""

from __future__ import annotations

import streamlit as st

from src.gui.components import info_card, latex_block


def sidebar() -> dict[str, float | int]:
    """Render the A2C hyperparameter sliders and return their values."""
    sb = st.sidebar
    sb.header("A2C hyperparameters")
    episodes = sb.slider(
        "episodes", 5, 200, 50, 5, help="Number of training episodes (synchronous A2C updates)."
    )
    actor_hidden = sb.select_slider(
        "actor_hidden",
        options=[32, 64, 128, 256],
        value=128,
        help="Hidden width of the actor head's 1-layer FC.",
    )
    critic_hidden = sb.select_slider(
        "critic_hidden",
        options=[32, 64, 128, 256],
        value=128,
        help="Hidden width of the critic head's 1-layer FC.",
    )
    actor_lr = sb.slider(
        "actor_lr", 1e-4, 1e-2, 3e-4, 1e-4, format="%.4f", help="Adam learning rate for actor parameters."
    )
    critic_lr = sb.slider(
        "critic_lr", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f", help="Adam learning rate for critic parameters."
    )
    entropy_coef = sb.slider(
        "entropy_coef",
        0.0,
        0.1,
        0.01,
        0.005,
        format="%.3f",
        help="Entropy bonus coefficient (encourages exploration).",
    )
    grad_clip_norm = sb.slider(
        "grad_clip_norm",
        0.1,
        2.0,
        0.5,
        0.1,
        format="%.1f",
        help="Max L2 norm of the gradient before clipping.",
    )
    return {
        "episodes": int(episodes),
        "actor_hidden": int(actor_hidden),
        "critic_hidden": int(critic_hidden),
        "actor_lr": float(actor_lr),
        "critic_lr": float(critic_lr),
        "entropy_coef": float(entropy_coef),
        "grad_clip_norm": float(grad_clip_norm),
    }


def info_block() -> None:
    """Render the §7.5 theory card with eq. (9), eq. (10), and eq. (12)."""
    info_card(
        "A2C - Synchronous Advantage Actor-Critic (brief §7.5)",
        "Two heads share a state encoder: the actor outputs the policy and the "
        "critic outputs a value estimate. Per step we compute the TD error, "
        "push the actor along the policy-gradient direction weighted by it, "
        "and regress the critic onto the bootstrapped TD target.",
    )
    latex_block(
        r"\delta_t \;=\; r_t \;+\; \gamma\,V_\phi(s_{t+1}) \;-\; V_\phi(s_t)",
        label="Eq. (9) - one-step TD error (advantage estimator)",
    )
    latex_block(
        r"\nabla_\theta J(\theta) \;=\; "
        r"\mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\delta_t\right] "
        r"\;+\; \beta\,\nabla_\theta H\!\left[\pi_\theta(\cdot\mid s_t)\right]",
        label="Eq. (10) - actor update with entropy bonus (coef beta)",
    )
    latex_block(
        r"\mathcal{L}_\phi \;=\; \frac{1}{T}\sum_{t=0}^{T}"
        r"\bigl(r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t)\bigr)^{2}",
        label="Eq. (12) - critic MSE against the TD target",
    )


__all__ = ["info_block", "sidebar"]
