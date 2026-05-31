"""Per-rollout / per-step diagnostic charts (heatmap, action bars, reward decomposition).

Split from :mod:`src.gui.charts` to keep both modules under the 150-LOC limit.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from src.env.state import STATE_CHANNEL_NAMES, State
from src.gui.chart_layout import fig
from src.gui.theme import THEME


def muscle_distribution_heatmap(trajectory: list[State]) -> go.Figure:
    """Heatmap of state-channel values across the rollout (day x channel)."""
    out = fig("Trainee state across the rollout")
    if not trajectory:
        return out
    z = [list(s.to_array()) for s in trajectory]
    out.add_trace(
        go.Heatmap(
            z=z,
            x=list(STATE_CHANNEL_NAMES),
            y=[f"d{i + 1}" for i in range(len(trajectory))],
            colorscale=[[0, "#FFFFFF"], [1, THEME.primary]],
            colorbar={"title": "value"},
        )
    )
    out.update_xaxes(title="state channel", tickangle=-30)
    out.update_yaxes(title="day")
    return out


def action_probability_bar(
    probs: tuple[float, ...],
    action_names: list[str],
    mask: list[bool] | None = None,
    title: str = "Policy action probabilities",
) -> go.Figure:
    """Per-action probability bar; masked actions rendered in the warning colour."""
    out = fig(title)
    colors = [
        THEME.warning if (mask is not None and i < len(mask) and not mask[i]) else THEME.primary
        for i in range(len(probs))
    ]
    out.add_trace(
        go.Bar(
            x=list(action_names),
            y=list(probs),
            marker_color=colors,
            text=[f"{p:.2f}" for p in probs],
            textposition="outside",
        )
    )
    out.update_yaxes(title="P(action)", range=[0, max([*list(probs), 1.0])])
    out.update_xaxes(title="action")
    return out


def reward_decomposition_bar(decomposition: dict[str, float]) -> go.Figure:
    """Gain / overload / imbalance reward breakdown."""
    out = fig("Reward decomposition")
    if not decomposition:
        return out
    keys = list(decomposition.keys())
    vals = [float(decomposition[k]) for k in keys]
    colors: list[Any] = [THEME.success if v >= 0 else THEME.danger for v in vals]
    out.add_trace(
        go.Bar(x=keys, y=vals, marker_color=colors, text=[f"{v:+.2f}" for v in vals], textposition="outside")
    )
    out.update_xaxes(title="component")
    out.update_yaxes(title="contribution")
    return out


__all__ = ["action_probability_bar", "muscle_distribution_heatmap", "reward_decomposition_bar"]
