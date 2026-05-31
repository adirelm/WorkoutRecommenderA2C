"""Shared Plotly layout + figure factory used by every chart in :mod:`src.gui.charts`."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from src.gui.theme import THEME

LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12, "color": "#1A202C"},
    "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
    "paper_bgcolor": "white",
    "plot_bgcolor": THEME.light_bg,
}


def fig(title: str) -> go.Figure:
    """Return a new Figure with the Bar-Ilan title style + shared layout applied."""
    out = go.Figure()
    out.update_layout(title={"text": title, "font": {"color": THEME.primary, "size": 16}}, **LAYOUT)
    return out


__all__ = ["LAYOUT", "fig"]
