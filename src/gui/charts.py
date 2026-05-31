"""§9 Plotly chart factories (Bar-Ilan palette, used by every GUI page).

Training-curve helpers (LSTM loss, REINFORCE reward, A2C panels, comparison band)
live here. Per-rollout diagnostics (heatmap, action bars, reward decomposition)
live in :mod:`src.gui.charts_diag` and are re-exported below to preserve the
single public-API entry point promised by CLAUDE.md §3.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.gui.chart_layout import fig
from src.gui.charts_diag import (
    action_probability_bar,
    muscle_distribution_heatmap,
    reward_decomposition_bar,
)
from src.gui.theme import THEME
from src.model.types import LSTMTrainHistory
from src.services.a2c_types import A2CHistory
from src.services.comparator import ComparisonResult


def loss_curve(history: LSTMTrainHistory, title: str = "LSTM training loss") -> go.Figure:
    """Train + val MSE per epoch, with a best-epoch marker."""
    out = fig(title)
    epochs = list(range(1, history.epochs_run + 1))
    out.add_trace(
        go.Scatter(
            x=epochs, y=list(history.train_loss), name="train", line={"color": THEME.primary, "width": 2}
        )
    )
    out.add_trace(
        go.Scatter(
            x=epochs,
            y=list(history.val_loss),
            name="val",
            line={"color": THEME.accent, "width": 2, "dash": "dash"},
        )
    )
    if history.val_loss:
        be = history.best_epoch + 1
        out.add_vline(
            x=be,
            line_dash="dot",
            line_color=THEME.success,
            annotation_text=f"best @ {be}",
            annotation_position="top",
        )
    out.update_xaxes(title="epoch")
    out.update_yaxes(title="MSE loss")
    return out


def reward_curve(rewards: tuple[float, ...], title: str = "Episode reward") -> go.Figure:
    """Per-episode reward line."""
    out = fig(title)
    out.add_trace(
        go.Scatter(
            x=list(range(1, len(rewards) + 1)),
            y=list(rewards),
            name="reward",
            line={"color": THEME.primary, "width": 2},
        )
    )
    out.update_xaxes(title="episode")
    out.update_yaxes(title="total reward")
    return out


def actor_critic_panels(history: A2CHistory) -> go.Figure:
    """3-panel A2C diagnostics: actor loss / critic loss / episode reward."""
    out = make_subplots(rows=1, cols=3, subplot_titles=("Actor loss", "Critic loss", "Episode reward"))
    x = list(range(1, history.episodes_run + 1))
    out.add_trace(
        go.Scatter(x=x, y=list(history.actor_losses), name="actor", line={"color": THEME.primary}),
        row=1,
        col=1,
    )
    out.add_trace(
        go.Scatter(x=x, y=list(history.critic_losses), name="critic", line={"color": THEME.accent}),
        row=1,
        col=2,
    )
    out.add_trace(
        go.Scatter(x=x, y=list(history.rewards), name="reward", line={"color": THEME.success}), row=1, col=3
    )
    for c in (1, 2, 3):
        out.update_xaxes(title_text="episode", row=1, col=c)
    out.update_layout(
        template="plotly_white",
        showlegend=False,
        title={"text": "A2C training diagnostics", "font": {"color": THEME.primary, "size": 16}},
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        height=380,
    )
    return out


def comparison_band(result: ComparisonResult) -> go.Figure:
    """Mean ± std reward bands: REINFORCE vs A2C."""
    out = fig(f"REINFORCE vs A2C (n={result.seed_count} seeds)")
    x = list(range(1, result.episode_count + 1))
    for name, mean, std, color in (
        ("REINFORCE", result.reinforce_mean_reward, result.reinforce_std_reward, THEME.primary),
        ("A2C", result.a2c_mean_reward, result.a2c_std_reward, THEME.accent),
    ):
        upper = (mean + std).tolist()
        lower = (mean - std).tolist()
        out.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.18,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} ±std",
            )
        )
        out.add_trace(go.Scatter(x=x, y=mean.tolist(), name=name, line={"color": color, "width": 2.5}))
    out.update_xaxes(title="episode")
    out.update_yaxes(title="mean reward")
    return out


__all__ = [
    "action_probability_bar",
    "actor_critic_panels",
    "comparison_band",
    "loss_curve",
    "muscle_distribution_heatmap",
    "reward_curve",
    "reward_decomposition_bar",
]
