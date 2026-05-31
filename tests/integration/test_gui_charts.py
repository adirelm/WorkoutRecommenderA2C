"""Structural tests for src/gui/charts.py (§9 Plotly factories).

Each factory must return a ``plotly.graph_objects.Figure`` with the
expected trace count, title text and axis labels. The fixtures are
hand-built (no SDK call) so the suite stays deterministic and fast —
training a real A2C run just to chart it would be wasteful.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.env.state import ACTION_NAMES, State
from src.gui import charts
from src.model.types import LSTMTrainHistory
from src.services.a2c_types import A2CHistory
from src.services.comparator import ComparisonResult


def _lstm_history() -> LSTMTrainHistory:
    return LSTMTrainHistory(
        epochs_run=4,
        train_loss=(0.9, 0.7, 0.5, 0.4),
        val_loss=(1.0, 0.8, 0.6, 0.55),
        best_epoch=2,
    )


def _a2c_history() -> A2CHistory:
    return A2CHistory(
        episodes_run=3,
        rewards=(1.0, 1.5, 2.0),
        actor_losses=(0.3, 0.25, 0.2),
        critic_losses=(0.4, 0.35, 0.3),
        advantages_mean=(0.0, 0.1, 0.05),
        seed=42,
    )


def _comparison() -> ComparisonResult:
    n = 5
    return ComparisonResult(
        reinforce_mean_reward=np.linspace(0.0, 1.0, n, dtype=np.float32),
        reinforce_std_reward=np.full(n, 0.1, dtype=np.float32),
        a2c_mean_reward=np.linspace(0.2, 1.4, n, dtype=np.float32),
        a2c_std_reward=np.full(n, 0.15, dtype=np.float32),
        episode_count=n,
        seed_count=3,
    )


def test_loss_curve_has_train_val_traces_and_best_marker() -> None:
    fig = charts.loss_curve(_lstm_history())
    assert isinstance(fig, go.Figure)
    names = [t.name for t in fig.data]
    assert names == ["train", "val"]
    assert "LSTM training loss" in fig.layout.title.text
    assert any(getattr(s, "type", None) == "line" for s in fig.layout.shapes or [])


def test_reward_curve_single_trace_and_axes() -> None:
    fig = charts.reward_curve((0.1, 0.5, 1.2, 1.8))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "reward"
    assert fig.layout.xaxis.title.text == "episode"
    assert fig.layout.yaxis.title.text == "total reward"


def test_actor_critic_panels_has_three_subplots() -> None:
    fig = charts.actor_critic_panels(_a2c_history())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert {t.name for t in fig.data} == {"actor", "critic", "reward"}
    assert "A2C training diagnostics" in fig.layout.title.text


def test_comparison_band_has_band_and_mean_per_method() -> None:
    fig = charts.comparison_band(_comparison())
    assert isinstance(fig, go.Figure)
    # 2 methods x (band fill + mean line) = 4 traces
    assert len(fig.data) == 4
    line_names = [t.name for t in fig.data if t.fill != "toself"]
    assert sorted(line_names) == ["A2C", "REINFORCE"]
    assert "n=3 seeds" in fig.layout.title.text


def test_muscle_distribution_heatmap_empty_and_populated() -> None:
    empty = charts.muscle_distribution_heatmap([])
    assert isinstance(empty, go.Figure)
    assert len(empty.data) == 0

    fig = charts.muscle_distribution_heatmap([State.initial(), State.initial()])
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].y) == ["d1", "d2"]


def test_action_probability_bar_marks_masked_actions() -> None:
    probs = tuple(1.0 / len(ACTION_NAMES) for _ in ACTION_NAMES)
    mask = [True] * len(ACTION_NAMES)
    mask[0] = False  # Rest is masked
    fig = charts.action_probability_bar(probs, list(ACTION_NAMES), mask=mask)
    assert len(fig.data) == 1
    colors = list(fig.data[0].marker.color)
    assert colors[0] != colors[1]  # masked colour differs from primary
    assert list(fig.data[0].x) == list(ACTION_NAMES)


def test_reward_decomposition_bar_empty_and_signed() -> None:
    empty = charts.reward_decomposition_bar({})
    assert isinstance(empty, go.Figure)
    assert len(empty.data) == 0

    fig = charts.reward_decomposition_bar({"gain": 1.2, "overload": -0.4, "imbalance": -0.1})
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert list(bar.x) == ["gain", "overload", "imbalance"]
    assert list(bar.y) == [1.2, -0.4, -0.1]
