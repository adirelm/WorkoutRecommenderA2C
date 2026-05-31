"""Tests for src/services/types.py — REINFORCE config defaults match config.yaml + frozen invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from src.services.types import EpisodeResult, REINFORCEConfig, REINFORCEHistory

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"


def test_reinforce_config_defaults_match_config_yaml():
    cfg = yaml.safe_load(_CONFIG_PATH.read_text())
    rcfg = cfg["reinforce"]
    defaults = REINFORCEConfig()
    assert defaults.policy_hidden == rcfg["policy_hidden"] == 128
    assert defaults.lr == float(rcfg["lr"]) == 3.0e-4
    assert defaults.gamma == rcfg["gamma"] == 0.99
    assert defaults.episodes == rcfg["episodes"] == 500
    assert defaults.baseline_alpha == rcfg["baseline_alpha"] == 0.05
    assert defaults.entropy_coef == rcfg["entropy_coef"] == 0.0


def test_reinforce_history_frozen():
    hist = REINFORCEHistory(
        episodes_run=2,
        rewards=(1.0, 2.0),
        losses=(0.5, 0.3),
        baseline=(0.0, 1.0),
        seed=42,
    )
    assert hist.episodes_run == 2
    assert hist.rewards == (1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        hist.seed = 7  # type: ignore[misc]


def test_episode_result_immutable():
    ep = EpisodeResult(
        episode=0,
        total_reward=3.5,
        length=28,
        actions=(0, 1, 2),
        states=(),
        rewards=(1.0, 1.5, 1.0),
        loss=0.42,
    )
    assert ep.total_reward == 3.5
    assert ep.length == 28
    with pytest.raises(FrozenInstanceError):
        ep.loss = 0.0  # type: ignore[misc]
