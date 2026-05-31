"""Tests for src/services/a2c_types.py — A2C config defaults match config.yaml + frozen invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from src.services.a2c_types import A2CConfig, A2CHistory

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def test_a2c_config_defaults_match_config_yaml():
    cfg = yaml.safe_load(_CONFIG_PATH.read_text())
    acfg = cfg["a2c"]
    defaults = A2CConfig()
    assert defaults.actor_hidden == acfg["actor_hidden"] == 128
    assert defaults.critic_hidden == acfg["critic_hidden"] == 128
    assert defaults.gamma == acfg["gamma"] == 0.99
    assert defaults.entropy_coef == acfg["entropy_coef"] == 0.01


def test_a2c_history_frozen():
    hist = A2CHistory(
        episodes_run=2,
        rewards=(1.0, 2.0),
        actor_losses=(0.5, 0.3),
        critic_losses=(0.4, 0.2),
        advantages_mean=(0.1, -0.05),
        seed=42,
    )
    assert hist.episodes_run == 2
    assert hist.rewards == (1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        hist.seed = 7  # type: ignore[misc]
