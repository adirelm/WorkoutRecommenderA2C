"""Tests for src/services/comparator.py."""

from __future__ import annotations

import pytest

from src.services.comparator import ComparisonResult, compare


class _StubHistory:
    def __init__(self, rewards: tuple[float, ...]) -> None:
        self.rewards = rewards


def test_compare_returns_correct_shapes():
    r_hist = [_StubHistory((1.0, 2.0, 3.0)), _StubHistory((2.0, 2.0, 4.0))]
    a_hist = [_StubHistory((5.0, 4.0, 3.0)), _StubHistory((6.0, 4.0, 2.0))]
    result = compare(r_hist, a_hist)
    assert isinstance(result, ComparisonResult)
    assert result.episode_count == 3
    assert result.seed_count == 2
    assert result.reinforce_mean_reward.shape == (3,)
    assert result.a2c_mean_reward.shape == (3,)


def test_compare_mean_matches_per_seed_mean():
    r_hist = [_StubHistory((1.0, 2.0)), _StubHistory((3.0, 4.0))]
    a_hist = [_StubHistory((10.0, 20.0)), _StubHistory((30.0, 40.0))]
    result = compare(r_hist, a_hist)
    assert result.reinforce_mean_reward[0] == 2.0
    assert result.reinforce_mean_reward[1] == 3.0
    assert result.a2c_mean_reward[0] == 20.0
    assert result.a2c_mean_reward[1] == 30.0


def test_compare_rejects_seed_count_mismatch():
    r_hist = [_StubHistory((1.0,)), _StubHistory((2.0,))]
    a_hist = [_StubHistory((1.0,))]
    with pytest.raises(ValueError, match="seed-count mismatch"):
        compare(r_hist, a_hist)


def test_compare_rejects_episode_count_mismatch():
    r_hist = [_StubHistory((1.0, 2.0))]
    a_hist = [_StubHistory((1.0,))]
    with pytest.raises(ValueError, match="episode-count mismatch"):
        compare(r_hist, a_hist)


def test_compare_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        compare([], [_StubHistory((1.0,))])
