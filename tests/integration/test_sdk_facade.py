"""Tests for the WorkoutSDK facade (Phase 6 / PRD §6, CLAUDE.md §3)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from src.env.state import ACTION_COUNT, STATE_DIM, State
from src.sdk.sdk import WorkoutSDK
from src.sdk.types import (
    LogbookHandle,
    PolicyHandle,
    WorkoutRecommendation,
)
from src.services.a2c_types import A2CHistory
from src.services.comparator import ComparisonResult
from src.services.types import REINFORCEHistory


# ---------------------------------------------------------------- ctor + data
def test_sdk_init_with_seed() -> None:
    sdk = WorkoutSDK(seed=7)
    assert sdk.seed == 7


def test_prepare_data_returns_logbook_handle() -> None:
    sdk = WorkoutSDK(seed=42)
    handle = sdk.prepare_data()
    assert isinstance(handle, LogbookHandle)
    assert handle.state_dim == STATE_DIM
    assert handle.n_days > 0
    assert handle.program_name


# --------------------------------------------------------------- trainers
def test_train_reinforce_returns_handle_and_history() -> None:
    sdk = WorkoutSDK(seed=42)
    handle, history = sdk.train_reinforce(episodes=2)
    assert isinstance(handle, PolicyHandle)
    assert handle.algorithm == "REINFORCE"
    assert handle.episodes_trained == 2
    assert isinstance(history, REINFORCEHistory)
    assert history.episodes_run == 2
    assert len(history.rewards) == 2
    assert all(math.isfinite(r) for r in history.rewards)


def test_train_a2c_returns_handle_and_history() -> None:
    sdk = WorkoutSDK(seed=42)
    handle, history = sdk.train_a2c(episodes=2)
    assert isinstance(handle, PolicyHandle)
    assert handle.algorithm == "A2C"
    assert handle.episodes_trained == 2
    assert isinstance(history, A2CHistory)
    assert history.episodes_run == 2
    assert all(math.isfinite(r) for r in history.rewards)


# ---------------------------------------------------------------- compare
def test_compare_returns_comparison_result() -> None:
    sdk = WorkoutSDK(seed=42)
    result = sdk.compare(seeds=2, episodes=2)
    assert isinstance(result, ComparisonResult)
    assert result.seed_count == 2
    assert result.episode_count == 2
    assert result.reinforce_mean_reward.shape == (2,)
    assert result.a2c_mean_reward.shape == (2,)


# ---------------------------------------------------------------- recommend
def test_recommend_returns_valid_recommendation() -> None:
    sdk = WorkoutSDK(seed=42)
    handle, _ = sdk.train_reinforce(episodes=1)
    rec = sdk.recommend(State.initial(), policy=handle)
    assert isinstance(rec, WorkoutRecommendation)
    assert 0 <= rec.action_id < ACTION_COUNT
    assert len(rec.probs) == ACTION_COUNT
    assert math.isclose(sum(rec.probs), 1.0, abs_tol=1e-5)
    assert len(rec.next_state_predicted) == STATE_DIM


def test_recommend_without_explicit_policy_uses_last_trained() -> None:
    sdk = WorkoutSDK(seed=42)
    sdk.train_a2c(episodes=1)
    rec = sdk.recommend(State.initial())  # no policy arg
    assert isinstance(rec, WorkoutRecommendation)
    assert 0 <= rec.action_id < ACTION_COUNT


def test_recommend_raises_when_no_policy_trained() -> None:
    sdk = WorkoutSDK(seed=42)
    with pytest.raises(RuntimeError, match=r"No policy trained yet"):
        sdk.recommend(State.initial())


# ------------------------------------------------------- import discipline
def test_ui_layers_import_only_from_src_sdk() -> None:
    """UIs (CLI menu, notebooks) MUST only import business logic from src.sdk."""
    repo = Path(__file__).resolve().parents[2]
    ui_files = list((repo / "src" / "cli").rglob("*.py")) + list((repo / "notebooks").rglob("*.py"))
    if not ui_files:
        pytest.skip("no UI files to inspect")
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+src\.(?!sdk(?:\.|\s|$))(env|model|services|utils|data)\b",
        re.MULTILINE,
    )
    offenders: list[str] = []
    for path in ui_files:
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        for match in forbidden.finditer(text):
            offenders.append(f"{path.relative_to(repo)}: {match.group(0).strip()}")
    assert not offenders, "UI layers must only import from src.sdk:\n" + "\n".join(offenders)
