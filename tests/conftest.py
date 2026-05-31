"""Shared pytest fixtures for unit/ and integration/ tests. V3 §6.

Fixtures defined here are auto-discovered by pytest for any test under
``tests/``. Keep this module thin — only fixtures that are duplicated
across multiple test files belong here. Test-local fixtures stay in the
test module that owns them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sdk.sdk import WorkoutSDK


@pytest.fixture(scope="session")
def sdk() -> WorkoutSDK:
    """Session-scoped WorkoutSDK facade for integration tests.

    Constructed once per pytest session (CLAUDE.md §3: SDK is the single
    entry point). Integration tests that exercise multi-module flows
    (training, recommendation) should depend on this fixture instead of
    instantiating ``WorkoutSDK`` themselves.
    """
    return WorkoutSDK(seed=42)


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Per-test scratch dir for artefacts (charts, checkpoints, JSON).

    Returns an *empty* ``tmp_path / "results"`` directory. Tests that
    write to ``results/`` should accept this fixture instead of touching
    the repo-level ``results/`` tree.
    """
    out = tmp_path / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures/`` (sample CSVs)."""
    return Path(__file__).parent / "fixtures"
