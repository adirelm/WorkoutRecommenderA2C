"""Tests for BaseTrainer + trainer registry (V3 §4 + §12 closure).

Two regressions that motivate this file:

* BaseTrainer must be *abstract* — direct instantiation has to raise
  ``TypeError``, otherwise the ABC is purely cosmetic and subclasses can
  silently skip the shared seed/rollout contract.
* The SDK trainer registry must (a) include both shipped algorithms and
  (b) only hold types that actually inherit BaseTrainer; if either side
  drifts the open-closed extension point (V3 §12) is broken.
"""

from __future__ import annotations

import pytest

from src.env.workout_env import WorkoutEnv
from src.sdk.sdk import WorkoutSDK
from src.services.a2c_trainer import A2CTrainer
from src.services.a2c_types import A2CHistory
from src.services.base_trainer import BaseTrainer
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEHistory


def test_base_trainer_is_abstract() -> None:
    """Instantiating BaseTrainer directly must raise TypeError (abstract ABC)."""
    env = WorkoutEnv(seed=42)
    with pytest.raises(TypeError):
        BaseTrainer(env=env, config=None, seed=42)  # type: ignore[abstract]


def test_trainer_registry_includes_reinforce_and_a2c() -> None:
    """Both shipped algos must be registered, and their classes must extend BaseTrainer."""
    registry = WorkoutSDK._TRAINER_REGISTRY
    assert "reinforce" in registry, f"reinforce missing from registry: {sorted(registry)}"
    assert "a2c" in registry, f"a2c missing from registry: {sorted(registry)}"
    assert registry["reinforce"] is REINFORCETrainer
    assert registry["a2c"] is A2CTrainer
    for name, cls in registry.items():
        assert issubclass(cls, BaseTrainer), f"{name!r} entry {cls} does not extend BaseTrainer"


def test_sdk_generic_train_dispatches_to_reinforce() -> None:
    """SDK.train('reinforce', episodes=1) returns a (handle, REINFORCEHistory) tuple."""
    sdk = WorkoutSDK(seed=42)
    handle, history = sdk.train("reinforce", episodes=1)
    assert handle.algorithm == "REINFORCE"
    assert isinstance(history, REINFORCEHistory)
    assert history.episodes_run == 1


def test_sdk_generic_train_dispatches_to_a2c() -> None:
    """SDK.train('a2c', episodes=1) returns a (handle, A2CHistory) tuple."""
    sdk = WorkoutSDK(seed=42)
    handle, history = sdk.train("A2C", episodes=1)  # case-insensitive
    assert handle.algorithm == "A2C"
    assert isinstance(history, A2CHistory)
    assert history.episodes_run == 1


def test_sdk_generic_train_rejects_unknown_algo() -> None:
    """Unknown algo names must raise ValueError naming the registered keys."""
    sdk = WorkoutSDK(seed=42)
    with pytest.raises(ValueError, match=r"unknown algo"):
        sdk.train("ppo", episodes=1)


def test_sdk_ensure_env_is_public_and_idempotent() -> None:
    """ensure_env() must be a public method that caches the WorkoutEnv instance."""
    sdk = WorkoutSDK(seed=42)
    env_a = sdk.ensure_env()
    env_b = sdk.ensure_env()
    assert env_a is env_b, "ensure_env must return the cached WorkoutEnv on subsequent calls"
