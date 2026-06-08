"""Tests for the LSTM->RL wiring (closes audit finding F-1).

The frozen world model trained on the real PHUL trajectory becomes the
WorkoutEnv transition kernel that REINFORCE/A2C roll out against.
"""

from __future__ import annotations

from src.env.state import STATE_DIM, State
from src.model.lstm_env_adapter import LSTMEnvAdapter
from src.model.world_model_builder import (
    build_lstm_env,
    train_program_world_model,
    world_model_history,
)


def test_world_model_is_trained_and_frozen() -> None:
    model = train_program_world_model(seed=42, epochs=3)
    assert model.is_frozen()


def test_world_model_history_reports_loss_curves() -> None:
    """world_model_history exposes the fit history (loss curves + best epoch)."""
    hist = world_model_history(seed=42, epochs=3)
    assert hist.epochs_run == 3
    assert len(hist.train_loss) == 3
    assert len(hist.val_loss) == 3
    assert 0 <= hist.best_epoch < 3
    # Memoised: same (seed, epochs) returns the identical history object as the model fit.
    assert world_model_history(seed=42, epochs=3) is hist


def test_build_lstm_env_uses_lstm_adapter_as_transition() -> None:
    env = build_lstm_env(seed=42, model=train_program_world_model(seed=42, epochs=3))
    assert isinstance(env._trainee, LSTMEnvAdapter)


def test_lstm_env_steps_produce_valid_states() -> None:
    env = build_lstm_env(seed=42, model=train_program_world_model(seed=42, epochs=3))
    state = env.reset()
    assert isinstance(state, State)
    next_state, reward, _done, _info = env.step(1)
    assert isinstance(next_state, State)
    assert len(next_state.to_array()) == STATE_DIM
    assert isinstance(float(reward), float)


def test_lstm_env_rollout_is_deterministic_per_seed() -> None:
    model = train_program_world_model(seed=42, epochs=3)
    env_a = build_lstm_env(seed=42, model=model)
    env_b = build_lstm_env(seed=42, model=model)
    env_a.reset()
    env_b.reset()
    a = env_a.step(2)[0].to_array().tolist()
    b = env_b.step(2)[0].to_array().tolist()
    assert a == b
