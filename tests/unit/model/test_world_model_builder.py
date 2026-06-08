"""Tests for the LSTM->RL wiring (closes audit finding F-1).

The frozen world model trained on the real PHUL trajectory becomes the
WorkoutEnv transition kernel that REINFORCE/A2C roll out against.
"""

from __future__ import annotations

from src.env.state import STATE_DIM, State
from src.model.lstm_env_adapter import LSTMEnvAdapter
from src.model.world_model_builder import build_lstm_env, train_program_world_model


def test_world_model_is_trained_and_frozen() -> None:
    model = train_program_world_model(seed=42, epochs=3)
    assert model.is_frozen()


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
