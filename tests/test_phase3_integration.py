"""Phase 3 integration: generate trajectory → build windows → train LSTM → wrap in adapter → rollout."""

import math

from src.env.state import STATE_DIM, State
from src.model.dataset import build_windows, split_train_val
from src.model.lstm_env_adapter import LSTMEnvAdapter
from src.model.lstm_trainer import LSTMTrainer
from src.model.lstm_world import LSTMWorldModel
from src.model.trajectory_builder import generate_trajectory
from src.model.types import LSTMTrainConfig


def test_end_to_end_lstm_pipeline():
    """Generate small trajectory → train LSTM (few epochs) → wrap in adapter → rollout 5 steps without crashing."""
    traj = generate_trajectory(num_days=28, seed=42)
    windows = build_windows(traj, window_len=7)
    train_w, val_w = split_train_val(windows, val_days=7)
    assert len(train_w) > 0 and len(val_w) > 0

    model = LSTMWorldModel(hidden_size=16, num_layers=1)  # small for test speed
    config = LSTMTrainConfig(epochs=3, hidden_size=16, batch_size=4)
    trainer = LSTMTrainer(model, config, seed=42)
    history = trainer.fit(train_w, val_w)
    assert history.epochs_run == 3
    assert all(not math.isnan(v) for v in history.train_loss)  # no NaN

    model.freeze()
    adapter = LSTMEnvAdapter(model, window_len=7)
    adapter.reset(State.initial())
    for _ in range(5):
        next_s = adapter.next_state(State.initial(), action_id=0)
        assert isinstance(next_s, State)
        arr = next_s.to_array()
        assert arr.shape == (STATE_DIM,)
        assert not any(math.isnan(x) for x in arr)  # no NaN


def test_lstm_predictions_in_state_ranges():
    """After training + freezing, LSTM-predicted states are clamped to valid ranges."""
    traj = generate_trajectory(num_days=21, seed=7)
    windows = build_windows(traj, window_len=7)
    train_w, val_w = split_train_val(windows, val_days=4)
    model = LSTMWorldModel(hidden_size=8, num_layers=1)
    config = LSTMTrainConfig(epochs=2, hidden_size=8, batch_size=4)
    trainer = LSTMTrainer(model, config, seed=7)
    trainer.fit(train_w, val_w)

    model.freeze()
    adapter = LSTMEnvAdapter(model, window_len=7)
    adapter.reset(State.initial())
    state = State.initial()
    for action_id in (0, 1, 2, 3, 4, 5, 6):
        nxt = adapter.next_state(state, action_id=action_id)
        # Channel range invariants per ADR-002 / lstm_env_adapter._vector_to_state.
        assert 0.0 <= nxt.fatigue <= 1.0
        assert 0.0 <= nxt.soreness_push <= 1.0
        assert 0.0 <= nxt.soreness_pull <= 1.0
        assert 0.0 <= nxt.soreness_legs <= 1.0
        assert 0.0 <= nxt.soreness_core <= 1.0
        assert 0.0 <= nxt.readiness <= 1.0
        assert nxt.rolling_7d_volume >= 0.0
        assert nxt.streak_days_trained >= 0
        assert nxt.days_since_last_rest >= 0
        assert -1.0 <= nxt.muscle_balance_push_vs_pull <= 1.0
        assert -1.0 <= nxt.adherence_signal <= 1.0
        assert 0.0 <= nxt.weekly_progress <= 1.2
        state = nxt
