"""Train + freeze the LSTM world model, then expose it as the RL environment.

This closes audit finding F-1: REINFORCE and A2C now roll out against the
*trained, frozen* LSTM world model (brief §7.4/§7.5), not the hand-coded
SyntheticTrainee. The world model itself is fitted on the real PHUL trajectory
(:func:`src.model.program_trajectory.generate_program_trajectory`), so the whole
Kaggle → trajectory → LSTM → policy-gradient chain is wired end-to-end.

``train_program_world_model`` is memoised so the (deterministic) fit happens
once per (seed, epochs) and is reused across every seed/episode of a sweep.
"""

from __future__ import annotations

from functools import lru_cache

from src.env.reward import RewardConfig
from src.env.workout_env import WorkoutEnv
from src.model.dataset import build_windows, split_train_val
from src.model.lstm_env_adapter import LSTMEnvAdapter
from src.model.lstm_trainer import LSTMTrainer
from src.model.lstm_world import LSTMWorldModel
from src.model.program_trajectory import generate_program_trajectory
from src.model.types import LSTMTrainConfig, LSTMTrainHistory
from src.utils.config_loader import load_config
from src.utils.seeding import set_global_seed


@lru_cache(maxsize=8)
def _fit_world_model(seed: int, epochs: int | None) -> tuple[LSTMWorldModel, LSTMTrainHistory]:
    """Fit + freeze an LSTM world model on the real PHUL trajectory (memoised)."""
    cfg = load_config()
    lc = cfg["lstm"]
    window = int(lc["window"])
    set_global_seed(seed)
    trajectory = generate_program_trajectory(cfg, seed=seed)
    train, val = split_train_val(
        build_windows(trajectory, window_len=window), val_days=int(lc["val_split_days"])
    )
    model = LSTMWorldModel(
        hidden_size=int(lc["hidden_size"]),
        num_layers=int(lc["num_layers"]),
        dropout=float(lc["dropout"]),
        action_embed_dim=int(lc["action_embed_dim"]),
        seed=seed,
    )
    train_cfg = LSTMTrainConfig(
        hidden_size=int(lc["hidden_size"]),
        num_layers=int(lc["num_layers"]),
        dropout=float(lc["dropout"]),
        lr=float(lc["lr"]),
        epochs=int(epochs if epochs is not None else lc["epochs"]),
        window_len=window,
        batch_size=int(lc["batch_size"]),
        val_split_days=int(lc["val_split_days"]),
        action_embed_dim=int(lc["action_embed_dim"]),
    )
    history = LSTMTrainer(model=model, config=train_cfg, device="cpu", seed=seed).fit(train, val)
    model.freeze()
    return model, history


def train_program_world_model(seed: int = 42, epochs: int | None = None) -> LSTMWorldModel:
    """Return the frozen LSTM world model fitted on the real PHUL trajectory."""
    return _fit_world_model(seed, epochs)[0]


def world_model_history(seed: int = 42, epochs: int | None = None) -> LSTMTrainHistory:
    """Return the fit history (loss curves, best epoch) for the world model."""
    return _fit_world_model(seed, epochs)[1]


def build_lstm_env(
    seed: int = 42,
    model: LSTMWorldModel | None = None,
    reward_config: RewardConfig | None = None,
) -> WorkoutEnv:
    """Return a WorkoutEnv whose transition kernel is the frozen LSTM world model.

    ``reward_config`` lets callers (e.g. the λ-sensitivity sweep) vary the reward
    shaping while keeping the same learned transition model.
    """
    world_model = model if model is not None else train_program_world_model(seed=seed)
    window = int(load_config()["lstm"]["window"])
    adapter = LSTMEnvAdapter(world_model, window_len=window)
    return WorkoutEnv(trainee=adapter, reward_config=reward_config, seed=seed)
