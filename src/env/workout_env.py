"""WorkoutEnv — gym-like env wrapping the Phase-2 SyntheticTrainee (brief §7.3).

Phase 3 swaps the trainee for the frozen LSTM world model — the env API
stays identical. Composition over inheritance: the env owns a transition
provider, a reward function, and an action-mask service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.env.action_mask import ActionMaskService
from src.env.reward import RewardConfig, RewardFunction
from src.env.state import ACTION_COUNT, STATE_DIM, State
from src.env.synthetic_trainee import SyntheticTrainee
from src.env.workout_env_helpers import (
    ACTION_GROUP,
    ACTION_VOLUME,
    build_step_info,
    normalised_share,
    update_muscle_share,
    zero_share,
)


@dataclass(frozen=True)
class EnvConfig:
    """Episode + target shape — algorithm-relevant, mirrored from config.yaml."""

    episode_length: int = 28
    weekly_target_volume: float = 100.0
    target_muscle_dist: tuple = field(
        default=(
            ("push", 1 / 6),
            ("pull", 1 / 6),
            ("legs", 1 / 6),
            ("core", 1 / 6),
            ("cardio", 1 / 6),
            ("mobility", 1 / 6),
        )
    )
    user_baseline_7d_volume: float = 50.0


class WorkoutEnv:
    """Gym-like env. Phase-2 transition provider = SyntheticTrainee."""

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        trainee: SyntheticTrainee | None = None,
        mask_service: ActionMaskService | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = env_config if env_config is not None else EnvConfig()
        self._reward_cfg = reward_config
        self._reward_fn = RewardFunction(reward_config)
        self._injected_trainee = trainee
        self._mask_service = mask_service if mask_service is not None else ActionMaskService()
        self._seed = int(seed)
        self._trainee: SyntheticTrainee = self._build_trainee()
        self._state: State = State.initial()
        self._seed_trainee_history(self._state)
        self._step_count: int = 0
        self._history: list[int] = []
        self._mask: np.ndarray = self._mask_service.mask(self._state, self._history)
        self._muscle_volume_14d: dict[str, float] = zero_share()

    # ----------------------------------------------------------------- props
    @property
    def action_space(self) -> int:
        return ACTION_COUNT

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    # ----------------------------------------------------------------- core
    def reset(self, seed: int | None = None) -> State:
        """Reset env to State.initial(); optionally re-seed."""
        if seed is not None:
            self._seed = int(seed)
        self._trainee = self._build_trainee()
        self._state = State.initial()
        self._seed_trainee_history(self._state)
        self._step_count = 0
        self._history = []
        self._muscle_volume_14d = zero_share()
        self._mask = self._mask_service.mask(self._state, self._history)
        return self._state

    def step(self, action_id: int) -> tuple[State, float, bool, dict]:
        """Apply action_id; return (next_state, reward, done, info-with-reward-decomposition)."""
        if not 0 <= action_id < ACTION_COUNT:
            raise ValueError(f"action_id {action_id} out of range [0, {ACTION_COUNT})")
        prev_state = self._state
        volume_delta = ACTION_VOLUME[action_id]
        group = ACTION_GROUP[action_id]
        next_state = self._trainee.next_state(
            prev_state,
            action_id=action_id,
            prescribed_volume=volume_delta,
        )
        update_muscle_share(self._muscle_volume_14d, group, volume_delta)
        share_norm = normalised_share(self._muscle_volume_14d)
        target_share = {k: v for k, v in self.cfg.target_muscle_dist if k in share_norm}
        decomp = self._reward_fn.compute(
            state=prev_state,
            next_state=next_state,
            action_id=action_id,
            weekly_target=self.cfg.weekly_target_volume,
            baseline_7d_volume=self.cfg.user_baseline_7d_volume,
            muscle_share_14d=share_norm,
            target_muscle_dist=target_share,
        )
        self._state = next_state
        self._step_count += 1
        self._history.append(int(action_id))
        self._mask = self._mask_service.mask(self._state, self._history)
        done = self._step_count >= self.cfg.episode_length
        info = build_step_info(decomp, volume_delta, group, self._step_count)
        return next_state, float(decomp["reward"]), done, info

    # --------------------------------------------------------------- masks
    def action_mask(self) -> np.ndarray:
        """Current legal-action mask of shape (ACTION_COUNT,); per ADR-004 / Huang & Ontañón 2022."""
        return self._mask.copy()

    def history(self) -> list[int]:
        return list(self._history)

    # ---------------------------------------------------------- internals
    def _build_trainee(self) -> SyntheticTrainee:
        if self._injected_trainee is not None:
            return self._injected_trainee
        return SyntheticTrainee(rng=np.random.default_rng(self._seed))

    def _seed_trainee_history(self, initial_state: State) -> None:
        """Seed a history-carrying transition provider (e.g. LSTMEnvAdapter) per episode.

        SyntheticTrainee is stateless and exposes no ``reset``; the LSTM adapter
        (brief §7.3) needs its (state, action) window primed with the initial
        state at the start of every episode, so we call ``reset`` when present.
        """
        reset = getattr(self._trainee, "reset", None)
        if callable(reset):
            reset(initial_state)
