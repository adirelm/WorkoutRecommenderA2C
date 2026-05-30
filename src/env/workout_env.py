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

# Per-action (prescribed volume, dominant muscle bucket) for share tracking.
_ACTIONS: tuple[tuple[float, str], ...] = (
    (0.0, "rest"),
    (10.0, "push"),
    (10.0, "pull"),
    (12.0, "legs"),
    (14.0, "full"),
    (6.0, "cardio"),
    (4.0, "mobility"),
)
_ACTION_VOLUME: dict[int, float] = {i: v for i, (v, _) in enumerate(_ACTIONS)}
_ACTION_GROUP: dict[int, str] = {i: g for i, (_, g) in enumerate(_ACTIONS)}
_GROUP_TO_SHARE_KEY: dict[str, str] = {
    "push": "push",
    "pull": "pull",
    "legs": "legs",
    "core": "core",
    "full": "push",
    "cardio": "core",
    "mobility": "core",
    "rest": "core",
}


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
        self._step_count: int = 0
        self._history: list[int] = []
        self._mask: np.ndarray = self._mask_service.mask(self._state, self._history)
        self._muscle_volume_14d: dict[str, float] = self._zero_share()

    # ----------------------------------------------------------------- props
    @property
    def action_space(self) -> int:
        return ACTION_COUNT

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    # ----------------------------------------------------------------- core
    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._seed = int(seed)
        self._trainee = self._build_trainee()
        self._state = State.initial()
        self._step_count = 0
        self._history = []
        self._muscle_volume_14d = self._zero_share()
        self._mask = self._mask_service.mask(self._state, self._history)
        return self._state

    def step(self, action_id: int) -> tuple[State, float, bool, dict]:
        if not 0 <= action_id < ACTION_COUNT:
            raise ValueError(f"action_id {action_id} out of range [0, {ACTION_COUNT})")
        prev_state = self._state
        volume_delta = _ACTION_VOLUME[action_id]
        group = _ACTION_GROUP[action_id]
        next_state = self._trainee.next_state(
            prev_state,
            action_id=action_id,
            prescribed_volume=volume_delta,
        )
        self._update_muscle_share(group, volume_delta)
        share_norm = self._normalised_share()
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
        info = {
            "gain": decomp["gain"],
            "overload": decomp["overload"],
            "imbalance": decomp["imbalance"],
            "progress": decomp.get("progress", 0.0),
            "variety": decomp.get("variety", 0.0),
            "volume_delta": float(volume_delta),
            "muscle_group": group,
            "step": self._step_count,
        }
        return next_state, float(decomp["reward"]), done, info

    # --------------------------------------------------------------- masks
    def action_mask(self) -> np.ndarray:
        return self._mask.copy()

    def history(self) -> list[int]:
        return list(self._history)

    # ---------------------------------------------------------- internals
    def _build_trainee(self) -> SyntheticTrainee:
        if self._injected_trainee is not None:
            return self._injected_trainee
        return SyntheticTrainee(rng=np.random.default_rng(self._seed))

    @staticmethod
    def _zero_share() -> dict[str, float]:
        return {"push": 0.0, "pull": 0.0, "legs": 0.0, "core": 0.0}

    def _update_muscle_share(self, group: str, volume_delta: float) -> None:
        key = _GROUP_TO_SHARE_KEY.get(group, "core")
        self._muscle_volume_14d[key] = self._muscle_volume_14d.get(key, 0.0) + float(volume_delta)

    def _normalised_share(self) -> dict[str, float]:
        total = sum(self._muscle_volume_14d.values())
        if total <= 0:
            return dict.fromkeys(self._muscle_volume_14d, 0.25)
        return {k: v / total for k, v in self._muscle_volume_14d.items()}
