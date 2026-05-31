"""WorkoutSDK — single entry point for UIs / notebook. CLAUDE.md §3.

UIs (CLI menu, notebooks) MUST import only from ``src.sdk``. Business logic
lives behind this facade: env construction, trainer wiring, policy storage,
and recommendation are all routed through here.
"""

from __future__ import annotations

from src.env.state import ACTION_COUNT, ACTION_NAMES, STATE_DIM, State
from src.env.workout_env import WorkoutEnv
from src.model.actor_critic import ActorCriticNet
from src.model.policy_net import PolicyNet
from src.sdk.sdk_helpers import build_policy_handle, recommend_from_net
from src.sdk.types import (
    LogbookHandle,
    PolicyHandle,
    WorkoutRecommendation,
    WorldModelHandle,
)
from src.services.a2c_trainer import A2CTrainer
from src.services.a2c_types import A2CConfig, A2CHistory
from src.services.comparator import ComparisonResult, compare
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig, REINFORCEHistory


class WorkoutSDK:
    """Single business-logic entry point. UIs (CLI/notebook) call only this class."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._env: WorkoutEnv | None = None
        self._last_policy_handle: PolicyHandle | None = None
        self._last_net: PolicyNet | ActorCriticNet | None = None

    # ------------------------------------------------------------------ data
    def prepare_data(self) -> LogbookHandle:
        """Construct a WorkoutEnv and surface its config as a LogbookHandle."""
        self._env = WorkoutEnv(seed=self.seed)
        return LogbookHandle(
            program_name="synthetic_trainee",
            n_days=int(self._env.cfg.episode_length),
            state_dim=STATE_DIM,
        )

    def train_world_model(self) -> WorldModelHandle:
        """Stub kept for CLI compatibility — full Phase-3 LSTM training is offline."""
        raise NotImplementedError("train_world_model is exercised via Phase-3 scripts, not the SDK")

    # ------------------------------------------------------------- trainers
    def train_reinforce(self, episodes: int = 10) -> tuple[PolicyHandle, REINFORCEHistory]:
        env = self._ensure_env()
        policy = PolicyNet(seed=self.seed)
        cfg = REINFORCEConfig(episodes=int(episodes))
        history = REINFORCETrainer(policy, env, cfg, seed=self.seed).train(episodes=int(episodes))
        handle = build_policy_handle("REINFORCE", history.rewards, history.episodes_run)
        self._last_policy_handle = handle
        self._last_net = policy
        return handle, history

    def train_a2c(self, episodes: int = 10) -> tuple[PolicyHandle, A2CHistory]:
        env = self._ensure_env()
        ac = ActorCriticNet(seed=self.seed)
        cfg = A2CConfig(episodes=int(episodes))
        history = A2CTrainer(ac, env, cfg, seed=self.seed).train(episodes=int(episodes))
        handle = build_policy_handle("A2C", history.rewards, history.episodes_run)
        self._last_policy_handle = handle
        self._last_net = ac
        return handle, history

    # ----------------------------------------------------------- comparison
    def compare(self, seeds: int = 3, episodes: int = 5) -> ComparisonResult:
        r_hists: list[REINFORCEHistory] = []
        a_hists: list[A2CHistory] = []
        for s in range(int(seeds)):
            seed = self.seed + s
            env_r = WorkoutEnv(seed=seed)
            policy = PolicyNet(seed=seed)
            r_hists.append(
                REINFORCETrainer(policy, env_r, REINFORCEConfig(episodes=int(episodes)), seed=seed).train(
                    episodes=int(episodes)
                )
            )
            env_a = WorkoutEnv(seed=seed)
            ac = ActorCriticNet(seed=seed)
            a_hists.append(
                A2CTrainer(ac, env_a, A2CConfig(episodes=int(episodes)), seed=seed).train(
                    episodes=int(episodes)
                )
            )
        return compare(r_hists, a_hists)

    # ----------------------------------------------------------- recommend
    def recommend(
        self,
        state: State,
        policy: PolicyHandle | None = None,
    ) -> WorkoutRecommendation:
        if policy is not None and policy is not self._last_policy_handle:
            raise ValueError("Only the most recently trained policy handle is supported.")
        if self._last_net is None or self._last_policy_handle is None:
            raise RuntimeError("No policy trained yet — call train_reinforce or train_a2c first.")
        env = self._ensure_env()
        mask = env.action_mask()
        return recommend_from_net(self._last_net, state, mask, ACTION_NAMES, ACTION_COUNT)

    # --------------------------------------------------------------- helpers
    def _ensure_env(self) -> WorkoutEnv:
        if self._env is None:
            self.prepare_data()
        assert self._env is not None
        return self._env
