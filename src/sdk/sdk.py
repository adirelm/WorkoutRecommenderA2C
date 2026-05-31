"""WorkoutSDK — single entry point for UIs / notebook. CLAUDE.md §3.

UIs (CLI menu, notebooks) MUST import only from ``src.sdk``. Business logic
lives behind this facade: env construction, trainer wiring, policy storage,
and recommendation are all routed through here.

V3 §12 extension point: ``_TRAINER_REGISTRY`` + :meth:`train` give an
open-closed seam — add a new on-policy algorithm by subclassing
:class:`src.services.base_trainer.BaseTrainer` and registering it; no
edits to this facade are required for the new algo to be reachable via
``sdk.train("ppo", ...)``.
"""

from __future__ import annotations

from typing import ClassVar

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
from src.services.base_trainer import BaseTrainer
from src.services.comparator import ComparisonResult, compare
from src.services.reinforce_trainer import REINFORCETrainer
from src.services.types import REINFORCEConfig, REINFORCEHistory


class WorkoutSDK:
    """Single business-logic entry point. UIs (CLI/notebook) call only this class."""

    # V3 §12 open-closed registry — new algos plug in here, never inside methods.
    _TRAINER_REGISTRY: ClassVar[dict[str, type[BaseTrainer]]] = {
        "reinforce": REINFORCETrainer,
        "a2c": A2CTrainer,
    }

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
    def train(self, algo: str, episodes: int = 10) -> tuple[PolicyHandle, REINFORCEHistory | A2CHistory]:
        """Pure registry-driven trainer dispatch (V3 §12 open-closed).

        Looks up ``algo`` in :attr:`_TRAINER_REGISTRY`, delegates net+config
        wiring to the trainer's own ``build`` classmethod, runs ``train()``,
        caches the trained net + handle, and returns the
        ``(handle, history)`` tuple the CLI / GUI / tests already consume.

        Adding a new algo (e.g. PPO) is a registry-only change: subclass
        :class:`BaseTrainer`, implement ``build`` + ``train`` + ``net``, and
        register the class — no edits to this method body.
        """
        key = algo.lower()
        if key not in self._TRAINER_REGISTRY:
            raise ValueError(f"unknown algo {algo!r}; registered: {sorted(self._TRAINER_REGISTRY)}")
        trainer_cls = self._TRAINER_REGISTRY[key]
        trainer = trainer_cls.build(env=self.ensure_env(), seed=self.seed, episodes=int(episodes))
        history = trainer.train(episodes=int(episodes))
        handle = build_policy_handle(key.upper(), history.rewards, history.episodes_run)
        self._last_policy_handle = handle
        self._last_net = trainer.net
        return handle, history

    def train_reinforce(self, episodes: int = 10) -> tuple[PolicyHandle, REINFORCEHistory]:
        """Backward-compat wrapper around ``train("reinforce", ...)``. Brief §7.4."""
        handle, history = self.train("reinforce", episodes=episodes)
        assert isinstance(history, REINFORCEHistory)
        return handle, history

    def train_a2c(self, episodes: int = 10) -> tuple[PolicyHandle, A2CHistory]:
        """Backward-compat wrapper around ``train("a2c", ...)``. Brief §7.5."""
        handle, history = self.train("a2c", episodes=episodes)
        assert isinstance(history, A2CHistory)
        return handle, history

    # ----------------------------------------------------------- comparison
    def compare(self, seeds: int = 3, episodes: int = 5) -> ComparisonResult:
        """Run both REINFORCE + A2C over N seeds x E episodes, return mean ± std bands. Brief §7.6."""
        r_hists: list[REINFORCEHistory] = []
        a_hists: list[A2CHistory] = []
        a2c_nets: list[ActorCriticNet] = []
        a2c_handles: list[PolicyHandle] = []
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
            a_hist = A2CTrainer(ac, env_a, A2CConfig(episodes=int(episodes)), seed=seed).train(
                episodes=int(episodes)
            )
            a_hists.append(a_hist)
            a2c_nets.append(ac)
            a2c_handles.append(build_policy_handle("A2C", a_hist.rewards, a_hist.episodes_run))
        self._last_net = a2c_nets[-1]
        self._last_policy_handle = a2c_handles[-1]
        return compare(r_hists, a_hists)

    # ----------------------------------------------------------- recommend
    def recommend(
        self,
        state: State,
        policy: PolicyHandle | None = None,
    ) -> WorkoutRecommendation:
        """Next-day recommendation; uses the most-recently trained policy if none specified."""
        if policy is not None and policy is not self._last_policy_handle:
            raise ValueError("Only the most recently trained policy handle is supported.")
        if self._last_net is None or self._last_policy_handle is None:
            raise RuntimeError("No policy trained yet — call train_reinforce or train_a2c first.")
        env = self.ensure_env()
        mask = env.action_mask()
        return recommend_from_net(self._last_net, state, mask, ACTION_NAMES, ACTION_COUNT)

    # --------------------------------------------------------------- helpers
    def ensure_env(self) -> WorkoutEnv:
        """Return the cached WorkoutEnv, building one via prepare_data() if needed.

        Public API (V3 §4 encapsulation fix) — GUI / notebook callers that need
        direct env access (e.g. for trajectory rollouts) call this instead of
        reaching into the previously-private ``_ensure_env``.
        """
        if self._env is None:
            self.prepare_data()
        assert self._env is not None
        return self._env
