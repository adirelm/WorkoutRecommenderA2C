"""config.yaml consumption guard (CLAUDE.md §4 / V3 §7.2).

Every tunable in config.yaml must be the value the runtime actually uses —
not a dead mirror of dataclass defaults. These tests bind yaml sections to
the objects the SDK/env/trainers construct, and fail on drift either way.
"""

from __future__ import annotations

from src.data.preprocessor import TIME_ENCODED_KEYWORDS
from src.env.state import ACTION_COUNT, STATE_DIM
from src.env.workout_env import WorkoutEnv, _env_config_from_config
from src.env.workout_env_helpers import reward_config_from_yaml
from src.services.a2c_types import A2CConfig
from src.services.types import REINFORCEConfig
from src.utils.config_loader import load_config

# Keys that are deliberately documentation-only (no runtime consumer); a new
# unconsumed key must be added here EXPLICITLY or these tests fail.
DOCUMENTED_ONLY_TOP_LEVEL = {"seed"}  # SDK takes seed as a parameter (default 42)


def test_rewards_section_is_the_runtime_reward_config() -> None:
    r = load_config()["rewards"]
    cfg = reward_config_from_yaml()
    assert cfg.lambda_1 == float(r["lambda_1"])
    assert cfg.lambda_2 == float(r["lambda_2"])
    assert cfg.w_progress == float(r["w_progress"])
    assert cfg.w_variety == float(r["w_variety"])
    assert cfg.overload_threshold_multiplier == float(r["overload_threshold_multiplier"])
    assert cfg.overload_exponent == float(r["overload_exponent"])
    assert cfg.progress_clip_ceiling == float(r["progress_clip_ceiling"])


def test_reinforce_section_is_the_runtime_trainer_config() -> None:
    c = load_config()["reinforce"]
    cfg = REINFORCEConfig.from_yaml()
    assert (cfg.policy_hidden, cfg.lr, cfg.gamma, cfg.episodes) == (
        int(c["policy_hidden"]),
        float(c["lr"]),
        float(c["gamma"]),
        int(c["episodes"]),
    )
    assert (cfg.baseline_alpha, cfg.entropy_coef, cfg.grad_clip_norm) == (
        float(c["baseline_alpha"]),
        float(c["entropy_coef"]),
        float(c["grad_clip_norm"]),
    )
    assert REINFORCEConfig.from_yaml(episodes=7).episodes == 7


def test_a2c_section_is_the_runtime_trainer_config() -> None:
    c = load_config()["a2c"]
    cfg = A2CConfig.from_yaml()
    assert (cfg.actor_hidden, cfg.critic_hidden) == (int(c["actor_hidden"]), int(c["critic_hidden"]))
    assert (cfg.actor_lr, cfg.critic_lr, cfg.gamma) == (
        float(c["actor_lr"]),
        float(c["critic_lr"]),
        float(c["gamma"]),
    )
    assert (cfg.entropy_coef, cfg.episodes, cfg.grad_clip_norm) == (
        float(c["entropy_coef"]),
        int(c["episodes"]),
        float(c["grad_clip_norm"]),
    )
    assert A2CConfig.from_yaml(episodes=3).episodes == 3


def test_environment_section_drives_the_env_and_matches_constants() -> None:
    e = load_config()["environment"]
    assert _env_config_from_config().episode_length == int(e["episode_length"])
    assert WorkoutEnv(seed=0).cfg.episode_length == int(e["episode_length"])
    # Structural keys are informational mirrors of code constants — assert no drift.
    assert int(e["state_dim"]) == STATE_DIM
    assert int(e["action_count"]) == ACTION_COUNT


def test_action_masking_section_drives_the_mask_service() -> None:
    am = load_config()["action_masking"]
    svc = WorkoutEnv(seed=0)._mask_service
    assert svc._rest_streak_threshold == int(am["rest_streak_threshold_days"])
    assert svc._legs_soreness_threshold == float(am["legs_soreness_threshold"])
    assert svc._conditioning_overload_threshold == float(am["conditioning_overload_threshold"])


def test_data_quality_keywords_subset_of_code_vocabulary() -> None:
    yaml_kw = set(load_config()["data_quality"]["time_encoded_exercise_keywords"])
    assert yaml_kw <= set(TIME_ENCODED_KEYWORDS)


def test_no_unexplained_top_level_keys() -> None:
    """Every top-level yaml section must be consumed by code or whitelisted above."""
    consumed = {
        "version",
        "environment",
        "rewards",
        "data_quality",
        "lstm",
        "reinforce",
        "a2c",
        "action_masking",
        "dataset",
        "paths",
        "device",
    }
    assert set(load_config()) == consumed | DOCUMENTED_ONLY_TOP_LEVEL
