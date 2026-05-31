"""Tests for env.synthetic_trainee — Phase-2 transition model (brief §7.3, §7.6 Q4).

Rules under test (kept intentionally simple — "plausible plus noise"):
- Training actions raise fatigue + soreness on targeted chains.
- Rest decays fatigue + all soreness multiplicatively.
- readiness ≈ 1 - fatigue - mean(soreness), clipped to [0, 1].
- streak_days_trained: +1 on any training action, reset to 0 on Rest.
- seeded rng → deterministic transitions.
- state stays in valid ranges over many steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.env.synthetic_trainee as st_mod
from src.env.state import State
from src.env.synthetic_trainee import SyntheticTrainee


def _fatigued_state() -> State:
    return State(
        fatigue=0.6,
        soreness_push=0.5,
        soreness_pull=0.4,
        soreness_legs=0.5,
        soreness_core=0.3,
        readiness=0.2,
        rolling_7d_volume=12.0,
        streak_days_trained=5,
        days_since_last_rest=5,
        muscle_balance_push_vs_pull=0.0,
        adherence_signal=0.0,
        weekly_progress=0.3,
    )


def test_rest_decays_fatigue_and_soreness() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(0), noise_sigma=0.0)
    s = _fatigued_state()
    nxt = trainee.next_state(s, action_id=0, prescribed_volume=0.0)
    assert nxt.fatigue < s.fatigue
    assert nxt.soreness_push < s.soreness_push
    assert nxt.soreness_pull < s.soreness_pull
    assert nxt.soreness_legs < s.soreness_legs
    assert nxt.soreness_core < s.soreness_core


def test_push_action_raises_soreness_push_only() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(1), noise_sigma=0.0)
    s = State.initial()
    nxt = trainee.next_state(s, action_id=1, prescribed_volume=10.0)
    # push chain rose meaningfully; non-push chains did not rise by similar amount
    assert nxt.soreness_push > 0.05
    assert nxt.soreness_pull < nxt.soreness_push / 2
    assert nxt.soreness_legs < nxt.soreness_push / 2
    assert nxt.soreness_core < nxt.soreness_push / 2


def test_full_body_distributes_soreness_across_chains() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(2), noise_sigma=0.0)
    s = State.initial()
    nxt = trainee.next_state(s, action_id=4, prescribed_volume=12.0)
    assert nxt.soreness_push > 0.0
    assert nxt.soreness_pull > 0.0
    assert nxt.soreness_legs > 0.0
    assert nxt.soreness_core > 0.0


def test_readiness_is_inverse_of_fatigue_plus_soreness() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(3), noise_sigma=0.0)
    s = _fatigued_state()
    nxt = trainee.next_state(s, action_id=0, prescribed_volume=0.0)
    mean_sore = (nxt.soreness_push + nxt.soreness_pull + nxt.soreness_legs + nxt.soreness_core) / 4.0
    expected = max(0.0, min(1.0, 1.0 - nxt.fatigue - mean_sore))
    assert nxt.readiness == pytest.approx(expected, abs=1e-6)


def test_streak_increments_on_training_resets_on_rest() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(4), noise_sigma=0.0)
    s = State.initial()
    after_push = trainee.next_state(s, action_id=1, prescribed_volume=10.0)
    assert after_push.streak_days_trained == s.streak_days_trained + 1
    after_rest = trainee.next_state(after_push, action_id=0, prescribed_volume=0.0)
    assert after_rest.streak_days_trained == 0


def test_seeded_rng_is_deterministic() -> None:
    s = _fatigued_state()
    t1 = SyntheticTrainee(rng=np.random.default_rng(42), noise_sigma=0.05)
    t2 = SyntheticTrainee(rng=np.random.default_rng(42), noise_sigma=0.05)
    a = t1.next_state(s, action_id=2, prescribed_volume=8.0)
    b = t2.next_state(s, action_id=2, prescribed_volume=8.0)
    np.testing.assert_allclose(a.to_array(), b.to_array(), rtol=0, atol=0)


def test_unknown_action_id_raises_value_error() -> None:
    """Line 67: unknown action_id falls outside _ACTION_MUSCLES and must raise."""
    trainee = SyntheticTrainee(rng=np.random.default_rng(0), noise_sigma=0.0)
    with pytest.raises(ValueError, match="unknown action_id"):
        trainee.next_state(State.initial(), action_id=99, prescribed_volume=5.0)


def test_non_rest_action_with_empty_targeted_skips_soreness_gain(monkeypatch) -> None:
    """Branch 88→93: non-rest action whose chain tuple is empty hits the
    `if targeted:` False arm and skips the per-chain gain loop. Reached via
    monkey-patching the module-level mapping for one action id."""
    patched = dict(st_mod._ACTION_MUSCLES)
    patched[1] = ()  # Push now claims no muscle chain
    monkeypatch.setattr(st_mod, "_ACTION_MUSCLES", patched)

    trainee = SyntheticTrainee(rng=np.random.default_rng(0), noise_sigma=0.0)
    s = _fatigued_state()
    nxt = trainee.next_state(s, action_id=1, prescribed_volume=10.0)
    # Soreness chains should only have *decayed* — no targeted gain was added.
    assert nxt.soreness_push < s.soreness_push
    assert nxt.soreness_pull < s.soreness_pull
    assert nxt.soreness_legs < s.soreness_legs
    assert nxt.soreness_core < s.soreness_core


def test_state_remains_in_valid_ranges_after_many_steps() -> None:
    trainee = SyntheticTrainee(rng=np.random.default_rng(7), noise_sigma=0.05)
    s = State.initial()
    rng = np.random.default_rng(7)
    for _ in range(500):
        action = int(rng.integers(0, 7))
        vol = float(rng.uniform(0.0, 15.0))
        s = trainee.next_state(s, action_id=action, prescribed_volume=vol)
        assert 0.0 <= s.fatigue <= 1.0
        assert 0.0 <= s.soreness_push <= 1.0
        assert 0.0 <= s.soreness_pull <= 1.0
        assert 0.0 <= s.soreness_legs <= 1.0
        assert 0.0 <= s.soreness_core <= 1.0
        assert 0.0 <= s.readiness <= 1.0
        assert s.rolling_7d_volume >= 0.0
        assert s.streak_days_trained >= 0
