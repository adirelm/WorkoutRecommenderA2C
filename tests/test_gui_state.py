"""Tests for src/gui/state.py (§7 namespaced session-state accessor).

``streamlit.session_state`` is a singleton tied to a script run, so we
substitute it with a plain dict via ``monkeypatch``. That lets us
exercise the GUIState namespacing rules + the cross-page singleton
accessors without booting a Streamlit runtime.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.gui import state as state_mod
from src.gui.state import (
    GUIState,
    get_last_a2c_history,
    get_last_reinforce_history,
    get_last_world_model_history,
    set_last_a2c_history,
    set_last_reinforce_history,
    set_last_world_model_history,
)
from src.model.types import LSTMTrainHistory
from src.services.a2c_types import A2CHistory
from src.services.types import REINFORCEHistory


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``st.session_state`` with a fresh dict for one test."""
    fake: dict[str, Any] = {}
    monkeypatch.setattr(state_mod.st, "session_state", fake, raising=True)
    return fake


def test_constructor_rejects_empty_or_reserved_pages() -> None:
    with pytest.raises(ValueError):
        GUIState("")
    with pytest.raises(ValueError):
        GUIState("__shared__")


def test_set_and_get_round_trip(fake_session: dict[str, Any]) -> None:
    gs = GUIState("train_a2c")
    assert gs.get("episodes") is None
    assert gs.get("episodes", default=0) == 0
    gs.set("episodes", 500)
    assert gs.has("episodes")
    assert gs.get("episodes") == 500
    assert fake_session == {"gui.train_a2c.episodes": 500}


def test_pages_are_isolated(fake_session: dict[str, Any]) -> None:
    a = GUIState("train_a2c")
    b = GUIState("train_reinforce")
    a.set("history", "a-history")
    b.set("history", "b-history")
    assert a.get("history") == "a-history"
    assert b.get("history") == "b-history"
    assert set(fake_session.keys()) == {
        "gui.train_a2c.history",
        "gui.train_reinforce.history",
    }


def test_clear_page_only_removes_own_namespace(fake_session: dict[str, Any]) -> None:
    a = GUIState("train_a2c")
    b = GUIState("train_reinforce")
    a.set("history", 1)
    a.set("seed", 42)
    b.set("history", 2)
    fake_session["unrelated.key"] = "keep"
    a.clear_page()
    assert not a.has("history") and not a.has("seed")
    assert b.has("history")
    assert fake_session["unrelated.key"] == "keep"


def test_full_key_rejects_empty_arguments() -> None:
    with pytest.raises(ValueError):
        state_mod._full_key("", "k")
    with pytest.raises(ValueError):
        state_mod._full_key("page", "")


def test_cross_page_history_round_trip(fake_session: dict[str, Any]) -> None:
    assert get_last_reinforce_history() is None
    assert get_last_a2c_history() is None
    assert get_last_world_model_history() is None

    r = REINFORCEHistory(episodes_run=2, rewards=(0.1, 0.2), losses=(0.5, 0.4), baseline=(0.0, 0.05), seed=1)
    a = A2CHistory(
        episodes_run=2,
        rewards=(0.3, 0.4),
        actor_losses=(0.2, 0.15),
        critic_losses=(0.1, 0.08),
        advantages_mean=(0.0, 0.0),
        seed=2,
    )
    w = LSTMTrainHistory(epochs_run=1, train_loss=(0.5,), val_loss=(0.6,), best_epoch=0)
    set_last_reinforce_history(r)
    set_last_a2c_history(a)
    set_last_world_model_history(w)

    assert get_last_reinforce_history() is r
    assert get_last_a2c_history() is a
    assert get_last_world_model_history() is w
    # Keys land in the reserved __shared__ namespace, not any page namespace.
    assert all(k.startswith("gui.__shared__.") for k in fake_session)
