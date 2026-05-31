"""§7 Streamlit session-state accessor + cross-page singletons.

Keys in ``st.session_state`` are namespaced ``gui.<page>.<key>`` so pages
cannot collide on shared names like ``"history"`` or ``"episodes"``.
The SDK is cached via ``st.cache_resource`` so it survives reruns; the
last training histories are stored under reserved cross-page keys.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.model.types import LSTMTrainHistory
from src.sdk.sdk import WorkoutSDK
from src.services.a2c_types import A2CHistory
from src.services.types import REINFORCEHistory

# Reserved cross-page namespace (not tied to any single page).
_SHARED = "gui.__shared__"
_KEY_REINFORCE = f"{_SHARED}.last_reinforce_history"
_KEY_A2C = f"{_SHARED}.last_a2c_history"
_KEY_WORLD = f"{_SHARED}.last_world_model_history"


def _full_key(page: str, key: str) -> str:
    """Return the namespaced session-state key for ``page`` + ``key``."""
    if not page:
        raise ValueError("GUIState page name must be non-empty")
    if not key:
        raise ValueError("GUIState key must be non-empty")
    return f"gui.{page}.{key}"


class GUIState:
    """Namespaced accessor for ``st.session_state``.

    Every page constructs ``GUIState("<page>")`` once at the top of its
    render function; reads / writes go through this object so the raw
    session-state dict is never touched directly.
    """

    def __init__(self, page: str) -> None:
        if not page or not isinstance(page, str):
            raise ValueError("GUIState requires a non-empty page name")
        if page.startswith("__"):
            raise ValueError("page names starting with '__' are reserved")
        self._page = page
        self._prefix = f"gui.{page}."

    @property
    def page(self) -> str:
        """Page name this accessor is bound to."""
        return self._page

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``st.session_state[gui.<page>.<key>]`` or ``default``."""
        return st.session_state.get(_full_key(self._page, key), default)

    def set(self, key: str, value: Any) -> None:
        """Write ``value`` to ``st.session_state[gui.<page>.<key>]``."""
        st.session_state[_full_key(self._page, key)] = value

    def has(self, key: str) -> bool:
        """Return ``True`` iff this namespaced key exists in session state."""
        return _full_key(self._page, key) in st.session_state

    def clear_page(self) -> None:
        """Delete every session-state key in this page's namespace."""
        doomed = [k for k in st.session_state.keys() if k.startswith(self._prefix)]
        for k in doomed:
            del st.session_state[k]


# --------------------------------------------------------------------------- #
# Cross-page singletons                                                        #
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_sdk() -> WorkoutSDK:
    """Return the process-wide :class:`WorkoutSDK` (cached across reruns)."""
    return WorkoutSDK()


def get_last_reinforce_history() -> REINFORCEHistory | None:
    """Most recent REINFORCE training history, or ``None`` if never trained."""
    return st.session_state.get(_KEY_REINFORCE)


def set_last_reinforce_history(history: REINFORCEHistory) -> None:
    """Record the latest REINFORCE history so other pages can chart it."""
    st.session_state[_KEY_REINFORCE] = history


def get_last_a2c_history() -> A2CHistory | None:
    """Most recent A2C training history, or ``None`` if never trained."""
    return st.session_state.get(_KEY_A2C)


def set_last_a2c_history(history: A2CHistory) -> None:
    """Record the latest A2C history so other pages can chart it."""
    st.session_state[_KEY_A2C] = history


def get_last_world_model_history() -> LSTMTrainHistory | None:
    """Most recent LSTM world-model training history, or ``None``."""
    return st.session_state.get(_KEY_WORLD)


def set_last_world_model_history(history: LSTMTrainHistory) -> None:
    """Record the latest LSTM world-model training history."""
    st.session_state[_KEY_WORLD] = history
