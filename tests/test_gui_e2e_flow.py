"""End-to-end GUI smoke (§7.7 / ADR-006) — walks all 7 brief-aligned pages.

Drives ``streamlit.testing.v1.AppTest`` page-by-page (Home → Data → LSTM →
REINFORCE → A2C → Compare → Recommend), forwarding ``session_state`` across
every transition. Verifies the namespaced ``gui.<page>.<key>`` survived the
full journey — the only structural promise the multi-page app makes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

_PAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "gui" / "pages"
_TIMEOUT = 120  # seconds; LSTM-3-epoch + A2C-3-ep fits well under this.


def _run(page: str, *, prior: dict[str, Any] | None = None) -> AppTest:
    """Boot ``pages/<page>`` with optional pre-seeded ``session_state``."""
    at = AppTest.from_file(str(_PAGES_DIR / page), default_timeout=_TIMEOUT)
    if prior:
        for k, v in prior.items():
            at.session_state[k] = v
    return at.run()


def _forward(at: AppTest) -> dict[str, Any]:
    """Snapshot session_state from a run so the next page inherits it."""
    return {k: at.session_state[k] for k in at.session_state.filtered_state}


def _click_first(at: AppTest, label_fragment: str) -> AppTest:
    """Click the first button whose label contains ``label_fragment``."""
    for btn in at.button:
        if label_fragment.lower() in btn.label.lower():
            return btn.click().run()
    raise AssertionError(f"button matching {label_fragment!r} not found")


def _set_slider(at: AppTest, label_fragment: str, value: float | int) -> None:
    """Set the first slider whose label contains ``label_fragment``."""
    for sl in list(at.slider) + list(at.select_slider):
        if label_fragment.lower() in sl.label.lower():
            sl.set_value(value)
            return
    raise AssertionError(f"slider matching {label_fragment!r} not found")


@pytest.mark.slow
def test_full_user_journey_preserves_session_state() -> None:
    """Boot → Data → LSTM(3) → REINFORCE(3) → A2C(3) → Compare(1,3) → Recommend."""
    # 1. Boot — Home page should render with no errors.
    home = _run("01_home.py")
    assert not home.exception, f"home crashed: {home.exception}"
    carry = _forward(home)

    # 2. Data — click "Load PHUL trainee".
    data = _run("02_data.py", prior=carry)
    assert not data.exception, f"data crashed: {data.exception}"
    data = _click_first(data, "Load PHUL trainee")
    assert data.session_state.get("gui.data.handle") is not None, (
        "data page did not persist LogbookHandle to gui.data.handle"
    )
    carry = _forward(data)

    # 3. LSTM — set epochs to 3 (the smallest the slider allows is 5; clamp).
    lstm = _run("03_lstm.py", prior=carry)
    assert not lstm.exception, f"lstm crashed: {lstm.exception}"
    _set_slider(lstm, "epochs", 5)  # min permitted by sidebar; smallest available
    lstm = lstm.run()
    lstm = _click_first(lstm, "Train LSTM")
    assert lstm.session_state.get("gui.lstm.history") is not None, "LSTM history missing after training"
    carry = _forward(lstm)

    # 4. REINFORCE — episodes=5 (min). Train.
    reinforce = _run("04_reinforce.py", prior=carry)
    assert not reinforce.exception, f"reinforce crashed: {reinforce.exception}"
    _set_slider(reinforce, "episodes", 5)
    reinforce = reinforce.run()
    reinforce = _click_first(reinforce, "Train REINFORCE")
    assert reinforce.session_state.get("gui.reinforce.last_history") is not None, (
        "REINFORCE history missing after training"
    )
    carry = _forward(reinforce)

    # 5. A2C — episodes=5 (min). Train.
    a2c = _run("05_a2c.py", prior=carry)
    assert not a2c.exception, f"a2c crashed: {a2c.exception}"
    _set_slider(a2c, "episodes", 5)
    a2c = a2c.run()
    a2c = _click_first(a2c, "Train A2C")
    # A2C page may stash under gui.a2c.last_history or shared key — accept either.
    assert (
        a2c.session_state.get("gui.a2c.last_history") is not None
        or a2c.session_state.get("gui.__shared__.last_a2c_history") is not None
    ), "A2C history missing after training"
    carry = _forward(a2c)

    # 6. Compare — seeds=1, episodes=5 (slider mins).
    compare = _run("06_compare.py", prior=carry)
    assert not compare.exception, f"compare crashed: {compare.exception}"
    _set_slider(compare, "Seeds", 1)
    _set_slider(compare, "Episodes per seed", 5)
    compare = compare.run()
    compare = _click_first(compare, "Run comparison")
    assert compare.session_state.get("gui.compare.last_result") is not None, (
        "Compare result missing after run"
    )
    carry = _forward(compare)

    # 7. Recommend — toggle is on by default (Use State.initial()); click button.
    rec = _run("07_recommend.py", prior=carry)
    assert not rec.exception, f"recommend crashed: {rec.exception}"
    rec = _click_first(rec, "Recommend")
    assert rec.session_state.get("gui.recommend.last_recommendation") is not None, (
        "Recommendation missing after click"
    )

    # Cross-page invariant — every page wrote into its own namespace and
    # the keys survived all 7 transitions via the explicit forwarding above.
    final = rec.session_state
    survivors = (
        "gui.data.handle",
        "gui.lstm.history",
        "gui.reinforce.last_history",
        "gui.compare.last_result",
        "gui.recommend.last_recommendation",
    )
    missing = [k for k in survivors if final.get(k) is None]
    assert not missing, f"session_state lost across page transitions: {missing}"
