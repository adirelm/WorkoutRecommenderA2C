"""GUI smoke tests for the A2C Streamlit page (§7.5).

Uses Streamlit's ``AppTest`` to render ``src/gui/pages/05_a2c.py``
in-process, asserts that the sidebar exposes the documented sliders with
the right ranges, then drives a tiny training run and verifies the
resulting history is surfaced both on-page (metric row) and in the
cross-page session-state singleton. Mirrors ``test_gui_page_reinforce.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.gui.state import get_last_a2c_history

PAGE_PATH = Path(__file__).resolve().parents[2] / "src" / "gui" / "pages" / "05_a2c.py"
TIMEOUT_SEC = 180.0


def _fresh_app() -> AppTest:
    """Return an un-run ``AppTest`` bound to the A2C page."""
    return AppTest.from_file(str(PAGE_PATH), default_timeout=TIMEOUT_SEC)


def _slider_by_label(app: AppTest, label: str):
    """Return the first sidebar slider/select_slider whose label matches ``label``."""
    for sl in list(app.sidebar.slider) + list(app.sidebar.select_slider):
        if sl.label == label:
            return sl
    raise AssertionError(f"no sidebar slider/select_slider with label {label!r}")


def test_sidebar_sliders_present_with_expected_ranges():
    """Sidebar must expose the documented A2C hyperparameter controls with §7.5 ranges."""
    app = _fresh_app().run()
    assert not app.exception, f"page raised on initial render: {app.exception}"

    episodes = _slider_by_label(app, "Training episodes")
    assert (episodes.min, episodes.max, episodes.step) == (5, 200, 5)

    for lr_label in ("Actor learning rate", "Critic learning rate"):
        sl = _slider_by_label(app, lr_label)
        assert sl.min == pytest.approx(1e-4) and sl.max == pytest.approx(1e-2)

    entropy = _slider_by_label(app, "Entropy bonus coefficient β")
    assert math.isclose(entropy.min, 0.0) and math.isclose(entropy.max, 0.1)
    clip = _slider_by_label(app, "Gradient clip")
    assert clip.min == pytest.approx(0.1) and clip.max == pytest.approx(2.0)

    gamma = _slider_by_label(app, "Discount factor γ")  # noqa: RUF001
    assert gamma.min == pytest.approx(0.9) and gamma.max == pytest.approx(0.999)

    for label in ("Actor hidden width", "Critic hidden width"):
        hidden = _slider_by_label(app, label)
        assert {int(o) for o in hidden.options} >= {32, 64, 128, 256}


def test_train_button_present():
    """The primary 'Train A2C' button must be rendered on first paint."""
    app = _fresh_app().run()
    labels = [b.label for b in app.button]
    assert "Train A2C" in labels, f"missing Train button; saw {labels}"


def test_train_click_populates_metric_row_and_session_state():
    """Clicking Train with the minimum episode count must finish and surface history."""
    app = _fresh_app()
    app.run()
    # Shrink the run to the minimum (5 episodes) so the test stays fast.
    _slider_by_label(app, "Training episodes").set_value(5)
    app.run()

    train_btn = next(b for b in app.button if b.label == "Train A2C")
    train_btn.click()
    app.run()
    assert not app.exception, f"training raised: {app.exception}"

    # Metric row: 5 KPI metrics must all be present with finite numeric values.
    metric_labels = {m.label for m in app.metric}
    expected = {"Episodes run", "Final reward", "Final actor loss", "Final critic loss", "Mean advantage"}
    assert expected <= metric_labels, f"missing KPIs: {expected - metric_labels}"
    for m in app.metric:
        if m.label in {"Final reward", "Final actor loss", "Final critic loss", "Mean advantage"}:
            assert math.isfinite(float(m.value.lstrip("+"))), m

    # Cross-page singleton must be populated for compare/recommend pages to reuse.
    shared_key = "gui.__shared__.last_a2c_history"
    assert shared_key in app.session_state, "A2C history not stored in session_state"
    history = app.session_state[shared_key]
    assert history is not None
    assert history.episodes_run >= 1
    assert len(history.rewards) == history.episodes_run
    assert len(history.actor_losses) == history.episodes_run
    assert len(history.critic_losses) == history.episodes_run

    # Sanity: get_last_a2c_history is the documented public accessor and exists.
    assert callable(get_last_a2c_history)
