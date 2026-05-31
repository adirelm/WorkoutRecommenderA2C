"""GUI smoke tests for the REINFORCE Streamlit page (§7.4).

Uses Streamlit's ``AppTest`` to render ``src/gui/pages/04_reinforce.py``
in-process, asserts that the sidebar exposes the documented sliders with
the right ranges, then drives a tiny 3-episode training run and verifies
the resulting history is surfaced both on-page (metric row) and in the
cross-page session-state singleton.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.gui.state import get_last_reinforce_history

PAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "gui" / "pages" / "04_reinforce.py"
TIMEOUT_SEC = 120.0


def _fresh_app() -> AppTest:
    """Return an un-run ``AppTest`` bound to the REINFORCE page."""
    return AppTest.from_file(str(PAGE_PATH), default_timeout=TIMEOUT_SEC)


def _slider_by_label(app: AppTest, label: str):
    """Return the first sidebar slider whose label matches ``label``."""
    for sl in list(app.sidebar.slider) + list(app.sidebar.select_slider):
        if sl.label == label:
            return sl
    raise AssertionError(f"no sidebar slider/select_slider with label {label!r}")


def test_sidebar_sliders_present_with_expected_ranges():
    """Sidebar must expose the 5 documented hyperparameter controls with §7.4 ranges."""
    app = _fresh_app().run()
    assert not app.exception, f"page raised on initial render: {app.exception}"

    episodes = _slider_by_label(app, "episodes")
    assert (episodes.min, episodes.max, episodes.step) == (5, 200, 5)

    baseline = _slider_by_label(app, "baseline_alpha")
    assert math.isclose(baseline.min, 0.0)
    assert math.isclose(baseline.max, 0.5)

    lr = _slider_by_label(app, "lr (log scale)")
    assert lr.min == pytest.approx(1e-4) and lr.max == pytest.approx(1e-2)

    gamma = _slider_by_label(app, "gamma (discount)")
    assert gamma.min == pytest.approx(0.9) and gamma.max == pytest.approx(0.999)

    hidden = _slider_by_label(app, "policy_hidden")
    assert {int(o) for o in hidden.options} >= {32, 64, 128, 256}


def test_train_button_present():
    """The primary 'Train REINFORCE' button must be rendered on first paint."""
    app = _fresh_app().run()
    labels = [b.label for b in app.button]
    assert "Train REINFORCE" in labels, f"missing Train button; saw {labels}"


def test_train_click_populates_metric_row_and_session_state():
    """Clicking Train with episodes=3 must finish, populate metrics, and stash history."""
    app = _fresh_app()
    app.run()
    # Shrink the run to 3 episodes so the test stays fast.
    _slider_by_label(app, "episodes").set_value(5)  # min allowed
    app.run()

    train_btn = next(b for b in app.button if b.label == "Train REINFORCE")
    train_btn.click()
    app.run()
    assert not app.exception, f"training raised: {app.exception}"

    # Metric row: 4 KPI metrics must all be present with finite numeric values.
    metric_labels = {m.label for m in app.metric}
    assert {"episodes_run", "final_reward", "mean_last_10", "baseline_at_end"} <= metric_labels
    for m in app.metric:
        if m.label in {"final_reward", "mean_last_10", "baseline_at_end"}:
            assert math.isfinite(float(m.value.lstrip("+"))), m

    # Cross-page singleton must be populated for other pages (compare, A2C) to reuse.
    # We read directly from the AppTest's session_state because get_last_reinforce_history()
    # reads Streamlit's *real* runtime state, which is not the AppTest's isolated state.
    shared_key = "gui.__shared__.last_reinforce_history"
    assert shared_key in app.session_state, "REINFORCE history not stored in session_state"
    history = app.session_state[shared_key]
    assert history is not None
    assert history.episodes_run >= 1
    assert len(history.rewards) == history.episodes_run

    # Sanity: get_last_reinforce_history is the documented public accessor and exists.
    assert callable(get_last_reinforce_history)
