"""GUI smoke tests for the §7.6 comparison Streamlit page.

Uses Streamlit's ``AppTest`` to render ``src/gui/pages/06_compare.py``
in-process, asserts that the sidebar exposes the seed / episode sliders
with the right ranges, then drives a tiny 1-seed x 5-episode comparison
and verifies the resulting :class:`ComparisonResult` is surfaced on-page
(chart + metric row) and stashed in the page's session-state slot.
Mirrors ``test_gui_page_reinforce.py``.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.services.comparator import ComparisonResult

PAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "gui" / "pages" / "06_compare.py"
TIMEOUT_SEC = 180.0


def _fresh_app() -> AppTest:
    """Return an un-run ``AppTest`` bound to the compare page."""
    return AppTest.from_file(str(PAGE_PATH), default_timeout=TIMEOUT_SEC)


def _slider_by_label(app: AppTest, label: str):
    """Return the first sidebar slider whose label matches ``label``."""
    for sl in app.sidebar.slider:
        if sl.label == label:
            return sl
    raise AssertionError(f"no sidebar slider with label {label!r}")


def _run_button(app: AppTest):
    """Return the primary 'Run comparison ...' button (label has dynamic counts)."""
    for b in app.button:
        if b.label.startswith("Run comparison"):
            return b
    raise AssertionError(f"no 'Run comparison' button; saw {[b.label for b in app.button]}")


def test_sidebar_sliders_present_with_expected_ranges():
    """Sidebar must expose Seeds (1-10) and Episodes per seed (5-100) sliders."""
    app = _fresh_app().run()
    assert not app.exception, f"page raised on initial render: {app.exception}"

    seeds = _slider_by_label(app, "Seeds")
    assert (seeds.min, seeds.max, seeds.step) == (1, 10, 1)

    episodes = _slider_by_label(app, "Episodes per seed")
    assert (episodes.min, episodes.max, episodes.step) == (5, 100, 5)


def test_run_button_present_on_initial_paint():
    """The primary 'Run comparison (...)' button must be rendered on first paint."""
    app = _fresh_app().run()
    btn = _run_button(app)
    # Label should mention the current slider values for transparency.
    assert "seeds" in btn.label and "episodes" in btn.label


def test_idle_state_shows_configure_info():
    """Before any run, the page shows the 'Configure seeds / episodes...' hint."""
    app = _fresh_app().run()
    info_text = "\n".join(i.value for i in app.info)
    assert "Configure seeds" in info_text or "Run comparison" in info_text


def test_run_click_populates_chart_metrics_and_session_state():
    """Clicking Run with min sliders (1 seed x 5 episodes) must finish + surface results."""
    app = _fresh_app()
    app.run()
    _slider_by_label(app, "Seeds").set_value(1)
    _slider_by_label(app, "Episodes per seed").set_value(5)
    app.run()

    _run_button(app).click()
    app.run()
    assert not app.exception, f"comparison raised: {app.exception}"

    # Metric row: the 4 'honest statistics' rows must all be present.
    metric_labels = {m.label for m in app.metric}
    expected = {"REINFORCE final", "A2C final", "Seeds", "Episodes"}
    assert expected <= metric_labels, f"missing KPIs: {expected - metric_labels}"

    # Page-local session-state slot must hold the ComparisonResult.
    state_key = "gui.compare.last_result"
    assert state_key in app.session_state, "comparison result not stashed in session_state"
    result = app.session_state[state_key]
    assert isinstance(result, ComparisonResult)
    assert result.seed_count == 1
    assert result.episode_count == 5
    assert len(result.reinforce_mean_reward) == 5
    assert len(result.a2c_mean_reward) == 5

    # 'Honest statistics' subheader marks the rendered results panel.
    subheaders = {sh.value for sh in app.subheader}
    assert "Honest statistics" in subheaders
