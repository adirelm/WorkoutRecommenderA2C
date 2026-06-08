"""Deep tests for ``src/gui/pages/02_data.py`` (§7.2 Data & Environment).

Drives the page in-process with :class:`streamlit.testing.v1.AppTest`,
then walks the resulting widget tree. Covers:

* the "Load PHUL trainee" button renders on first paint,
* clicking that button populates the ``gui.data.*`` session-state
  namespace (handle / trajectory / infos),
* after load the ``metric_row`` exposes the real chosen program name
  ``'Optimized Phul (Power Hypertrophy Upper Lower)'`` (title-cased from the
  real Kaggle ``program_name``), ``n_days=28``, and ``state_dim=12`` (the
  LogbookHandle contract from :meth:`WorkoutSDK.prepare_data`),
* the muscle-distribution heatmap appears as a ``plotly_chart`` element
  in the post-load element tree.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

DATA_PATH = "src/gui/pages/02_data.py"
LOAD_BUTTON_LABEL = "Load PHUL trainee"
_RUN_TIMEOUT_SEC = 60.0


def _find_button(app: AppTest, label: str):
    """Return the first button whose ``.label`` matches ``label``."""
    for btn in app.button:
        if btn.label == label:
            return btn
    raise AssertionError(f"button {label!r} not found; available labels: {[b.label for b in app.button]!r}")


def _metric_dict(app: AppTest) -> dict[str, str]:
    """Project ``app.metric`` into a ``{label: value}`` dict."""
    return {m.label: m.value for m in app.metric}


def _plotly_count(app: AppTest) -> int:
    """Count ``st.plotly_chart`` nodes in the rendered element tree."""
    return len(list(app.get("plotly_chart")))


@pytest.fixture
def fresh_app() -> AppTest:
    """Fresh AppTest instance (no button click yet) for the Data page."""
    app = AppTest.from_file(DATA_PATH, default_timeout=_RUN_TIMEOUT_SEC).run()
    assert not app.exception, f"data page raised on initial render: {list(app.exception)}"
    return app


@pytest.fixture
def loaded_app(fresh_app: AppTest) -> AppTest:
    """AppTest after the user has clicked the ``Load PHUL trainee`` button."""
    _find_button(fresh_app, LOAD_BUTTON_LABEL).click()
    fresh_app.run()
    assert not fresh_app.exception, (
        f"data page raised after clicking {LOAD_BUTTON_LABEL!r}: {list(fresh_app.exception)}"
    )
    return fresh_app


def test_data_page_shows_load_button_on_first_render(fresh_app: AppTest) -> None:
    """The ``Load PHUL trainee`` primary button is visible before any click."""
    labels = [btn.label for btn in fresh_app.button]
    assert LOAD_BUTTON_LABEL in labels, f"expected {LOAD_BUTTON_LABEL!r} in {labels!r}"


def test_load_button_click_populates_session_state(loaded_app: AppTest) -> None:
    """Clicking the load button writes the ``gui.data.*`` namespace keys."""
    state_keys = set(loaded_app.session_state.filtered_state.keys())
    expected = {"gui.data.handle", "gui.data.trajectory", "gui.data.infos"}
    missing = expected - state_keys
    assert not missing, (
        f"missing gui.data.* keys after load: {missing!r}; "
        f"present data keys = {sorted(k for k in state_keys if k.startswith('gui.data.'))}"
    )
    handle = loaded_app.session_state["gui.data.handle"]
    assert handle.program_name == "Optimized PHUL (Power Hypertrophy Upper Lower)"
    assert int(handle.n_days) == 28
    assert int(handle.state_dim) == 12


def test_metric_row_surfaces_logbook_handle(loaded_app: AppTest) -> None:
    """After load, the KPI row mirrors the LogbookHandle fields."""
    metrics = _metric_dict(loaded_app)
    assert metrics.get("Program") == "Optimized Phul (Power Hypertrophy Upper Lower)"
    assert metrics.get("Episode length (days)") == "28"
    assert metrics.get("State dim (channels)") == "12"


def test_muscle_distribution_heatmap_renders(loaded_app: AppTest) -> None:
    """The §7.2 heatmap is emitted via ``st.plotly_chart`` after load."""
    assert _plotly_count(loaded_app) >= 1, (
        "expected at least one plotly_chart (muscle-distribution heatmap) in the post-load element tree"
    )


def test_first_render_has_no_plotly_chart(fresh_app: AppTest) -> None:
    """Before the user loads data, no chart should be drawn yet."""
    assert _plotly_count(fresh_app) == 0
