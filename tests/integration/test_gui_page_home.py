"""Deep tests for ``src/gui/pages/01_home.py`` (§7.7 deliverable status).

Uses Streamlit's :class:`streamlit.testing.v1.AppTest` to drive the page
in-process, then inspects the resulting widget tree. Covers:

* hero element renders (HTML markdown card),
* deliverable checklist has the brief's §7.7 items,
* the four quick-start buttons are present,
* page footer is rendered.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

HOME_PATH = "src/gui/pages/01_home.py"

# Labels from the §7.7 deliverable checklist in 01_home.py.
EXPECTED_DELIVERABLES: tuple[str, ...] = (
    "LSTM world-model loss chart",
    "REINFORCE reward curve",
    "A2C training chart",
    "REINFORCE vs A2C comparison",
    "Executed analysis notebook",
    "Source notebook",
)

# Labels from ``_quick_start`` in 01_home.py.
EXPECTED_QUICK_START: tuple[str, ...] = (
    "Go to Data",
    "Go to REINFORCE",
    "Go to A2C",
    "Show Theory",
)


@pytest.fixture(scope="module")
def home_app() -> AppTest:
    """Run the home page once per module and reuse the AppTest object."""
    app = AppTest.from_file(HOME_PATH).run(timeout=30)
    assert not app.exception, f"home page raised: {app.exception}"
    return app


def _all_markdown_text(app: AppTest) -> str:
    """Concatenate every markdown body the page emitted."""
    return "\n".join(block.value for block in app.markdown)


def test_home_renders_hero_element(home_app: AppTest) -> None:
    """Hero is an HTML markdown card with title + subtitle + icon."""
    text = _all_markdown_text(home_app)
    assert "WorkoutRecommenderA2C" in text
    assert "LSTM world-model + REINFORCE & A2C" in text
    # Hero uses the Bar-Ilan primary blue gradient.
    assert "#003D7A" in text
    # And the weightlifter icon configured in render().
    assert "🏋️" in text


def test_home_deliverable_checklist_has_brief_7_7_items(home_app: AppTest) -> None:
    """Each §7.7 deliverable label must appear in the rendered markdown."""
    text = _all_markdown_text(home_app)
    subheader_text = "\n".join(sh.value for sh in home_app.subheader)
    assert "§7.7 deliverable checklist" in subheader_text
    for label in EXPECTED_DELIVERABLES:
        assert label in text, f"missing §7.7 deliverable row: {label!r}"
    # Glyph is either present ('done') or absent ('todo'); at least one of
    # the two checklist glyphs must show up so the component actually ran.
    assert ("✅" in text) or ("⬜" in text)


def test_home_has_four_quick_start_buttons(home_app: AppTest) -> None:
    """Quick-start row exposes exactly the four navigation buttons."""
    labels = [btn.label for btn in home_app.button]
    for expected in EXPECTED_QUICK_START:
        assert expected in labels, f"quick-start button missing: {expected!r}"
    matching = [lbl for lbl in labels if lbl in EXPECTED_QUICK_START]
    assert len(matching) == 4, f"expected 4 quick-start buttons, got {matching}"


def test_home_footer_present(home_app: AppTest) -> None:
    """Footer renders the §1.4 architect/implementer reminder + repo link."""
    text = _all_markdown_text(home_app)
    assert "Bar-Ilan University" in text
    assert "Assignment 3" in text
    assert "§1.4" in text
    assert "Human is architect, AI is implementer" in text
    assert "AdirElmakyes/Assignment3-WorkoutRecommenderA2C" in text
