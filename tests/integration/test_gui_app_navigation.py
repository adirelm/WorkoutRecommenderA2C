"""GUI app multipage navigation tests (§7.7).

Verifies ``src/gui/app.py`` wires exactly the 10 expected pages into the
sidebar via ``st.navigation`` and that the default page (Home) renders.

We hit the wiring from two angles so a future refactor cannot silently
drop or rename a page:

  1. **Static**: parse the ``PAGES`` literal in ``app.py`` and assert the
     full ordered list of ``(title, icon)`` pairs matches the brief's
     pipeline (Home → Data → … → Discussion).
  2. **Runtime**: drive ``AppTest.from_file(...).run()`` and confirm the
     sidebar block is present (i.e. ``st.navigation(position="sidebar")``
     materialised) and the first page (Home) is what renders by default.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_PATH = _REPO_ROOT / "src" / "gui" / "app.py"

EXPECTED_PAGES: tuple[tuple[str, str], ...] = (
    ("Home", "🏠"),
    ("Data", "📊"),
    ("LSTM World Model", "🧠"),
    ("REINFORCE", "🎯"),
    ("A2C", "⚡"),
    ("Compare", "⚖️"),
    ("Recommend", "💡"),
    ("Action Masking", "🛡️"),
    ("Theory", "📐"),
    ("Discussion", "💬"),
)


def _parse_pages_from_app() -> list[tuple[str, str]]:
    """Statically read ``PAGES = [st.Page(..., title=..., icon=...), ...]``."""
    tree = ast.parse(_APP_PATH.read_text(encoding="utf-8"))
    pages: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "PAGES"):
            continue
        assert isinstance(node.value, ast.List | ast.Tuple), "PAGES must be a list/tuple literal"
        for call in node.value.elts:
            assert isinstance(call, ast.Call), "PAGES entries must be st.Page(...) calls"
            kw = {k.arg: k.value for k in call.keywords}
            title = ast.literal_eval(kw["title"])
            icon = ast.literal_eval(kw["icon"])
            pages.append((title, icon))
        return pages
    pytest.fail("PAGES registry not found in src/gui/app.py")


def test_app_defines_exactly_ten_pages() -> None:
    pages = _parse_pages_from_app()
    assert len(pages) == 10, f"expected 10 pages, got {len(pages)}: {pages}"


def test_app_page_titles_and_icons_match_brief() -> None:
    assert _parse_pages_from_app() == list(EXPECTED_PAGES)


def test_app_home_is_first_page_in_registry() -> None:
    """Streamlit's default page is the first entry of ``st.navigation``."""
    first_title, first_icon = _parse_pages_from_app()[0]
    assert first_title == "Home"
    assert first_icon == "🏠"


def test_app_runs_and_renders_sidebar_navigation() -> None:
    """Driving ``app.py`` via AppTest produces a sidebar (navigation strip)."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=30).run()
    assert not at.exception, f"app.py raised on first run: {list(at.exception)}"
    # ``st.navigation(position="sidebar")`` always materialises a sidebar block.
    assert at.sidebar is not None, "expected sidebar from st.navigation(position='sidebar')"


def test_app_default_page_is_home() -> None:
    """AppTest renders the default page for multipage apps — must be Home."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=30).run()
    assert not at.exception, f"default page raised: {list(at.exception)}"
    # Home page calls ``hero(title='WorkoutRecommenderA2C', ...)`` and renders a
    # '§7.7 deliverable checklist' subheader — both unique to 01_home.py.
    rendered = " ".join(getattr(el, "value", "") or "" for el in (*at.main, *at.sidebar))
    assert "deliverable checklist" in rendered.lower(), (
        f"home page markers not found in default render; got: {rendered[:300]!r}"
    )
