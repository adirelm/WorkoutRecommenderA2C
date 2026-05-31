"""GUI §7.7 — Theory page renders ≥10 KaTeX equation blocks + section anchors.

The page module ``src/gui/pages/09_theory.py`` parses ``docs/THEORY.md`` and
emits ``st.latex`` for every ``$$..$$`` display-math block. We load the page
under a stubbed ``streamlit`` so the top-level ``main()`` call is captured,
then assert (a) ≥10 KaTeX renders happened and (b) every SECTIONS anchor
declared by the page module is wired into the sidebar TOC.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PAGE_PATH = Path(__file__).resolve().parents[2] / "src" / "gui" / "pages" / "09_theory.py"


def _load_theory_page() -> SimpleNamespace:
    """Import ``09_theory.py`` under a fully-stubbed Streamlit + theme."""
    st_stub = MagicMock(name="streamlit")
    st_stub.sidebar.__enter__ = lambda *_: st_stub.sidebar
    st_stub.sidebar.__exit__ = lambda *_: False
    theme_stub = MagicMock(name="src.gui.theme")

    spec = importlib.util.spec_from_file_location("theory_page_under_test", PAGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {"streamlit": st_stub, "src.gui.theme": theme_stub}):
        spec.loader.exec_module(module)

    return SimpleNamespace(module=module, st=st_stub, theme=theme_stub)


@pytest.fixture
def loaded() -> SimpleNamespace:
    return _load_theory_page()


def test_theory_page_renders_at_least_ten_equation_blocks(loaded: SimpleNamespace) -> None:
    """≥10 ``st.latex`` calls — the brief's twelve load-bearing equations."""
    latex_calls = loaded.st.latex.call_args_list
    assert len(latex_calls) >= 10, f"expected ≥10 KaTeX equation blocks; got {len(latex_calls)}"
    for call in latex_calls:
        payload = call.args[0] if call.args else ""
        assert isinstance(payload, str) and payload.strip(), (
            "every st.latex call must carry a non-empty equation string"
        )


def test_theory_page_declares_section_anchors(loaded: SimpleNamespace) -> None:
    """The page exposes SECTIONS — at least one anchor per top-level equation §."""
    sections = getattr(loaded.module, "SECTIONS", ())
    assert len(sections) >= 5, f"need ≥5 anchored sections, got {len(sections)}"
    for entry in sections:
        anchor, label = entry
        assert anchor and isinstance(anchor, str), "anchor must be non-empty str"
        assert label and isinstance(label, str), "label must be non-empty str"
        assert " " not in anchor, "anchors must be url-safe (no spaces)"


def test_theory_page_section_anchors_wired_into_sidebar(loaded: SimpleNamespace) -> None:
    """Every SECTIONS anchor must appear in some ``st.markdown`` sidebar link."""
    sections = loaded.module.SECTIONS
    rendered = "\n".join(str(call.args[0]) for call in loaded.st.markdown.call_args_list if call.args)
    for anchor, _label in sections:
        assert f"#{anchor}" in rendered, f"anchor #{anchor} missing from sidebar TOC"


def test_theory_page_invokes_header_with_icon(loaded: SimpleNamespace) -> None:
    """The page must call ``theme.header(...)`` once with a non-empty title."""
    assert loaded.theme.header.called, "theory page must render a hero/header"
    args, kwargs = loaded.theme.header.call_args
    title = args[0] if args else kwargs.get("title", "")
    assert "Theory" in title, f"header title should mention Theory; got {title!r}"
