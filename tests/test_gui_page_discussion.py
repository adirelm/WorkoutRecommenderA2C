"""GUI §7.6 — Discussion page renders Q1-Q5 + §7.6.1 Action-Masking card.

The page at ``src/gui/pages/10_discussion.py`` builds five ``st.expander``
cards (one per discussion question) and a sixth section for the §7.6.1
action-masking proposal. Each card is bilingual: English ``info_card`` body
plus a Hebrew RTL block (``direction:rtl``). We capture every Streamlit
primitive via a stub, then assert (a) all five Q-expanders, (b) the
action-masking subsection, and (c) Hebrew RTL handling is visible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "gui" / "pages" / "10_discussion.py"
Q_LABELS = ("Q1", "Q2", "Q3", "Q4", "Q5")
HEBREW_QS = ("ש1", "ש2", "ש3", "ש4", "ש5")


def _stubs() -> dict[str, MagicMock]:
    """Build the sys.modules patch dict for an isolated page import."""
    st = MagicMock(name="streamlit")
    st.expander.return_value.__enter__ = lambda *_: st.expander.return_value
    st.expander.return_value.__exit__ = lambda *_: False
    st.session_state = {}
    components = MagicMock(name="src.gui.components")
    state = MagicMock(name="src.gui.state")
    state.get_last_a2c_history.return_value = None
    state.GUIState.return_value.get.return_value = None
    env_state = MagicMock(name="src.env.state")
    env_state.ACTION_NAMES = ("Rest", "Push", "Pull", "Legs", "Full", "Cond", "Mob")
    return {
        "streamlit": st,
        "src.gui.components": components,
        "src.gui.charts": MagicMock(name="src.gui.charts"),
        "src.gui.state": state,
        "src.env.state": env_state,
    }


def _load_discussion_page() -> SimpleNamespace:
    """Import ``10_discussion.py`` under a fully stubbed dependency tree."""
    patches = _stubs()
    spec = importlib.util.spec_from_file_location("discussion_page_under_test", PAGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, patches):
        spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        st=patches["streamlit"],
        components=patches["src.gui.components"],
    )


@pytest.fixture
def loaded() -> SimpleNamespace:
    return _load_discussion_page()


def test_discussion_page_renders_five_question_expanders(loaded: SimpleNamespace) -> None:
    """Q1-Q5 must each open an ``st.expander`` whose label starts with ``Qn —``."""
    titles = [c.args[0] for c in loaded.st.expander.call_args_list if c.args]
    assert len(titles) >= 5, f"expected ≥5 expander cards for Q1-Q5; got {len(titles)}"
    for tag in Q_LABELS:
        assert any(t.startswith(f"{tag} —") for t in titles), (
            f"missing expander for {tag}; rendered titles were {titles!r}"
        )


def test_discussion_page_renders_action_masking_card(loaded: SimpleNamespace) -> None:
    """§7.6.1 action-masking proposal must surface as its own subsection."""
    subheaders = [c.args[0] for c in loaded.st.subheader.call_args_list if c.args]
    assert any("Action Masking" in s for s in subheaders), (
        f"expected an 'Action Masking' subheader; got {subheaders!r}"
    )
    titles = [c.args[0] for c in loaded.components.info_card.call_args_list if c.args]
    assert "Proposal" in titles, "action-masking card must info_card('Proposal', ...)"


def test_discussion_page_emits_hebrew_rtl_blocks(loaded: SimpleNamespace) -> None:
    """≥6 ``st.markdown`` calls must contain ``direction:rtl`` + Hebrew chars."""
    md = [str(c.args[0]) for c in loaded.st.markdown.call_args_list if c.args]
    rtl = [p for p in md if "direction:rtl" in p]
    assert len(rtl) >= 6, f"expected ≥6 RTL blocks (5 Qs + action-masking); got {len(rtl)}"
    joined = "\n".join(rtl)
    assert any(0x0590 <= ord(ch) <= 0x05FF for ch in joined), (
        "RTL blocks must contain Hebrew characters (U+0590..U+05FF)"
    )
    for tag in HEBREW_QS:
        assert tag in joined, f"Hebrew question label {tag!r} missing from RTL blocks"


def test_discussion_page_renders_hero_and_footer(loaded: SimpleNamespace) -> None:
    """The page must wrap content in the standard hero + page_footer chrome."""
    assert loaded.components.hero.called, "discussion page must render a hero"
    assert loaded.components.page_footer.called, "discussion page must render footer"
