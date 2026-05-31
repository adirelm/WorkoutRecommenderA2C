"""Smoke tests for src/gui/components.py (§9 reusable widgets).

Streamlit primitives require a script-run context, so each component is
exercised through ``streamlit.testing.v1.AppTest`` — that gives us a real
runtime without spinning up a browser. The assertions are deliberately
lightweight: every widget must render without raising and must surface
at least one markdown / metric / latex element so a regression in the
component (e.g. an unclosed f-string) fails the suite loudly.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _run(body: str) -> AppTest:
    at = AppTest.from_string(body)
    at.run(timeout=10)
    assert not at.exception, f"AppTest raised: {at.exception}"
    return at


def test_hero_renders_title_and_subtitle() -> None:
    at = _run("from src.gui.components import hero\nhero('A2C', 'Workout Recommender', icon='X')\n")
    blob = " ".join(md.value for md in at.markdown)
    assert "A2C" in blob and "Workout Recommender" in blob


def test_metric_row_handles_empty_and_populated() -> None:
    at_empty = _run("from src.gui.components import metric_row\nmetric_row({})\n")
    assert len(at_empty.metric) == 0

    at_full = _run(
        "from src.gui.components import metric_row\n"
        "metric_row({'reward': '12.3', 'episodes': '500', 'seed': '7'})\n"
    )
    labels = {m.label for m in at_full.metric}
    assert labels == {"reward", "episodes", "seed"}


def test_info_card_includes_title_and_body() -> None:
    at = _run(
        "from src.gui.components import info_card\ninfo_card('Setup', 'Run uv sync --dev before training.')\n"
    )
    blob = " ".join(md.value for md in at.markdown)
    assert "Setup" in blob
    assert "uv sync" in blob


def test_deliverable_checklist_marks_done_and_pending() -> None:
    at = _run(
        "from src.gui.components import deliverable_checklist\n"
        "deliverable_checklist([\n"
        "    ('Policy net', True, 'https://example.com/p'),\n"
        "    ('Critic net', False, ''),\n"
        "])\n"
    )
    blob = " ".join(md.value for md in at.markdown)
    assert "Policy net" in blob and "Critic net" in blob
    assert "✅" in blob and "⬜" in blob
    assert "https://example.com/p" in blob


def test_latex_block_with_and_without_caption() -> None:
    at_caption = _run(
        "from src.gui.components import latex_block\nlatex_block(r'A_t = R_t - V(s_t)', label='Advantage')\n"
    )
    assert any("Advantage" in c.value for c in at_caption.caption)
    assert len(at_caption.latex) == 1
    assert "A_t" in at_caption.latex[0].value

    at_plain = _run("from src.gui.components import latex_block\nlatex_block(r'\\pi_\\theta(a|s)')\n")
    assert len(at_plain.caption) == 0
    assert len(at_plain.latex) == 1


def test_page_footer_renders_repo_link_and_section() -> None:
    at = _run("from src.gui.components import page_footer\npage_footer()\n")
    assert len(at.divider) >= 1
    blob = " ".join(md.value for md in at.markdown)
    assert "Bar-Ilan" in blob
    assert "§1.4" in blob
    assert "github.com" in blob
