"""§7.7 Theory page — renders docs/THEORY.md with KaTeX-rendered equations.

Parses ``docs/THEORY.md`` for ``$$..$$`` display-math blocks and emits an
alternating sequence of ``st.markdown`` (prose) and ``st.latex`` (equation)
calls so the brief's twelve load-bearing equations render natively in
Streamlit's KaTeX renderer rather than as raw ``$$..$$`` strings. Each
"How this maps to ``src/``" paragraph is highlighted as a small inline
``st.info`` block so the math ↔ code bridge stays visually distinct.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from src.gui.theme import header

THEORY_PATH = Path(__file__).resolve().parents[3] / "docs" / "THEORY.md"

# (anchor_id, sidebar_label) — anchors match Streamlit's auto-slug for headings.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("1-2-objective-function-j-theta-what-we-are-maximising", "§1 Policy / Objective"),
    ("2-4-vanilla-reinforce-update", "§2 REINFORCE"),
    ("3-2-baseline-subtracted-update-eq-4", "§3 Baseline"),
    ("4-2-reward-to-go-eq-7", "§4 Reward-to-go"),
    ("5-4-advantage-function-eq-8", "§5 A2C (Advantage / TD / Actor / Critic)"),
    ("7-4-reward-eq-15-the-domain-specific-shaping", "§6 Reward"),
    ("7-3-lstm-world-model-eq-14", "§7 LSTM World Model"),
)

EQ_BLOCK_RE = re.compile(r"\$\$\s*(.*?)\s*\$\$", re.DOTALL)
MAPS_RE = re.compile(r"^\*\*How this maps to `src/`\.\*\*\s*(.+)$", re.DOTALL)


def _load_theory() -> str:
    """Read THEORY.md from disk; show a friendly error if missing."""
    if not THEORY_PATH.exists():
        st.error(f"THEORY.md not found at {THEORY_PATH}")
        st.stop()
    return THEORY_PATH.read_text(encoding="utf-8")


def _split_segments(md: str) -> list[tuple[str, str]]:
    """Split the markdown into ('prose'|'latex', text) chunks in order."""
    segments: list[tuple[str, str]] = []
    cursor = 0
    for match in EQ_BLOCK_RE.finditer(md):
        prose = md[cursor : match.start()]
        if prose.strip():
            segments.append(("prose", prose))
        segments.append(("latex", match.group(1)))
        cursor = match.end()
    tail = md[cursor:]
    if tail.strip():
        segments.append(("prose", tail))
    return segments


def _render_prose(chunk: str) -> None:
    """Render a prose chunk, lifting any 'How this maps to src/' paragraph
    into an ``st.info`` callout so the bridge to code is visually distinct.
    """
    paragraphs = re.split(r"\n{2,}", chunk)
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        maps_match = MAPS_RE.match(stripped)
        if maps_match:
            st.info(f"**How this maps to `src/`** — {maps_match.group(1).strip()}")
        else:
            st.markdown(stripped, unsafe_allow_html=False)


def _render_sidebar() -> None:
    """Sidebar table of contents — one anchor link per top-level equation §."""
    with st.sidebar:
        st.markdown("### Theory sections")
        for anchor, label in SECTIONS:
            st.markdown(f"- [{label}](#{anchor})")
        st.divider()
        st.caption(
            "Equations render via Streamlit's built-in KaTeX. Source of truth: "
            "`docs/THEORY.md` — edits here have no effect."
        )


def main() -> None:
    """Page entry point invoked by Streamlit's multipage router."""
    header(
        "Theory — REINFORCE / A2C / LSTM World Model",
        "docs/THEORY.md rendered inline with KaTeX (brief equations 1–17)",
        icon="📐",
    )
    _render_sidebar()
    md = _load_theory()
    for kind, text in _split_segments(md):
        if kind == "latex":
            st.latex(text)
        else:
            _render_prose(text)


main()
