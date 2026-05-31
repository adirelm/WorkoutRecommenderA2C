"""§7.7 lightweight GUI accessibility checks (titles, buttons, contrast, sidebar).

Audits: every page renders a hero/title, no button has empty label text,
the Bar-Ilan primary blue ``#003D7A`` clears WCAG-AA contrast (≥ 4.5:1)
against the light page background ``#F5F7FA``, and every sidebar entry
in ``app.py`` declares BOTH an icon and a title. CLAUDE.md §1 (≤150 LOC).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_REPO = Path(__file__).resolve().parents[2]
_PAGES_DIR = _REPO / "src" / "gui" / "pages"
_APP_PY = _REPO / "src" / "gui" / "app.py"

# Mirrors src/gui/theme.py + src/gui/components.py.
_PRIMARY_BLUE, _LIGHT_BG, _WCAG_AA = "#003D7A", "#F5F7FA", 4.5

_PAGE_FILES: tuple[str, ...] = (
    "01_home.py",
    "02_data.py",
    "03_lstm.py",
    "04_reinforce.py",
    "05_a2c.py",
    "06_compare.py",
    "07_recommend.py",
    "08_action_masking.py",
    "09_theory.py",
    "10_discussion.py",
)


def _luminance(hex_str: str) -> float:
    """WCAG 2.1 relative luminance for an sRGB hex color."""
    s = hex_str.lstrip("#")
    chans = [int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg: str, bg: str) -> float:
    l1, l2 = _luminance(fg), _luminance(bg)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _run(filename: str) -> AppTest:
    path = _PAGES_DIR / filename
    if not path.exists():
        pytest.skip(f"page not yet implemented: {filename}")
    at = AppTest.from_file(str(path), default_timeout=15.0).run()
    assert not at.exception, f"{filename} raised: {[str(e.value) for e in at.exception]}"
    return at


@pytest.mark.parametrize("filename", _PAGE_FILES)
def test_every_page_has_a_title(filename: str) -> None:
    """Page is 'titled' via st.title/header/subheader or H1/H2 markdown/HTML."""
    at = _run(filename)
    if at.title or at.header or at.subheader:
        return
    body = "\n".join(b.value for b in at.markdown).lower()
    has_hero = "<h1" in body or "<h2" in body
    has_md_title = bool(re.search(r"(?m)^\s{0,3}#{1,2}\s+\S", body))
    assert has_hero or has_md_title, f"{filename} has no visible title/hero"


@pytest.mark.parametrize("filename", _PAGE_FILES)
def test_no_empty_button_labels(filename: str) -> None:
    """Every Streamlit button must carry non-whitespace label text."""
    at = _run(filename)
    for btn in at.button:
        assert btn.label and btn.label.strip(), f"{filename} has empty button label: {btn!r}"


def test_primary_blue_meets_wcag_aa_on_light_bg() -> None:
    """Bar-Ilan primary blue on the light page bg clears AA (≥ 4.5:1)."""
    ratio = _contrast(_PRIMARY_BLUE, _LIGHT_BG)
    assert ratio >= _WCAG_AA, f"{_PRIMARY_BLUE} on {_LIGHT_BG} is {ratio:.2f}:1, below AA ({_WCAG_AA}:1)"


def test_sidebar_entries_have_icon_and_title() -> None:
    """Every ``st.Page(...)`` call in app.py declares BOTH icon and title."""
    src = _APP_PY.read_text(encoding="utf-8")
    page_calls = re.findall(r"st\.Page\([^)]*\)", src)
    assert page_calls, "no st.Page(...) entries found in app.py"
    for call in page_calls:
        title_m = re.search(r"title=([\"'])(.+?)\1", call)
        icon_m = re.search(r"icon=([\"'])(.+?)\1", call)
        assert title_m and title_m.group(2).strip(), f"missing title: {call}"
        assert icon_m and icon_m.group(2).strip(), f"missing icon: {call}"
