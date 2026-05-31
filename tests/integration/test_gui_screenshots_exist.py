"""Verify docs/assets contains the expected GUI screenshot PNGs (CLAUDE.md §1, ≤150 LOC).

Naming convention mirrors scripts/capture_gui_screenshots.py: gui_<page_id>.png.
If screenshots are missing (capture is best-effort / optional), tests xfail with
a hint pointing at the capture script rather than hard-failing the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "docs" / "assets"

# Mirrors PAGES in scripts/capture_gui_screenshots.py.
EXPECTED_PAGE_IDS: list[str] = [
    "home",
    "data",
    "lstm",
    "reinforce",
    "a2c",
    "compare",
    "recommend",
    "action_masking",
    "theory",
    "discussion",
]

MIN_BYTES = 10 * 1024  # 10 KB sanity floor.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"  # first 8 bytes of every valid PNG.
SKIP_REASON = "screenshots optional; capture via scripts/capture_gui_screenshots.py"


def _screenshot_path(page_id: str) -> Path:
    return ASSETS_DIR / f"gui_{page_id}.png"


def test_assets_dir_exists() -> None:
    """docs/assets/ itself must exist (created by the capture script)."""
    assert ASSETS_DIR.is_dir(), f"missing directory: {ASSETS_DIR}"


def test_expected_page_count() -> None:
    """Guard against drift between this test and the capture script."""
    assert len(EXPECTED_PAGE_IDS) == 10, "expected exactly 10 GUI pages"


@pytest.mark.parametrize("page_id", EXPECTED_PAGE_IDS)
def test_screenshot_exists(page_id: str) -> None:
    path = _screenshot_path(page_id)
    if not path.exists():
        pytest.xfail(SKIP_REASON)
    assert path.is_file(), f"not a regular file: {path}"


@pytest.mark.parametrize("page_id", EXPECTED_PAGE_IDS)
def test_screenshot_min_size(page_id: str) -> None:
    path = _screenshot_path(page_id)
    if not path.exists():
        pytest.xfail(SKIP_REASON)
    size = path.stat().st_size
    assert size >= MIN_BYTES, f"{path.name} is {size} bytes; expected ≥{MIN_BYTES}"


@pytest.mark.parametrize("page_id", EXPECTED_PAGE_IDS)
def test_screenshot_png_magic(page_id: str) -> None:
    path = _screenshot_path(page_id)
    if not path.exists():
        pytest.xfail(SKIP_REASON)
    with path.open("rb") as fh:
        header = fh.read(8)
    assert header == PNG_MAGIC, f"{path.name} is not a valid PNG (header={header!r})"
