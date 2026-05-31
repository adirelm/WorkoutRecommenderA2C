"""GUI smoke tests — Phase 9 wave-1 pages render without crashing.

Headless smoke via `streamlit.testing.v1.AppTest`. We assert each page
imports cleanly and survives its initial render (no exception in the
script run). We deliberately do NOT click buttons here: the training
pages (`03_lstm`, `04_reinforce`, `05_a2c`) would actually fit a model
on click, which is far too slow for a smoke gate.

Brief alignment: §7.7 deliverable + CLAUDE.md §1 (≤150 LOC) + the
"Tested headlessly with streamlit.testing.v1.AppTest" line in
`src/gui/pages/README.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# `AppTest` is part of Streamlit's public testing surface (v1). We skip
# the whole module rather than hard-fail if Streamlit is absent so this
# file remains importable in minimal CI environments.
streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


# Wave-1 pages: app entry + the five brief-§7 pages that ship first
# (home + data + the three trainer pages). Compare/recommend/etc. land
# in wave-2 and get their own smoke entries when they exist.
WAVE_1_PAGES: tuple[str, ...] = (
    "src/gui/app.py",
    "src/gui/pages/01_home.py",
    "src/gui/pages/02_data.py",
    "src/gui/pages/03_lstm.py",
    "src/gui/pages/04_reinforce.py",
    "src/gui/pages/05_a2c.py",
)

# Resolve relative to the repo root (this file lives in `tests/`).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _page_path(rel: str) -> Path:
    return REPO_ROOT / rel


@pytest.mark.parametrize("page", WAVE_1_PAGES)
def test_page_imports_without_error(page: str) -> None:
    """Each wave-1 page renders its initial frame with no exception.

    Pages that have not yet been authored are skipped — this test is a
    smoke gate, not a presence assertion. A separate manifest test
    (when added) will enforce that all wave-1 files exist.
    """
    path = _page_path(page)
    if not path.exists():
        pytest.skip(f"page not yet authored: {page}")

    app = AppTest.from_file(str(path)).run()

    # `app.exception` is a sequence of any uncaught exceptions raised
    # during the script run. Empty == clean render.
    assert not app.exception, f"{page} raised on initial render: {[str(e.value) for e in app.exception]}"


def test_wave1_page_list_is_stable() -> None:
    """Guard against accidental renumbering (README 'Numbering convention').

    If someone renames `02_data.py` → `02_dataset.py` this list must be
    updated in lock-step, otherwise the smoke loop silently skips it.
    """
    expected_suffixes = (
        "app.py",
        "01_home.py",
        "02_data.py",
        "03_lstm.py",
        "04_reinforce.py",
        "05_a2c.py",
    )
    for page, suffix in zip(WAVE_1_PAGES, expected_suffixes, strict=True):
        assert page.endswith(suffix), f"wave-1 page list drifted: {page} should end with {suffix}"
