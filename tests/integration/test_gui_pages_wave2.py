"""§7.7 GUI smoke tests for pages 6-10 (wave 2).

Mirrors ``test_gui_smoke.py``: each test boots the Streamlit page
through ``AppTest.from_file`` with a short timeout, asserts the run
completed without raising an exception, and confirms at least one
visible element was emitted. We do **not** assert on copy — only that
the page can be rendered headless, which is the smoke contract.

Pages exercised:

* ``06_compare.py``         — REINFORCE vs A2C comparison view
* ``07_recommend.py``       — Recommendation page
* ``08_action_masking.py``  — Action-masking explainer
* ``09_theory.py``          — Theory write-up
* ``10_discussion.py``      — §1.4 discussion / lessons learned

If a page file is missing on disk (e.g. still in flight on a feature
branch) the corresponding test is skipped rather than failed, so the
suite never blocks an in-progress GUI build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# AppTest lives under ``streamlit.testing.v1`` since Streamlit 1.28.
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_PAGES_DIR = Path(__file__).resolve().parents[2] / "src" / "gui" / "pages"
_RUN_TIMEOUT_SEC = 10.0

WAVE2_PAGES: tuple[str, ...] = (
    "06_compare.py",
    "07_recommend.py",
    "08_action_masking.py",
    "09_theory.py",
    "10_discussion.py",
)


def _run_page(filename: str) -> AppTest:
    """Boot a Streamlit page headless and return the executed AppTest."""
    page_path = _PAGES_DIR / filename
    if not page_path.exists():
        pytest.skip(f"page not yet implemented: {filename}")
    at = AppTest.from_file(str(page_path), default_timeout=_RUN_TIMEOUT_SEC)
    at.run()
    return at


def _assert_clean_render(at: AppTest, filename: str) -> None:
    """Common assertions: no exception was raised and *some* element rendered."""
    assert not at.exception, (
        f"{filename} raised during headless render: {[str(e.value) for e in at.exception]}"
    )
    rendered_any = bool(at.markdown or at.title or at.header or at.subheader or at.caption)
    assert rendered_any, f"{filename} produced no visible elements"


@pytest.mark.parametrize("filename", WAVE2_PAGES)
def test_wave2_page_renders_without_error(filename: str) -> None:
    """Each wave-2 page renders headless without raising."""
    at = _run_page(filename)
    _assert_clean_render(at, filename)


def test_wave2_page_filenames_match_app_registry() -> None:
    """Guard against drift between this smoke list and ``src/gui/app.py``."""
    app_py = (_PAGES_DIR.parent / "app.py").read_text(encoding="utf-8")
    for filename in WAVE2_PAGES:
        assert filename in app_py, (
            f"{filename} not referenced in src/gui/app.py — "
            "wave-2 smoke list is out of sync with the navigation registry."
        )
