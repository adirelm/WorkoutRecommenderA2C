"""Best-effort Playwright screenshotter for the Streamlit GUI (CLAUDE.md §1, ≤150 LOC).

Spins up `streamlit run src/gui/app.py` on port 8501, waits for readiness,
then drives a headless Chromium across all 10 pages and saves PNGs to
docs/assets/gui_<page_id>.png at 1280×800. Skips silently if Playwright is
not installed — this is a documentation helper, not a CI gate.

Usage:
    uv run python scripts/capture_gui_screenshots.py
    uv run python scripts/capture_gui_screenshots.py --pages home,a2c,compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _capture_helpers import (
    HOST,
    PORT,
    READY_TIMEOUT_S,
    capture,
    have_playwright,
    parse_pages,
    start_streamlit,
    wait_for_server,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", help="Comma-separated page_ids (e.g. home,a2c).")
    args = parser.parse_args()

    if not have_playwright():
        print("⚠ playwright not installed; skipping screenshot capture (best-effort).")
        print("  install with: uv add --dev playwright && uv run playwright install chromium")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "assets"
    selected = parse_pages(args.pages)
    if not selected:
        return 0

    print(f"→ starting streamlit on http://{HOST}:{PORT}")
    proc = start_streamlit(repo_root)
    try:
        if not wait_for_server(f"http://{HOST}:{PORT}", READY_TIMEOUT_S):
            print(f"⚠ streamlit did not become ready within {READY_TIMEOUT_S}s; aborting.")
            return 0
        print(f"→ capturing {len(selected)} page(s) to {out_dir.relative_to(repo_root)}/")
        saved = capture(out_dir, selected)
        print(f"✓ wrote {saved}/{len(selected)} screenshot(s)")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
