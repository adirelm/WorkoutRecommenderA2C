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
import importlib.util
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8501
HOST = "localhost"
VIEWPORT = {"width": 1280, "height": 800}
READY_TIMEOUT_S = 45
PAGE_SETTLE_S = 2.5

# (page_id, sidebar nav label) — labels mirror src/gui/app.py PAGES titles.
PAGES: list[tuple[str, str]] = [
    ("home", "Home"),
    ("data", "Data"),
    ("lstm", "LSTM World Model"),
    ("reinforce", "REINFORCE"),
    ("a2c", "A2C"),
    ("compare", "Compare"),
    ("recommend", "Recommend"),
    ("action_masking", "Action Masking"),
    ("theory", "Theory"),
    ("discussion", "Discussion"),
]


def _have_playwright() -> bool:
    return importlib.util.find_spec("playwright") is not None


def _wait_for_server(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _start_streamlit(repo_root: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/gui/app.py",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.runOnSave",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _capture(out_dir: Path, selected: list[tuple[str, str]]) -> int:
    from playwright.sync_api import sync_playwright  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()
        page.goto(f"http://{HOST}:{PORT}", wait_until="networkidle")
        time.sleep(PAGE_SETTLE_S)
        for page_id, label in selected:
            try:
                link = page.get_by_role("link", name=label).first
                link.click(timeout=5000)
                page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(PAGE_SETTLE_S)
                target = out_dir / f"gui_{page_id}.png"
                page.screenshot(path=str(target), full_page=False)
                saved += 1
                print(f"  saved {target.name}")
            except Exception as exc:
                print(f"  skipped {page_id}: {exc}")
        browser.close()
    return saved


def _parse_pages(arg: str | None) -> list[tuple[str, str]]:
    if not arg:
        return PAGES
    wanted = {p.strip().lower() for p in arg.split(",") if p.strip()}
    selected = [pl for pl in PAGES if pl[0] in wanted]
    if not selected:
        print(f"⚠ no matching pages for --pages={arg}; valid: {[p for p, _ in PAGES]}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", help="Comma-separated page_ids (e.g. home,a2c).")
    args = parser.parse_args()

    if not _have_playwright():
        print("⚠ playwright not installed; skipping screenshot capture (best-effort).")
        print("  install with: uv add --dev playwright && uv run playwright install chromium")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "assets"
    selected = _parse_pages(args.pages)
    if not selected:
        return 0

    print(f"→ starting streamlit on http://{HOST}:{PORT}")
    proc = _start_streamlit(repo_root)
    try:
        if not _wait_for_server(f"http://{HOST}:{PORT}", READY_TIMEOUT_S):
            print(f"⚠ streamlit did not become ready within {READY_TIMEOUT_S}s; aborting.")
            return 0
        print(f"→ capturing {len(selected)} page(s) to {out_dir.relative_to(repo_root)}/")
        saved = _capture(out_dir, selected)
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
