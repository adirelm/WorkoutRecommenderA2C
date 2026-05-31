"""Helpers for capture_gui_screenshots.py — kept separate so script stays ≤150 LOC."""

from __future__ import annotations

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


def have_playwright() -> bool:
    return importlib.util.find_spec("playwright") is not None


def wait_for_server(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start_streamlit(repo_root: Path) -> subprocess.Popen:
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


def capture(out_dir: Path, selected: list[tuple[str, str]]) -> int:
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


def parse_pages(arg: str | None) -> list[tuple[str, str]]:
    if not arg:
        return PAGES
    wanted = {p.strip().lower() for p in arg.split(",") if p.strip()}
    selected = [pl for pl in PAGES if pl[0] in wanted]
    if not selected:
        print(f"⚠ no matching pages for --pages={arg}; valid: {[p for p, _ in PAGES]}")
    return selected
