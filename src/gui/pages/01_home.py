"""§7.7 deliverable status home page (Phase 9 GUI)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import streamlit as st

from src.gui.components import (
    deliverable_checklist,
    hero,
    info_card,
    page_footer,
)
from src.gui.state import GUIState

REPO_URL = "https://github.com/AdirElmakyes/Assignment3-WorkoutRecommenderA2C"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# (label, relative-path, optional-link). Link is "" if no public artefact.
_DELIVERABLES: tuple[tuple[str, str, str], ...] = (
    ("LSTM world-model loss chart", "results/figures/lstm_loss.png", ""),
    ("REINFORCE reward curve", "results/figures/reinforce_rewards.png", ""),
    ("A2C training chart", "results/figures/a2c_training.png", ""),
    ("REINFORCE vs A2C comparison", "results/figures/comparison.png", ""),
    ("Executed analysis notebook", "notebooks/analysis_executed.ipynb", ""),
    ("Source notebook", "notebooks/analysis.ipynb", ""),
)


def _exists(rel: str) -> bool:
    """Return True iff ``rel`` (relative to repo root) exists on disk."""
    return (_REPO_ROOT / rel).exists()


def _ci_badge() -> str:
    """Return the latest CI conclusion via ``gh`` or ``"live"`` on failure."""
    if shutil.which("gh") is None:
        return "live"
    try:
        out = subprocess.run(
            ["gh", "run", "list", "--limit", "1", "--json", "conclusion"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
            cwd=str(_REPO_ROOT),
        )
    except (subprocess.SubprocessError, OSError):
        return "live"
    if out.returncode != 0:
        return "live"
    import json

    try:
        payload = json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return "live"
    if not payload:
        return "live"
    return str(payload[0].get("conclusion") or "live")


def _checklist_items() -> list[tuple[str, bool, str]]:
    """Materialise the deliverable rows the checklist component expects."""
    return [(label, _exists(rel), link) for label, rel, link in _DELIVERABLES]


def _quick_start(state: GUIState) -> None:
    """Four quick-start buttons that record the intended next page."""
    cols = st.columns(4)
    targets = (
        (
            "Go to Data",
            "data",
            "Jump to §7.2 — data ingestion (PHUL trainee loader + state heatmap).",
        ),
        (
            "Go to REINFORCE",
            "reinforce",
            "Jump to §7.4 — Monte-Carlo policy gradient with running-mean baseline.",
        ),
        (
            "Go to A2C",
            "a2c",
            "Jump to §7.5 — synchronous advantage actor-critic.",
        ),
        (
            "Go to Theory",
            "theory",
            "Show the underlying equations + algorithm walkthrough.",
        ),
    )
    for col, (label, target, tip) in zip(cols, targets):
        with col:
            if st.button(
                label,
                use_container_width=True,
                key=f"qs_{target}",
                help=tip,
            ):
                state.set("next_page", target)
                st.toast(f"Open the '{target}' page in the sidebar.", icon="➡️")


def render() -> None:
    """Render the home page in full."""
    state = GUIState("home")
    hero(
        title="WorkoutRecommenderA2C",
        subtitle="LSTM world-model + REINFORCE & A2C policies on workout logs.",
        icon="🏋️",
    )
    info_card(
        "Assignment context",
        "Bar-Ilan Vibe-Coding & RL Workshop · Assignment 3 (§1.4 architect "
        "decisions, §7.7 deliverables). Use the sidebar to walk the pipeline "
        "from raw data → world-model → REINFORCE → A2C → comparison.",
    )

    st.subheader("§7.7 — Deliverable checklist")
    items = _checklist_items()
    done = sum(1 for _, ok, _ in items if ok)
    st.caption(f"{done}/{len(items)} artefacts present on disk.")
    deliverable_checklist(items)

    st.subheader("Quick start")
    _quick_start(state)

    st.subheader("Build status")
    badge = _ci_badge()
    colour = {"success": "🟢", "failure": "🔴", "live": "🟡"}.get(badge, "⚪")
    st.markdown(f"{colour} **CI:** `{badge}` · [repository]({REPO_URL})")

    page_footer()


render()
