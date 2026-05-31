"""§7.7 Page navigation registry + deliverable checklist data.

Page modules under ``src/gui/pages/`` are discovered by Streamlit's
multipage router via filename ordering (``01_home.py`` … ``08_about.py``).
This registry is the single source of truth for sidebar icons / titles
so the navigation strip and the deliverable checklist on the Home page
stay in lockstep with the actual file tree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageMeta:
    """One row of the navigation strip."""

    icon: str
    title: str
    module: str  # filename stem under src/gui/pages/, e.g. "01_home"
    brief_section: str  # brief §X.Y this page satisfies, e.g. "§7.4"


PAGES: tuple[PageMeta, ...] = (
    PageMeta("🏠", "Home", "01_home", "§7.7"),
    PageMeta("📊", "Data & Environment", "02_data_env", "§7.3"),
    PageMeta("🧠", "LSTM World Model", "03_lstm_world_model", "§7.3"),
    PageMeta("🎯", "REINFORCE", "04_reinforce", "§7.4"),
    PageMeta("⚡", "A2C (Actor-Critic)", "05_a2c", "§7.5"),
    PageMeta("📈", "Comparison", "06_comparison", "§7.6"),
    PageMeta("💡", "Recommendation", "07_recommendation", "§7.5"),
    PageMeta("📝", "Discussion", "08_discussion", "§1.4"),
)


@dataclass(frozen=True)
class DeliverableItem:
    """One row of the brief §7.7 deliverable checklist on the Home page."""

    label: str
    satisfied_by_file: str  # repo-relative path proving the deliverable exists


DELIVERABLES: tuple[DeliverableItem, ...] = (
    DeliverableItem("PRD + ADRs + THEORY.md", "docs/PRD.md"),
    DeliverableItem("Synthetic trainee + WorkoutEnv (§7.3)", "src/env/workout_env.py"),
    DeliverableItem("Frozen LSTM world model (§7.3)", "src/model/lstm_world_model.py"),
    DeliverableItem("REINFORCE trainer (§7.4)", "src/services/reinforce_trainer.py"),
    DeliverableItem("A2C trainer (§7.5)", "src/services/a2c_trainer.py"),
    DeliverableItem("Comparison + mean±std plots (§7.6)", "src/services/comparator.py"),
    DeliverableItem("Streamlit GUI (§7.7)", "src/gui/app.py"),
    DeliverableItem("Tests ≥ 85% coverage (CLAUDE.md §2)", "tests/"),
)
