"""Terminal menu for WorkoutRecommenderA2C (PRD F17, CLAUDE.md §1.4 architect/implementer split)."""

from __future__ import annotations

import sys
from typing import TextIO

from src.sdk import State
from src.sdk.sdk import WorkoutSDK

_VERBS: tuple[tuple[str, str, str], ...] = (
    ("1", "prepare-data", "Load Kaggle program (or synthetic trainee) into a logbook"),
    ("2", "train-world-model", "Train the LSTM transition model (Phase 3)"),
    ("3", "train-reinforce", "Train REINFORCE policy"),
    ("4", "train-a2c", "Train A2C agent"),
    ("5", "compare", "REINFORCE-vs-A2C side-by-side over N seeds"),
    ("6", "recommend", "Show next-day recommendation for current state"),
    ("0", "exit", "Quit the menu"),
)


class CLIMenu:
    """Numeric menu wrapping the SDK — no business logic here (CLAUDE.md §3)."""

    def __init__(
        self,
        sdk: WorkoutSDK,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.sdk = sdk
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout

    def render_main_menu(self) -> str:
        """Render the 7-line numbered menu (6 verbs + exit) as a string."""
        lines = ["", "=== WorkoutRecommenderA2C — Main Menu ==="]
        for key, verb, desc in _VERBS:
            lines.append(f"  {key}. {verb:<20s} — {desc}")
        lines.append("Select an option: ")
        return "\n".join(lines)

    def _dispatch(self, choice: str) -> str:
        if choice == "1":
            handle = self.sdk.prepare_data()
            return f"[ok] prepare-data → {handle}"
        if choice == "2":
            handle = self.sdk.train_world_model()
            return f"[ok] train-world-model → {handle}"
        if choice == "3":
            handle, _history = self.sdk.train_reinforce()
            return (
                f"[ok] train-reinforce → {handle.algorithm} "
                f"final_reward={handle.final_reward:.2f} "
                f"episodes={handle.episodes_trained}"
            )
        if choice == "4":
            handle, _history = self.sdk.train_a2c()
            return (
                f"[ok] train-a2c → {handle.algorithm} "
                f"final_reward={handle.final_reward:.2f} "
                f"episodes={handle.episodes_trained}"
            )
        if choice == "5":
            result = self.sdk.compare()
            return f"[ok] compare → {result}"
        if choice == "6":
            rec = self.sdk.recommend(State.initial())
            return f"[ok] recommend → {rec.action_name} (id={rec.action_id}) probs={rec.probs}"
        raise ValueError(f"Unknown choice: {choice!r}")

    def handle_choice(self, choice: str) -> bool:
        """Returns False to exit, True to continue."""
        choice = choice.strip()
        if choice == "0":
            self.stdout.write("Goodbye.\n")
            return False
        try:
            message = self._dispatch(choice)
            self.stdout.write(message + "\n")
        except (ValueError, RuntimeError, NotImplementedError, TypeError) as exc:
            self.stdout.write(f"[error] {type(exc).__name__}: {exc}\n")
        return True

    def run(self) -> int:
        """Interactive loop. Returns exit code (0 on normal exit)."""
        while True:
            self.stdout.write(self.render_main_menu())
            self.stdout.flush()
            line = self.stdin.readline()
            if not line:  # EOF — treat as graceful exit
                self.stdout.write("\n[eof] exiting.\n")
                return 0
            if not self.handle_choice(line):
                return 0
