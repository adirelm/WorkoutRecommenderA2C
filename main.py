"""Entry point — delegates to the CLI menu (PRD F17)."""

from __future__ import annotations

import sys

from src.cli.menu import CLIMenu
from src.sdk.sdk import WorkoutSDK
from src.utils.seeding import set_global_seed


def main(argv: list[str] | None = None) -> int:
    set_global_seed(42)
    sdk = WorkoutSDK(seed=42)
    menu = CLIMenu(sdk)
    return menu.run()


if __name__ == "__main__":
    sys.exit(main())
