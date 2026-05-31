"""Tests for main.py entry point (PRD F17)."""

from __future__ import annotations

import io
from unittest.mock import patch

import main as main_module


def test_main_returns_zero_on_immediate_exit() -> None:
    with patch("sys.stdin", io.StringIO("0\n")):
        rc = main_module.main()
    assert rc == 0


def test_main_set_global_seed_called() -> None:
    with (
        patch("main.set_global_seed") as mock_seed,
        patch("sys.stdin", io.StringIO("0\n")),
    ):
        main_module.main()
    mock_seed.assert_called_once_with(42)
