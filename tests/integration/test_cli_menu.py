"""Tests for src/cli/menu.py (PRD §F17). Uses StringIO to drive the interactive loop.

The launch-gui and full-dispatch tests live in test_cli_menu_extra.py
so this file stays ≤150 LOC (CLAUDE.md §1)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from src.cli.menu import CLIMenu
from src.sdk.sdk import WorkoutSDK


@pytest.fixture
def sdk() -> WorkoutSDK:
    return WorkoutSDK()


@pytest.fixture
def stdout() -> io.StringIO:
    return io.StringIO()


def _menu(sdk, stdin_text: str, stdout: io.StringIO) -> CLIMenu:
    return CLIMenu(sdk=sdk, stdin=io.StringIO(stdin_text), stdout=stdout)


def test_render_main_menu_lists_all_verbs(sdk, stdout):
    menu = _menu(sdk, "", stdout)
    rendered = menu.render_main_menu()
    for verb in (
        "prepare-data",
        "train-world-model",
        "train-reinforce",
        "train-a2c",
        "compare",
        "recommend",
        "launch-gui",
    ):
        assert verb in rendered, f"verb {verb!r} missing from menu"
    # plus the exit option
    assert "0." in rendered
    assert "exit" in rendered
    assert "7." in rendered
    assert "Streamlit dashboard (Phase 9)" in rendered


def test_handle_choice_zero_exits(sdk, stdout):
    menu = _menu(sdk, "", stdout)
    assert menu.handle_choice("0") is False
    assert "Goodbye" in stdout.getvalue()


def test_handle_choice_invalid_prints_error(sdk, stdout):
    menu = _menu(sdk, "", stdout)
    keep_going = menu.handle_choice("99")
    assert keep_going is True
    out = stdout.getvalue()
    assert "[error]" in out
    assert "ValueError" in out


def test_handle_choice_non_numeric_does_not_crash(sdk, stdout):
    menu = _menu(sdk, "", stdout)
    assert menu.handle_choice("hello") is True
    assert "[error]" in stdout.getvalue()


def test_run_with_immediate_exit_returns_zero(sdk, stdout):
    menu = _menu(sdk, "0\n", stdout)
    assert menu.run() == 0
    assert "Goodbye" in stdout.getvalue()


def test_run_with_eof_returns_zero(sdk, stdout):
    menu = _menu(sdk, "", stdout)  # empty stdin → immediate EOF
    assert menu.run() == 0
    assert "[eof]" in stdout.getvalue()


def test_train_reinforce_verb_calls_sdk(sdk, stdout):
    menu = _menu(sdk, "", stdout)
    fake_handle = MagicMock(algorithm="REINFORCE", final_reward=1.23, episodes_trained=10)
    with patch.object(sdk, "train_reinforce", return_value=(fake_handle, MagicMock())) as m:
        keep_going = menu.handle_choice("3")
        assert keep_going is True
        m.assert_called_once()
    out = stdout.getvalue()
    assert "train-reinforce" in out
    assert "REINFORCE" in out
    assert "1.23" in out
