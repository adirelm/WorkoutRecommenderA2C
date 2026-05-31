"""Tests for src/cli/menu.py (PRD §F17). Uses StringIO to drive the interactive loop."""

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
    ):
        assert verb in rendered, f"verb {verb!r} missing from menu"
    # plus the exit option
    assert "0." in rendered
    assert "exit" in rendered


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


def test_all_six_verbs_dispatch_to_sdk(stdout):
    mock_sdk = MagicMock(spec=WorkoutSDK)
    mock_sdk.prepare_data.return_value = "LB"
    mock_sdk.train_world_model.return_value = "WM"
    fake_r = MagicMock(algorithm="REINFORCE", final_reward=0.5, episodes_trained=10)
    fake_a = MagicMock(algorithm="A2C", final_reward=0.7, episodes_trained=10)
    mock_sdk.train_reinforce.return_value = (fake_r, MagicMock())
    mock_sdk.train_a2c.return_value = (fake_a, MagicMock())
    mock_sdk.compare.return_value = {"winner": "A2C"}
    mock_sdk.recommend.return_value = MagicMock(action_name="REST", action_id=0, probs=[0.5, 0.5])

    menu = CLIMenu(sdk=mock_sdk, stdin=io.StringIO(""), stdout=stdout)
    for choice in ("1", "2", "3", "4", "5", "6"):
        assert menu.handle_choice(choice) is True, f"choice {choice} failed: {stdout.getvalue()}"

    mock_sdk.prepare_data.assert_called_once()
    mock_sdk.train_world_model.assert_called_once()
    mock_sdk.train_reinforce.assert_called_once()
    mock_sdk.train_a2c.assert_called_once()
    mock_sdk.compare.assert_called_once()
    mock_sdk.recommend.assert_called_once()


def test_runtime_error_caught_gracefully(stdout):
    mock_sdk = MagicMock(spec=WorkoutSDK)
    mock_sdk.train_a2c.side_effect = RuntimeError("CUDA OOM")
    menu = CLIMenu(sdk=mock_sdk, stdin=io.StringIO(""), stdout=stdout)
    assert menu.handle_choice("4") is True
    assert "[error]" in stdout.getvalue()
    assert "RuntimeError" in stdout.getvalue()
    assert "CUDA OOM" in stdout.getvalue()
