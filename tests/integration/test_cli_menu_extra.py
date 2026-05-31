"""Extra tests for src/cli/menu.py — split from test_cli_menu.py to stay ≤150 LOC."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from src.cli.menu import CLIMenu
from src.sdk.sdk import WorkoutSDK


def test_all_six_verbs_dispatch_to_sdk():
    stdout = io.StringIO()
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


def test_runtime_error_caught_gracefully():
    stdout = io.StringIO()
    mock_sdk = MagicMock(spec=WorkoutSDK)
    mock_sdk.train_a2c.side_effect = RuntimeError("CUDA OOM")
    menu = CLIMenu(sdk=mock_sdk, stdin=io.StringIO(""), stdout=stdout)
    assert menu.handle_choice("4") is True
    assert "[error]" in stdout.getvalue()
    assert "RuntimeError" in stdout.getvalue()
    assert "CUDA OOM" in stdout.getvalue()


def test_launch_gui_verb_invokes_streamlit():
    stdout = io.StringIO()
    sdk = WorkoutSDK()
    menu = CLIMenu(sdk=sdk, stdin=io.StringIO(""), stdout=stdout)
    with patch("src.cli.menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        keep_going = menu.handle_choice("7")
        assert keep_going is True
        mock_run.assert_called_once_with(
            [
                "uv",
                "run",
                "streamlit",
                "run",
                "src/gui/app.py",
                "--server.headless",
                "false",
            ],
            check=False,
        )
    out = stdout.getvalue()
    assert "launch-gui" in out
    assert "streamlit exited" in out


def test_launch_gui_continues_loop_after_streamlit_exits():
    """After streamlit exits, the CLI loop must keep going (returns True)."""
    stdout = io.StringIO()
    sdk = WorkoutSDK()
    menu = CLIMenu(sdk=sdk, stdin=io.StringIO(""), stdout=stdout)
    with patch("src.cli.menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert menu.handle_choice("7") is True
