"""Phase 6 integration: WorkoutSDK + CLIMenu end-to-end (small smoke)."""

from __future__ import annotations

import io

from src.cli.menu import CLIMenu
from src.env.state import State
from src.sdk.sdk import WorkoutSDK


def test_sdk_full_pipeline_smoke():
    sdk = WorkoutSDK(seed=42)
    logbook = sdk.prepare_data()
    assert logbook.state_dim == 12
    handle_r, _history_r = sdk.train_reinforce(episodes=3)
    assert handle_r.algorithm == "REINFORCE"
    handle_a, _history_a = sdk.train_a2c(episodes=3)
    assert handle_a.algorithm == "A2C"
    rec = sdk.recommend(State.initial())
    assert 0 <= rec.action_id < 7
    assert abs(sum(rec.probs) - 1.0) < 1e-5


def test_cli_drives_sdk():
    """CLI immediate-exit smoke."""
    sdk = WorkoutSDK(seed=42)
    stdin = io.StringIO("0\n")
    stdout = io.StringIO()
    menu = CLIMenu(sdk, stdin=stdin, stdout=stdout)
    exit_code = menu.run()
    assert exit_code == 0
