"""Tests for src/data/kaggle_client.py (brief §7.2.2)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.data.kaggle_client import (
    DEFAULT_SLUG,
    KaggleClient,
    KaggleCLINotInstalledError,
    KaggleCredentialsMissingError,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED_CSVS = ("fitness_exercises.csv", "program_summary.csv")


def _seed_raw_with_fixture_csvs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "fitness_exercises_sample.csv", raw_dir / EXPECTED_CSVS[0])
    shutil.copy(FIXTURES / "program_summary_sample.csv", raw_dir / EXPECTED_CSVS[1])


def test_returns_raw_dir_when_csvs_already_cached(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _seed_raw_with_fixture_csvs(raw_dir)

    called = {"n": 0}

    def fake_run(*args, **kwargs):  # pragma: no cover - asserted not called
        called["n"] += 1
        raise AssertionError("subprocess.run must not be invoked when CSVs cached")

    monkeypatch.setattr(subprocess, "run", fake_run)

    client = KaggleClient(raw_dir=raw_dir)
    result = client.ensure_dataset()
    assert result == raw_dir
    assert called["n"] == 0


def test_raises_when_credentials_missing(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client = KaggleClient(raw_dir=raw_dir)
    with pytest.raises(KaggleCredentialsMissingError):
        client.ensure_dataset()


def test_force_refresh_invokes_cli(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".kaggle").mkdir(parents=True)
    (fake_home / ".kaggle" / "kaggle.json").write_text('{"username":"x","key":"y"}')
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    raw_dir = tmp_path / "raw"
    _seed_raw_with_fixture_csvs(raw_dir)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        # Simulate kaggle CLI re-creating the CSVs after wipe
        _seed_raw_with_fixture_csvs(raw_dir)

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    client = KaggleClient(raw_dir=raw_dir)
    client.ensure_dataset(force_refresh=True)
    assert "cmd" in captured
    assert "kaggle" in captured["cmd"][0]
    assert DEFAULT_SLUG in captured["cmd"]


def test_error_message_has_setup_hint(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client = KaggleClient(raw_dir=raw_dir)
    with pytest.raises(KaggleCredentialsMissingError) as exc:
        client.ensure_dataset()
    assert "~/.kaggle/kaggle.json" in str(exc.value)


def test_raises_when_kaggle_cli_missing(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".kaggle").mkdir(parents=True)
    (fake_home / ".kaggle" / "kaggle.json").write_text('{"username":"x","key":"y"}')
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("kaggle")

    monkeypatch.setattr(subprocess, "run", fake_run)

    client = KaggleClient(raw_dir=raw_dir)
    with pytest.raises(KaggleCLINotInstalledError):
        client.ensure_dataset()


def test_post_download_verification_catches_silent_failure(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".kaggle").mkdir(parents=True)
    (fake_home / ".kaggle" / "kaggle.json").write_text('{"username":"x","key":"y"}')
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def fake_run(*args, **kwargs):
        # Simulate a "successful" CLI run that produces no CSVs.
        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    client = KaggleClient(raw_dir=raw_dir)
    with pytest.raises(RuntimeError) as exc:
        client.ensure_dataset()
    assert "missing" in str(exc.value)
