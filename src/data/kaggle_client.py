"""Kaggle dataset client (brief §7.2.2).

Downloads the Adnane Louardi 600K+ Fitness Exercise & Workout Program
dataset into ``data/raw/`` via the ``kaggle`` CLI, and caches the result
so subsequent runs make no network calls.

Credentials lookup order (matches the kaggle CLI itself):
  1. ``$KAGGLE_USERNAME`` and ``$KAGGLE_KEY`` env vars.
  2. ``~/.kaggle/kaggle.json``.

If neither is present *and* the CSVs are absent from ``raw_dir``, we
raise :class:`KaggleCredentialsMissingError` with a one-line setup hint
pointing at ``~/.kaggle/kaggle.json`` — failing loud is preferable to a
silent retry storm.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_SLUG = "adnanelouardi/600k-fitness-exercise-and-workout-program-dataset"
EXPECTED_CSVS = ("fitness_exercises.csv", "program_summary.csv")

_SETUP_HINT = (
    "Kaggle credentials not found. Place your API token at "
    "~/.kaggle/kaggle.json (chmod 600) or export KAGGLE_USERNAME and "
    "KAGGLE_KEY. See https://www.kaggle.com/docs/api#authentication."
)


class KaggleCredentialsMissingError(RuntimeError):
    """Raised when ~/.kaggle/kaggle.json is missing or KAGGLE_USERNAME/KAGGLE_KEY env unset."""


class KaggleClient:
    """Thin wrapper around the ``kaggle`` CLI with idempotent caching."""

    def __init__(self, raw_dir: Path, slug: str = DEFAULT_SLUG) -> None:
        self.raw_dir = Path(raw_dir)
        self.slug = slug

    # ------------------------------------------------------------------ public
    def ensure_dataset(self, force_refresh: bool = False) -> Path:
        """Ensure the Kaggle CSVs exist in :attr:`raw_dir`.

        Returns the ``raw_dir`` path. If the CSVs are already cached and
        ``force_refresh`` is False, no network call is made. Otherwise
        the ``kaggle`` CLI is invoked. Raises
        :class:`KaggleCredentialsMissingError` if no credentials are
        configured and the CSVs are not already cached.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if force_refresh:
            self._wipe_raw_dir()
        elif self._csvs_present():
            return self.raw_dir

        if not self._credentials_available():
            raise KaggleCredentialsMissingError(_SETUP_HINT)

        self._invoke_kaggle_cli()
        return self.raw_dir

    # ----------------------------------------------------------------- helpers
    def _csvs_present(self) -> bool:
        return all((self.raw_dir / name).exists() for name in EXPECTED_CSVS)

    def _credentials_available(self) -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").exists()

    def _wipe_raw_dir(self) -> None:
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _invoke_kaggle_cli(self) -> None:
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.slug,
            "-p",
            str(self.raw_dir),
            "--unzip",
        ]
        subprocess.run(cmd, check=True)
