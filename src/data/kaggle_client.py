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
import subprocess
from pathlib import Path

DEFAULT_SLUG = "adnanelouardi/600k-fitness-exercise-and-workout-program-dataset"
EXPECTED_CSVS = ("fitness_exercises.csv", "program_summary.csv")

_SETUP_HINT = (
    "Kaggle credentials not found. Place your API token at "
    "~/.kaggle/kaggle.json (chmod 600) or export KAGGLE_USERNAME and "
    "KAGGLE_KEY. See https://www.kaggle.com/docs/api#authentication."
)
_CLI_HINT = "kaggle CLI not found on PATH. Run `uv sync` (see README.md for setup), then re-run."


class KaggleCredentialsMissingError(RuntimeError):
    """Raised when ~/.kaggle/kaggle.json is missing or KAGGLE_USERNAME/KAGGLE_KEY env unset."""


class KaggleCLINotInstalledError(RuntimeError):
    """Raised when the ``kaggle`` executable is not on PATH."""


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
        missing = [n for n in EXPECTED_CSVS if not (self.raw_dir / n).exists()]
        if missing:
            raise RuntimeError(
                f"Kaggle reported success but expected CSVs missing in "
                f"{self.raw_dir}: {missing}. Check the dataset slug or unzip step."
            )
        return self.raw_dir

    # ----------------------------------------------------------------- helpers
    def _csvs_present(self) -> bool:
        return all((self.raw_dir / name).exists() for name in EXPECTED_CSVS)

    def _credentials_available(self) -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").exists()

    def _wipe_raw_dir(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for name in EXPECTED_CSVS:
            target = self.raw_dir / name
            if target.exists():
                target.unlink()
        for zip_path in self.raw_dir.glob("*.zip"):
            zip_path.unlink()

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
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise KaggleCLINotInstalledError(_CLI_HINT) from e
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "")[-500:]
            raise RuntimeError(
                f"kaggle CLI failed for slug '{self.slug}' (exit {e.returncode}). stderr tail: {stderr_tail}"
            ) from e
