"""Cached YAML config loader (V3 §7.3 + brief §6 single source of truth).

The project keeps every algorithm-relevant knob in ``config/config.yaml``
(CLAUDE.md §4 "No Hardcoded Values"). Until now nothing in ``src/``
actually *opened* that file — dataclasses mirrored the values by hand,
which silently allowed drift between the YAML and the running code.

This module closes that gap with the smallest possible surface:

* :func:`load_config` reads ``config/config.yaml`` once and memoises the
  result via :func:`functools.lru_cache`. Repeated calls return the
  *same* dict object (identity-equal), so callers can rely on cached
  parses without paying I/O cost.
* :func:`get_version` is a convenience accessor for the ``version`` key
  (brief §8 versioning) — the SDK logs this at init to prove the YAML
  is actually loaded, not just present on disk.

The loader stays intentionally minimal: no schema validation, no env
overrides, no dataclass coercion. Those are the *dataclasses'* job; this
module just guarantees the YAML on disk is reachable from Python.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache ``config/config.yaml``. Returns a plain ``dict``.

    Args:
        path: Optional override; defaults to the repo-level
            ``config/config.yaml`` resolved relative to this file.

    Returns:
        Parsed YAML as a nested ``dict``. The same object is returned on
        every call for a given ``path`` thanks to ``lru_cache``.
    """
    p = Path(path) if path is not None else _DEFAULT_PATH
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_version() -> str:
    """Return the ``version`` key from the cached config (brief §8)."""
    return str(load_config()["version"])
