"""Unit tests for src.utils.config_loader (closes code-config-loaded gate).

These tests prove three claims:

1. ``load_config()`` returns a populated dict parsed from
   ``config/config.yaml`` (i.e. the YAML is actually opened, not just
   present on disk).
2. The loader is memoised — two calls with the same path return the
   *same* dict object (identity, not just equality). This guarantees
   cheap repeated access from hot paths.
3. ``get_version()`` returns the YAML ``version`` field as a string,
   matching the in-code ``src.__version__``.
"""

from __future__ import annotations

from src.__version__ import __version__
from src.utils.config_loader import get_version, load_config


def test_load_config_returns_dict() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)
    # A few keys we know exist in the canonical config.yaml.
    assert "version" in cfg
    assert "environment" in cfg
    assert "a2c" in cfg


def test_load_config_is_cached() -> None:
    a = load_config()
    b = load_config()
    # lru_cache contract — same args -> same object identity.
    assert a is b


def test_get_version_returns_string() -> None:
    v = get_version()
    assert isinstance(v, str)
    # YAML version must agree with the Python single source of truth.
    assert v == __version__
