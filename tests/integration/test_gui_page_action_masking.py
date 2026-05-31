"""Deep tests for ``src/gui/pages/08_action_masking.py`` (brief §7.6.1, ADR-004).

The page wraps :class:`src.env.action_mask.ActionMaskService` and lets the
grader toggle each of the four mask rules on/off. Two contracts matter at
this boundary:

1. Toggling each rule (with a state that triggers it) produces a different
   bool mask — the toggle is wired through to ``ActionMaskService``, not a
   visual no-op.
2. Masked positions get **zero probability** after the page's masked-softmax
   path — the ``-inf`` logit substitution must collapse cleanly to 0.0,
   never NaN, regardless of which rule fired.

We exercise the same ``_build_service`` + ``_softmax`` helpers the page
uses so the test pins the page's actual code path, not a re-implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Streamlit's testing harness gives the page a script-run context so
# its ``render()`` invocation at import time does not raise.
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.env.action_mask import ActionMaskService
from src.env.state import ACTION_COUNT, State

_PAGE_PATH = Path(__file__).resolve().parents[2] / "src" / "gui" / "pages" / "08_action_masking.py"
_REST_ID, _LEGS_ID, _COND_ID = 0, 3, 5


def _load_page_module():
    """Import ``08_action_masking.py`` as a module (filename starts with a digit)."""
    spec = importlib.util.spec_from_file_location("page_action_masking", _PAGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The page calls ``render()`` at import time, which touches Streamlit; the
    # AppTest run gives it a script context so exec_module does not raise.
    AppTest.from_file(str(_PAGE_PATH), default_timeout=15.0).run()
    spec.loader.exec_module(mod)
    return mod


def _triggering_state() -> State:
    """State that simultaneously triggers legs + conditioning rules."""
    base = State.initial()
    return State(
        **{
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            "soreness_legs": 0.95,  # > 0.8 → masks Legs
            "rolling_7d_volume": 1.5,
        },  # > 1.2 → masks Conditioning
    )


def _all_on() -> dict[str, bool]:
    return {
        "rest_streak": True,
        "legs_soreness": True,
        "conditioning_overload": True,
        "mobility_always": True,
    }


def test_toggling_each_rule_produces_different_mask() -> None:
    """Disabling each rule (one at a time) must yield a mask different from all-on."""
    page = _load_page_module()
    state = _triggering_state()
    history = [_REST_ID, _REST_ID, _REST_ID]  # triggers rest_streak

    baseline_mask = page._build_service(_all_on()).mask(state, history)
    for rule in ("rest_streak", "legs_soreness", "conditioning_overload"):
        toggles = _all_on()
        toggles[rule] = False
        flipped_mask = page._build_service(toggles).mask(state, history)
        assert not np.array_equal(baseline_mask, flipped_mask), (
            f"toggling {rule!r} off did not change the mask "
            f"(baseline={baseline_mask.tolist()}, flipped={flipped_mask.tolist()})"
        )


def test_masked_positions_get_zero_probability() -> None:
    """After ``apply_to_logits`` + softmax, every masked slot is exactly 0.0."""
    page = _load_page_module()
    state = _triggering_state()
    history = [_REST_ID, _REST_ID, _REST_ID]
    service = ActionMaskService()
    mask = service.mask(state, history)
    # Mask must actually drop at least one action, otherwise the assertion is vacuous.
    assert mask.sum() < ACTION_COUNT, f"expected some masked action, got mask={mask.tolist()}"

    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.0, size=ACTION_COUNT).astype(np.float32)
    masked_logits = service.apply_to_logits(logits, mask)
    probs = np.asarray(page._softmax(masked_logits), dtype=np.float64)

    assert not np.isnan(probs).any(), f"softmax produced NaN: {probs.tolist()}"
    assert probs.sum() == pytest.approx(1.0, abs=1e-6), (
        f"masked-softmax probs must sum to 1.0, got {probs.sum()!r}"
    )
    for i, legal in enumerate(mask.tolist()):
        if not legal:
            assert probs[i] == 0.0, f"masked action id={i} got non-zero probability {probs[i]!r}"
