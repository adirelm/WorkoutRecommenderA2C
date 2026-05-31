# streamlit page filenames must start with NN_ for ordering
"""Action-masking demo page (brief §7.6.1, ADR-004).

Thin orchestrator: 12-channel sliders, rule toggles, softmax helpers,
and the §7.6.1 info block all live in
:mod:`src.gui.pages._action_masking_ui` so this file stays under the
150-LOC CLAUDE.md §1 limit.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.env.state import ACTION_NAMES
from src.gui.charts import action_probability_bar
from src.gui.components import hero, reference_callout
from src.gui.pages._action_masking_ui import (
    REST_ID,
    build_service,
    info_block,
    rule_reasons,
    rule_toggles,
    softmax,
    state_sliders,
)

_RNG_SEED = 7

# Back-compat re-exports — test_gui_page_action_masking.py loads this file by path
# (not import) and reaches for ``page._build_service`` / ``page._softmax``. The
# helpers themselves live in ``_action_masking_ui`` now, but the public names
# stay attached to the page module so the tests keep working unchanged.
_build_service = build_service
_softmax = softmax


def render() -> None:
    """Entry point invoked by ``st.navigation`` for the action-masking page."""
    hero("Action masking", "§7.6.1 demo - four hard rules over the 7-action head", icon="🛡️")
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Demo the four invalid-action mask rules over the 7-action head "
            "and compare unmasked vs masked softmax (brief §7.6.1)."
        )
    info_block()
    state = state_sliders()
    toggles = rule_toggles()
    prior_rest = st.slider(
        "Prior Rest days (mock action history)",
        0,
        3,
        0,
        help="Length of the trailing all-Rest run feeding rest_streak.",
    )
    history = [REST_ID] * int(prior_rest)
    rng = np.random.default_rng(_RNG_SEED)
    logits = rng.normal(0.0, 1.0, size=len(ACTION_NAMES)).astype(np.float32)
    service = build_service(toggles)
    mask = service.mask(state, history)
    masked_logits = service.apply_to_logits(logits, mask)
    unmasked_probs, masked_probs = softmax(logits), softmax(masked_logits)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            action_probability_bar(unmasked_probs, list(ACTION_NAMES), title="Unmasked: softmax(logits)"),
            use_container_width=True,
            key="action_masking_unmasked_probs",
        )
    with right:
        st.plotly_chart(
            action_probability_bar(
                masked_probs,
                list(ACTION_NAMES),
                mask=mask.tolist(),
                title="Masked: illegal actions zeroed (logits → −∞)",
            ),
            use_container_width=True,
            key="action_masking_masked_probs",
        )
    reasons = rule_reasons(state, history, toggles)
    if reasons:
        st.markdown(
            "**Masked this step:** "
            + " . ".join(f"`{ACTION_NAMES[a]}` - {why}" for a, why in reasons.items())
        )
    else:
        st.markdown("**Masked this step:** _none - every action is currently legal._")
    reference_callout(
        "Worked example",
        "Trainee did Legs yesterday with <code>soreness_legs = 0.85</code> -> "
        "Legs is masked the next day (rule 2). The policy still produces a logit for it; "
        "masking just zeroes its softmax weight so the sampler can never choose it.",
    )


render()
