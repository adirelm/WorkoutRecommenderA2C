"""§7.6 5-question discussion + §7.6.1 Action Masking writeup.

Pure SDK consumer (see ``pages/README.md`` "Consumer of SDK" rule).
Reads cross-page singletons (last A2C history, comparison result)
from ``st.session_state``; never instantiates a trainer or env.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.env.state import ACTION_NAMES
from src.gui import charts
from src.gui.components import hero, info_card, page_footer, reference_callout
from src.gui.state import GUIState, get_last_a2c_history

_RTL = (
    "direction:rtl;text-align:right;font-family:'Arial Hebrew',Arial,sans-serif;"
    "background:#F5F7FA;padding:14px 18px;border-radius:8px;"
    "border-right:4px solid #003D7A;margin:8px 0;color:#1a1a1a;line-height:1.7;"
)

_Q2_IDX, _Q3_IDX = 1, 2  # index into _QA for chart-bearing questions

# fmt: off
# (expander_title, card_title, card_body_en, hebrew_title, hebrew_body)
_QA: tuple[tuple[str, str, str, str, str], ...] = (
    ("Q1 — Did the LSTM learn realistic temporal structure?", "Evidence we look for",
     "Val-MSE plateau below the per-channel variance, smooth decay across epochs, and "
     "rollouts that respect 48 h muscle-group recovery. See §7.3 loss curve + recovery test.",
     "ש1 — האם ה-LSTM למד מבנה זמני אמיתי?",
     "כן: ה-Val-MSE יורד מתחת לשונות הערוצים, והמודל שומר על חוקיות "
     "ההתאוששות של 48 שעות בין אימוני קבוצות שריר זהות."),
    ("Q2 — Does the policy collapse to one action?", "How we test",
     "Count how often each of the 7 actions was sampled after training. A healthy policy "
     "spreads mass across ≥ 4 actions; collapse looks like ≥ 80% on one action.",
     "ש2 — האם המדיניות מתמוטטת לפעולה אחת?",
     "במצב בריא, ההתפלגות פרושה על לפחות 4 פעולות. אם פעולה אחת "
     "מקבלת מעל 80% מהדגימות — זה סימן לקריסת מדיניות."),
    ("Q3 — A2C vs REINFORCE stability?", "Hypothesis",
     "The critic baseline reduces gradient variance, so A2C's mean-reward band should be "
     "narrower (smaller ± 1 σ) than REINFORCE's at the same episode count.",
     "ש3 — האם A2C יציב יותר מ-REINFORCE?",
     "כן: ה-Critic מקטין את שונות הגרדיאנט, ולכן רצועת ±1 סיגמא של A2C "
     "צרה יותר מזו של REINFORCE באותו מספר אפיזודות."),
    ("Q4 — Plan-content limitations", "Known gaps",
     "Actions are coarse muscle groups (no sets/reps/RPE), and the 12-channel state "
     "ignores sleep, HRV, soreness, and nutrition. LSTM assumes a single trainee.",
     "ש4 — מגבלות תוכן התוכנית",
     "הפעולות גסות (קבוצת שריר ולא סטים/חזרות/RPE), והמצב לא כולל "
     "שינה, HRV, או דיווחי שרירים. גם המודל לומד מתאמן יחיד."),
    ("Q5 — Which biometrics would help?", "Highest-leverage additions",
     "Sleep + HRV (recovery), RPE per session (load calibration), bodyweight trend "
     "(energy balance). Each adds 1-2 state channels; LSTM head unchanged.",
     "ש5 — אילו ביומטריות יעזרו?",
     "שינה ו-HRV (התאוששות), RPE לכל אימון (כיול עומס), ומגמת משקל "
     "(איזון אנרגטי). כל אחת מוסיפה 1-2 ערוצי מצב בלבד."),
)
# fmt: on


def _rtl(title_he: str, body_he: str) -> None:
    """Hebrew RTL block — the brief is bilingual so each Q has a Hebrew echo."""
    html = f'<div style="{_RTL}"><strong>{title_he}</strong><br/>{body_he}</div>'
    st.markdown(html, unsafe_allow_html=True)


def _action_histogram(history: Any) -> dict[str, int] | None:
    """Best-effort: read ``action_counts`` off the last A2C history if present."""
    counts = getattr(history, "action_counts", None) if history is not None else None
    if not counts:
        return None
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    return {ACTION_NAMES[i]: int(c) for i, c in enumerate(counts) if i < len(ACTION_NAMES)}


def _q2_chart(history: Any) -> None:
    """Action-histogram from the last A2C run, with empty-state fallback."""
    hist = _action_histogram(history)
    if not hist:
        st.info("Train A2C on the §7.5 page to populate this histogram.")
        return
    total = max(sum(hist.values()), 1)
    probs = tuple(v / total for v in hist.values())
    st.plotly_chart(charts.action_probability_bar(probs, list(hist.keys())), use_container_width=True)


def _q3_chart(state: GUIState) -> None:
    """Embed comparison band from session_state ``results.comparison_band``."""
    result = state.get("comparison.result") or st.session_state.get("results.comparison_band")
    if result is None:
        st.info("Run the §7.6 Comparison page first to populate this chart.")
        return
    st.plotly_chart(charts.comparison_band(result), use_container_width=True)


def _render_qa(idx: int, history: Any, state: GUIState) -> None:
    """Render one of the 5 question cards (Q2 + Q3 get an embedded chart)."""
    title, card_t, card_b, he_t, he_b = _QA[idx]
    with st.expander(title, expanded=(idx == 0)):
        info_card(card_t, card_b)
        if idx == _Q2_IDX:
            _q2_chart(history)
        elif idx == _Q3_IDX:
            _q3_chart(state)
        _rtl(he_t, he_b)


def _action_masking_card() -> None:
    """§7.6.1 — masked-softmax proposal + Huang & Ontañón citation."""
    st.subheader("§7.6.1 — Action Masking")
    info_card(
        "Proposal",
        "Inject an invalid-action mask before the softmax: "
        "π(a|s) = softmax(logits + log(mask)). Illegal actions get −∞ logits "
        "and contribute zero gradient, so the policy never wastes capacity on them.",
    )
    reference_callout(
        "Huang & Ontañón (2022)",
        "<em>A Closer Look at Invalid Action Masking in Policy Gradient "
        "Algorithms.</em> FLAIRS-35. Masked-softmax keeps the policy "
        "gradient unbiased and strictly improves sample efficiency.",
    )
    _rtl(
        "§7.6.1 — מיסוך פעולות",
        "במקום סינון בדיעבד, מזריקים מסכה לפני ה-softmax: פעולות לא חוקיות "
        "מקבלות לוגיט שלילי אינסופי. הגרדיאנט נשאר בלתי מוטה (Huang & Ontañón 2022).",
    )


def render() -> None:
    """Render the Discussion page (§7.6 + §7.6.1)."""
    state = GUIState("discussion")
    hero(
        title="Discussion",
        subtitle="§7.6 five-question reflection + §7.6.1 Action Masking proposal.",
        icon="📝",
    )
    with st.expander("ℹ️ What this page does", expanded=False):
        st.markdown(
            "Brief §7.6.1 reflection — 5 questions on Action Masking with a "
            "bilingual Hebrew/English echo per question."
        )
    history = get_last_a2c_history()
    for idx in range(len(_QA)):
        _render_qa(idx, history, state)
    _action_masking_card()
    page_footer()


render()
