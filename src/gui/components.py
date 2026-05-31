"""Reusable Streamlit widgets for the WorkoutRecommenderA2C GUI.

Bar-Ilan blue palette: primary ``#003D7A``, accent ``#FFCD00``.
Every page composes its UI from these primitives so styling stays
consistent and CLAUDE.md §1 (≤150 LOC) holds for page modules.
"""

from __future__ import annotations

import streamlit as st

PRIMARY = "#003D7A"
ACCENT = "#FFCD00"
LIGHT_BG = "#F5F7FA"
REPO_URL = "https://github.com/AdirElmakyes/Assignment3-WorkoutRecommenderA2C"


def hero(title: str, subtitle: str, icon: str = "🏋️") -> None:
    """Big hero card at the top of a page."""
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{PRIMARY} 0%,#0056B3 100%);
                    padding:28px 32px;border-radius:12px;color:white;
                    margin-bottom:18px;border-left:6px solid {ACCENT};">
          <div style="font-size:42px;line-height:1;">{icon}</div>
          <h1 style="margin:6px 0 4px 0;color:white;font-weight:700;">{title}</h1>
          <p style="margin:0;font-size:15px;opacity:0.92;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: dict[str, str]) -> None:
    """KPI strip rendered via ``st.metric`` in equal-width columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=label, value=value)


def info_card(title: str, body_md: str) -> None:
    """Styled markdown card with the primary-blue left border."""
    st.markdown(
        f"""
        <div style="background:{LIGHT_BG};padding:16px 20px;border-radius:8px;
                    border-left:4px solid {PRIMARY};margin:10px 0;">
          <h4 style="margin:0 0 8px 0;color:{PRIMARY};">{title}</h4>
          <div style="color:#1a1a1a;font-size:14px;line-height:1.55;">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def code_block(code: str, language: str = "python") -> None:
    """Thin wrapper around ``st.code`` so pages don't import the API directly."""
    st.code(code, language=language)


def deliverable_checklist(items: list[tuple[str, bool, str]]) -> None:
    """Read-only checklist: each row is ``(label, done, link)``.

    ``link`` may be an empty string. Items render with a ✅ / ⬜ glyph
    so the visual state is obvious even if Streamlit theming changes.
    """
    for label, done, link in items:
        glyph = "✅" if done else "⬜"
        suffix = f" — [open]({link})" if link else ""
        st.markdown(f"{glyph} **{label}**{suffix}")


def latex_block(equation: str, label: str = "") -> None:
    """KaTeX render via ``st.latex`` with an optional caption above it."""
    if label:
        st.caption(label)
    st.latex(equation)


def reference_callout(title: str, body_md: str) -> None:
    """Left-bordered note used for citations and lecturer feedback echoes."""
    st.markdown(
        f"""
        <div style="background:#FFF8E1;padding:12px 16px;border-radius:6px;
                    border-left:4px solid {ACCENT};margin:8px 0;">
          <strong style="color:{PRIMARY};">{title}</strong>
          <div style="font-size:13px;margin-top:4px;color:#333;">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_footer() -> None:
    """Consistent footer with §1.4 reminder and repo link."""
    st.divider()
    st.markdown(
        f"""
        <div style="text-align:center;color:#666;font-size:12px;padding:10px 0;">
          Bar-Ilan University · Vibe Coding Workshop · Assignment 3 ·
          <a href="{REPO_URL}" style="color:{PRIMARY};">repository</a>
          <br/>
          <em>§1.4 — Human is architect, AI is implementer. All AI-generated
          changes were reviewed before commit.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
