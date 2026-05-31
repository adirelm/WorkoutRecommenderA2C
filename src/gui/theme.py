"""§9 Bar-Ilan theme: palette, page config, CSS, and shared header helper."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class BarIlanTheme:
    """Bar-Ilan University color palette used across every GUI page."""

    primary: str = "#003D7A"
    accent: str = "#FFCD00"
    light_bg: str = "#F5F7FA"
    success: str = "#1F8A4C"
    warning: str = "#E89B1D"
    danger: str = "#C8332E"


THEME = BarIlanTheme()


def set_page_config(title: str, icon: str = "🏋️") -> None:
    """Apply consistent Streamlit page chrome (wide layout, expanded sidebar)."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Workout Recommender A2C — Bar-Ilan Vibe Coding"},
    )


def inject_custom_css() -> None:
    """Inject hero typography, card shadows, and Hebrew-RTL support."""
    css = f"""
    <style>
      .hero-title {{
        font-size: 2.4rem; font-weight: 700; color: {THEME.primary};
        margin-bottom: 0.2rem; letter-spacing: -0.5px;
      }}
      .hero-subtitle {{
        font-size: 1.05rem; color: #4A5568; margin-top: 0;
      }}
      .bi-card {{
        background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 8px rgba(0,61,122,0.08);
        border-left: 4px solid {THEME.primary};
      }}
      .bi-accent {{ color: {THEME.accent}; }}
      .rtl, [data-testid="stMarkdownContainer"] .rtl {{
        direction: rtl; text-align: right; font-family: 'Arial Hebrew', Arial, sans-serif;
      }}
      div[data-testid="stSidebar"] {{ background: {THEME.light_bg}; }}
      .stButton>button[kind="primary"] {{
        background: {THEME.primary}; border-color: {THEME.primary};
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def header(title: str, subtitle: str = "", icon: str = "🏋️") -> None:
    """Render a consistent page header (logo placeholder + title + subtitle)."""
    cols = st.columns([1, 9])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:3rem;line-height:1;text-align:center;"
            f"color:{THEME.primary};'>{icon}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(f"<div class='hero-title'>{title}</div>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p class='hero-subtitle'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()
