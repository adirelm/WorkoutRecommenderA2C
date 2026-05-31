"""Streamlit GUI entry — WorkoutRecommenderA2C (Phase 9, brief §1.4 architect decision, ADR-006)."""

import streamlit as st

from src.gui.theme import header, inject_custom_css, set_page_config

set_page_config(title="WorkoutRecommenderA2C", icon="🏋️")
inject_custom_css()
header("WorkoutRecommenderA2C", "Bar-Ilan Vibe-Coding & RL Workshop · Assignment 3")

PAGES = [
    st.Page("pages/01_home.py", title="Home", icon="🏠"),
    st.Page("pages/02_data.py", title="Data", icon="📊"),
    st.Page("pages/03_lstm.py", title="LSTM World Model", icon="🧠"),
    st.Page("pages/04_reinforce.py", title="REINFORCE", icon="🎯"),
    st.Page("pages/05_a2c.py", title="A2C", icon="⚡"),
    st.Page("pages/06_compare.py", title="Compare", icon="⚖️"),
    st.Page("pages/07_recommend.py", title="Recommend", icon="💡"),
    st.Page("pages/08_action_masking.py", title="Action Masking", icon="🛡️"),
    st.Page("pages/09_theory.py", title="Theory", icon="📐"),
    st.Page("pages/10_discussion.py", title="Discussion", icon="💬"),
]
pg = st.navigation(PAGES, position="sidebar")
pg.run()
