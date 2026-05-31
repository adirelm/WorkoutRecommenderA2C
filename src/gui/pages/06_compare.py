"""§7.6 / DA4 / TR1 — REINFORCE vs A2C side-by-side comparison (Phase 9 GUI)."""

from __future__ import annotations

import streamlit as st

from src.gui import charts
from src.gui.components import hero, info_card, metric_row, page_footer, reference_callout
from src.gui.state import GUIState, get_sdk
from src.services.comparator import ComparisonResult

_PAGE = "compare"


def _sidebar_controls(state: GUIState) -> tuple[int, int]:
    """Render sidebar sliders for seed-count and episode-count."""
    st.sidebar.header("Comparison settings")
    seeds = st.sidebar.slider(
        "Seeds",
        min_value=1,
        max_value=10,
        value=int(state.get("seeds", 3)),
        step=1,
        help="How many independent random seeds to average over.",
    )
    episodes = st.sidebar.slider(
        "Episodes per seed",
        min_value=5,
        max_value=100,
        value=int(state.get("episodes", 20)),
        step=5,
        help="Per-seed training length. More episodes = smoother bands.",
    )
    state.set("seeds", seeds)
    state.set("episodes", episodes)
    return int(seeds), int(episodes)


def _run_comparison(seeds: int, episodes: int) -> ComparisonResult:
    """Drive sdk.compare with a deterministic progress bar (one tick per seed)."""
    sdk = get_sdk()
    bar = st.progress(0.0, text=f"Training {seeds} seed(s) x {episodes} episodes...")
    # The SDK runs the inner loop; we surface coarse progress before/after.
    bar.progress(0.05, text=f"Starting {seeds} seed(s)...")
    result = sdk.compare(seeds=seeds, episodes=episodes)
    bar.progress(1.0, text="Comparison complete.")
    return result


def _final_stats(result: ComparisonResult) -> dict[str, str]:
    """Final-mean ± std for REINFORCE and A2C (last episode of mean curve)."""
    r_mean = float(result.reinforce_mean_reward[-1])
    r_std = float(result.reinforce_std_reward[-1])
    a_mean = float(result.a2c_mean_reward[-1])
    a_std = float(result.a2c_std_reward[-1])
    return {
        "REINFORCE final": f"{r_mean:+.2f} ± {r_std:.2f}",
        "A2C final": f"{a_mean:+.2f} ± {a_std:.2f}",
        "Seeds": str(result.seed_count),
        "Episodes": str(result.episode_count),
    }


def _render_result(result: ComparisonResult) -> None:
    """Chart + honest-stats panel + caveat."""
    st.subheader("Mean ± std reward bands")
    st.plotly_chart(charts.comparison_band(result), use_container_width=True)

    st.subheader("Honest statistics")
    metric_row(_final_stats(result))
    st.markdown(
        f"Final-episode mean reward across **{result.seed_count}** seed(s), "
        f"each trained for **{result.episode_count}** episodes. "
        "Bands above show ± 1 standard deviation across seeds at every episode."
    )

    reference_callout(
        "Caveat — single-seed-count study",
        "This is a *single-seed-count study*: increasing the seed slider gives a "
        "tighter band but does **not** turn the gap into a statistically "
        "conclusive A2C-beats-REINFORCE (or vice-versa) result. Treat the "
        "ordering as *suggestive*, not significant — formal claims would need a "
        "paired bootstrap or Welch's t-test across many more seeds.",
    )


def render() -> None:
    """Render the §7.6 comparison page."""
    state = GUIState(_PAGE)
    hero(
        title="REINFORCE vs A2C",
        subtitle="§7.6 / DA4 / TR1 — mean ± std reward bands across N seeds.",
        icon="⚖️",
    )
    info_card(
        "What this page does",
        "Trains both policies (REINFORCE and A2C) over <b>N</b> random seeds for "
        "<b>E</b> episodes each, then plots the mean episode-reward with a "
        "± 1 std band per algorithm. The point is to see whether A2C's "
        "actor-critic baseline reduces variance vs REINFORCE's plain returns — "
        "exactly the §7.6 comparison the brief asks for.",
    )

    seeds, episodes = _sidebar_controls(state)

    if st.button(
        f"Run comparison ({seeds} seeds x {episodes} episodes)",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Running paired REINFORCE + A2C training..."):
            result = _run_comparison(seeds, episodes)
        state.set("last_result", result)

    last: ComparisonResult | None = state.get("last_result")
    if last is None:
        st.info("Configure seeds / episodes in the sidebar, then click 'Run comparison'.")
    else:
        _render_result(last)

    page_footer()


render()
