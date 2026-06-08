"""10-seed REINFORCE vs A2C comparison — Phase 10 results-optimization pass.

The prior results/figures/comparison.png used 3 seeds — directional only.
This script re-runs §7.6 at 10 seeds × 50 episodes per algo so the mean
± 1σ bands have enough samples for a Welch t-test on end-of-training
reward. Outputs (overwrite the prior 3-seed artifact):

    results/figures/comparison.png   — overlayed mean ± 1σ bands
    results/comparison_seeded.json   — raw per-(algo, seed, episode) rewards

Trim policy: if 10 × 50 exceeds the ~10 min wall-clock budget, drop to
N_SEEDS=7 / N_EPISODES=30 below. Dry run measured ~2 min, no trim needed.

Run: uv run --active python scripts/run_seeded_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _seeded_comparison_helpers import (
    REPO_ROOT,
    dump_json,
    plot_bands,
    report_significance,
    run_sweep,
)

N_SEEDS: int = 30
N_EPISODES: int = 200
BASE_SEED: int = 42

OUT_PNG: Path = REPO_ROOT / "results" / "figures" / "comparison.png"
OUT_JSON: Path = REPO_ROOT / "results" / "comparison_seeded.json"


def main() -> int:
    print(f"Seeded comparison: {N_SEEDS} seeds × {N_EPISODES} episodes × (REINFORCE, A2C, random)")
    t0 = time.perf_counter()
    r_stack, a_stack, rnd_stack = run_sweep(N_SEEDS, N_EPISODES, BASE_SEED)
    elapsed = time.perf_counter() - t0
    print(f"\nsweep complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    plot_bands(r_stack, a_stack, rnd_stack, OUT_PNG, N_SEEDS, N_EPISODES)
    dump_json(r_stack, a_stack, rnd_stack, OUT_JSON, N_SEEDS, N_EPISODES, BASE_SEED)
    print(f"wrote chart → {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"wrote json  → {OUT_JSON.relative_to(REPO_ROOT)}")
    report_significance(r_stack, a_stack, rnd_stack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
