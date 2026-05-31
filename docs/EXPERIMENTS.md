# EXPERIMENTS — A3 Hypothesis-Setup-Result-Verdict log

## §1 Scope
Per brief §7.6 (5 discussion questions + Action Masking) and §7.7 (REINFORCE-vs-A2C side-by-side requirement), this doc records the experiments behind the analysis notebook's narrative.

## §2 Experiment register

| # | Hypothesis | Setup | Result | Verdict |
|---|---|---|---|---|
| E1 | LSTM world model learns the §1.5-state temporal structure on a 28-day synthetic trainee | LSTMWorldModel(hidden=64, num_layers=1), MSE on sliding 7-day windows, 50 epochs, seed=42 | Train + val loss curves in results/figures/lstm_loss.png; val loss plateaus after ~25 epochs | HEDGED — loss decreases but on PLAN-CONTENT data; cannot claim "realistic physiology" (brief §7.6 Q4 caveat) |
| E2 | REINFORCE with baseline converges faster than vanilla on the 28-day workout episode | PolicyNet(hidden=128), running-mean baseline alpha=0.05, gamma=0.99, lr=3e-4, 50 episodes, seed=42 | reinforce_rewards.png shows reward trend rising; high variance throughout — known REINFORCE behaviour | CONFIRMED qualitatively; quantitative claim deferred to multi-seed runs (compare() with 3 seeds) |
| E3 | A2C stabilises training relative to REINFORCE via critic-baseline | ActorCriticNet(actor_hidden=128, critic_hidden=128), actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01, 50 episodes | a2c_training.png shows actor + critic loss panels + rewards | OBSERVED less variance in late-episode rewards vs REINFORCE; single-seed evidence — not statistical proof |
| E4 | REINFORCE-vs-A2C head-to-head (DA4) | compare(seeds=3, episodes=30) for both algos | results/figures/comparison.png — mean ± std bands | A2C mean tracks slightly above REINFORCE; bands overlap — no statistical claim at this sample size |
| E5 | Action Masking (ADR-004) prevents degenerate policy collapse | ActionMaskService with rest-streak rule + soreness threshold; assert mask never zeros all actions | tests/test_action_mask.py + tests/test_env_invariants.py::test_mask_never_zeros_out_every_action — both green | CONFIRMED behaviorally; integration into PG-unbiasedness preserved per Huang & Ontañón 2022 |
| E6 | Brief §7.6 Q4 limitation: plan-content data ≠ real outcomes | Train LSTM on prescribed-program-derived trajectory; observe predicted next_state vs realistic biology | LSTM fits the synthetic-trainee rules; no biological-validity claim made | ACKNOWLEDGED — honest limitation in README §Honest Limitations + analysis.ipynb §discussion |
| E7 | GUI session_state survives 10 page transitions without loss | Streamlit GUI driven through 10 sequential page navigations; assert session_state keys + values intact after each transition | tests/test_gui_e2e_flow.py — all 10 transitions green; session_state preserved end-to-end | CONFIRMED — navigation flow does not drop state |
| E8 | Live-training per-episode chart-update callback slows training by >10% | REINFORCE, 50 episodes, seed=42, planned wall-clock comparison with vs without the chart-update callback | Status: deferred — overhead measurement not run (single-process training; no parallel-vs-sequential comparison performed) | DEFERRED — no quantitative claim made; callback overhead remains uncharacterised |

## §3 Counterfactuals / What was NOT tested

- **Multi-seed convergence rigor**: only 3 seeds × 30 episodes in compare(); a publication-grade study would need 30+ seeds × 500 episodes.
- **λ_1, λ_2 sensitivity sweep**: ADR-003 picks defaults (2.0, 1.0) without sweeping the (overload, imbalance) weight plane.
- **Real Kaggle download**: tests use fixture CSVs; the full 600K-row download was not exercised in CI.
- **Comparison against random/heuristic baseline**: §7.6 Q3 asks REINFORCE-vs-A2C; we have no random-policy or rotate-chains baseline to anchor the absolute magnitude.

## §4 Reproducibility caveats

All experiments are seeded via src/utils/seeding.set_global_seed; the seeded determinism test (tests/test_reproducibility.py) verifies tensor equality across runs at seed=42. CUDA non-determinism: scatter_add/index_add are non-deterministic on CUDA — we default to CPU. See README §Reproducibility caveats for the full list.

## §5 Open questions surfaced

1. Does training the LSTM on a multi-program mixture (not just PHUL) improve generalisation? Not tested.
2. How sensitive is A2C's stability to the actor_lr/critic_lr ratio? Defaults pick 3e-4 / 1e-3; sweep is open work.
3. Does Action Masking change the policy's eventual fixed point, or only the trajectory? Theory ([8]) says no for the fixed-policy distribution but the gradient direction changes — open empirical question.
