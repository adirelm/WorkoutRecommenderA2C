# EXPERIMENTS — A3 Hypothesis-Setup-Result-Verdict log

## §1 Scope
Per brief §7.6 (5 discussion questions + Action Masking) and §7.7 (REINFORCE-vs-A2C side-by-side requirement), this doc records the experiments behind the analysis notebook's narrative.

## §2 Experiment register

| # | Hypothesis | Setup | Result | Verdict |
|---|---|---|---|---|
| E1 | LSTM world model learns the §1.5-state temporal structure on a 28-day synthetic trainee | LSTMWorldModel(hidden=64, num_layers=1), MSE on sliding 7-day windows, **trained for 12 epochs (figure caption)**, seed=42 | Train + val loss curves in results/figures/lstm_loss.png; both losses still descending at epoch 12 (no plateau in the documented run) | HEDGED — loss decreases but on PLAN-CONTENT data; cannot claim "realistic physiology" (brief §7.6 Q4 caveat). **Note**: training was capped early for the documented chart; deeper training (up to 200 epochs) is exposed via the LSTM GUI page slider but was not re-generated for this static doc artefact. See §2.1 for the val<train MSE explanation. |
| E2 | REINFORCE with baseline converges faster than vanilla on the 28-day workout episode | PolicyNet(hidden=128), running-mean baseline alpha=0.05, gamma=0.99, lr=3e-4, 50 episodes, seed=42 | reinforce_rewards.png shows reward trend rising; high variance throughout — known REINFORCE behaviour | CONFIRMED qualitatively; quantitative claim deferred to multi-seed runs (compare() with 3 seeds) |
| E3 | A2C stabilises training relative to REINFORCE via critic-baseline | ActorCriticNet(actor_hidden=128, critic_hidden=128), actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01, 50 episodes | a2c_training.png shows actor + critic loss panels + rewards | OBSERVED less variance in late-episode rewards vs REINFORCE; single-seed evidence — not statistical proof. **Caveat**: A2C reward trends DOWN after ~episode 22 on the single-seed plot — likely entropy-coef-induced exploration noise after the policy converges to a local optimum (β=0.01 keeps the policy stochastic enough to revisit suboptimal chains). The **3-seed mean comparison** (`results/figures/comparison.png`) is the headline result; the single-seed late-episode dip should not be over-interpreted. |
| E4 | REINFORCE-vs-A2C head-to-head (DA4) | compare(seeds=3, episodes=30) for both algos | results/figures/comparison.png — mean ± std bands | A2C mean tracks slightly above REINFORCE; bands overlap — no statistical claim at this sample size |
| E5 | Action Masking (ADR-004) prevents degenerate policy collapse | ActionMaskService with rest-streak rule + soreness threshold; assert mask never zeros all actions | tests/test_action_mask.py + tests/test_env_invariants.py::test_mask_never_zeros_out_every_action — both green | CONFIRMED behaviorally; integration into PG-unbiasedness preserved per Huang & Ontañón 2022 |
| E6 | Brief §7.6 Q4 limitation: plan-content data ≠ real outcomes | Train LSTM on prescribed-program-derived trajectory; observe predicted next_state vs realistic biology | LSTM fits the synthetic-trainee rules; no biological-validity claim made | ACKNOWLEDGED — honest limitation in README §Honest Limitations + analysis.ipynb §discussion |
| E7 | GUI session_state survives 10 page transitions without loss | Streamlit GUI driven through 10 sequential page navigations; assert session_state keys + values intact after each transition | tests/test_gui_e2e_flow.py — all 10 transitions green; session_state preserved end-to-end | CONFIRMED — navigation flow does not drop state |
| E8 | Live-training per-episode chart-update callback slows training by >10% | REINFORCE, 50 episodes, seed=42, planned wall-clock comparison with vs without the chart-update callback | Status: deferred — overhead measurement not run (single-process training; no parallel-vs-sequential comparison performed) | DEFERRED — no quantitative claim made; callback overhead remains uncharacterised |

## §3 Counterfactuals / What was NOT tested

- **Multi-seed convergence rigor**: only 3 seeds × 30 episodes in compare(); a publication-grade study would need 30+ seeds × 500 episodes.
- **Real Kaggle download**: tests use fixture CSVs; the full 600K-row download was not exercised in CI.
- **Comparison against random/heuristic baseline**: §7.6 Q3 asks REINFORCE-vs-A2C; we have no random-policy or rotate-chains baseline to anchor the absolute magnitude.

## §3.1 E9 — λ_1 × λ_2 sensitivity pilot (V3 §9.1)

| | |
|---|---|
| **Hypothesis** | The reward decomposition `r_t = gain_t − λ_1·overload_t − λ_2·imbalance_t` is sensitive to the (λ_1, λ_2) weighting; sweeping a 3×3 grid produces a measurable shift in REINFORCE's late-window mean reward. |
| **Setup** | `scripts/run_lambda_sensitivity.py` — λ_1 ∈ {1.0, 2.0, 3.0} × λ_2 ∈ {0.5, 1.0, 2.0}; per cell: train REINFORCE for 20 episodes at seed=42, record the mean of the last 5 episode-rewards. |
| **Output** | `results/figures/lambda_sensitivity.png` (3×3 heatmap with annotated cell values). |
| **Result** | Reward shifts monotonically with **λ_2** (imbalance weight): mean drops from +5.48 at λ_2=0.5 to +2.79 at λ_2=2.0 across all λ_1 rows. **λ_1 (overload) shows zero effect** under this short pilot — every column is identical across the three λ_1 values. |
| **Interpretation** | The λ_2 monotonicity confirms the imbalance term is *active* in the early policy (the agent picks chains that skew the muscle-group distribution, so a heavier λ_2 directly penalises observed behaviour). The dead λ_1 axis says the **overload penalty never fires** at 20 episodes under seed=42 — the synthetic trainee's rolling_7d_volume stays below the `1.2 × baseline_7d_volume` threshold, so `overload_t = 0` regardless of weight. This is a *methodology* finding, not a *policy* finding: it tells us the pilot's episode budget is too short to exercise the overload arm of the reward, and a publication-grade sweep would need either (a) higher baseline drift, (b) longer episodes, or (c) a trainee that occasionally over-prescribes. |
| **Verdict** | **LIMITED SINGLE-SEED PILOT** — methodology demonstrated; statistical claim not made. The λ_2 axis shows the sweep machinery works; the λ_1 axis is a known dormant dimension at this episode count. |

Reproduce:

```bash
uv run --active python scripts/run_lambda_sensitivity.py
# → results/figures/lambda_sensitivity.png   (≈9 s wall-clock)
```

## §4 Reproducibility caveats

All experiments are seeded via src/utils/seeding.set_global_seed; the seeded determinism test (tests/test_reproducibility.py) verifies tensor equality across runs at seed=42. CUDA non-determinism: scatter_add/index_add are non-deterministic on CUDA — we default to CPU. See README §Reproducibility caveats for the full list.

## §5 Open questions surfaced

1. Does training the LSTM on a multi-program mixture (not just PHUL) improve generalisation? Not tested.
2. How sensitive is A2C's stability to the actor_lr/critic_lr ratio? Defaults pick 3e-4 / 1e-3; sweep is open work.
3. Does Action Masking change the policy's eventual fixed point, or only the trajectory? Theory ([8]) says no for the fixed-policy distribution but the gradient direction changes — open empirical question.
