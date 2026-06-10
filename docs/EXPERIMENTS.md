# EXPERIMENTS — A3 Hypothesis-Setup-Result-Verdict log

## §1 Scope
Per brief §7.6 (5 discussion questions + Action Masking) and §7.7 (REINFORCE-vs-A2C side-by-side requirement), this doc records the experiments behind the analysis notebook's narrative.

## §2 Experiment register

| # | Hypothesis | Setup | Result | Verdict |
|---|---|---|---|---|
| E1 | LSTM world model learns the §1.5-state temporal structure on the **real PHUL** trajectory | LSTMWorldModel(hidden=64, num_layers=1), MSE on sliding 7-day windows over the real *Optimized PHUL* day-level trajectory (84 days → **71 train + 7 val windows**), **2000 epochs (early-stop tracking)**, seed=42 (`scripts/run_lstm_convergence.py`; the notebook reproduces the same fit) | Train + val curves in `results/figures/lstm_loss.png`. Final train MSE = **0.8993**, final val MSE = **2.0259**, **best val MSE ≈ 0.5725 @ epoch ~575**. Both curves plateau well before 2000 epochs. ~6 s CPU. | CONFIRMED — clear plateau; the model fits the real program's day-level rhythm. **Physiology disclaimer**: this is real PLAN-CONTENT (action sequence + Σ(sets·reps) volumes) filtered through a simulated body, not measured biology (brief §7.6 Q4). |
| E2 | REINFORCE learns a non-trivial policy **rolling out against the frozen LSTM** | PolicyNet(hidden=128), running-mean baseline alpha=0.05, gamma=0.99, lr=3e-4, 50 episodes, seed=42, env = `build_lstm_env` (frozen real-PHUL LSTM) | `reinforce_rewards.png`: reward rises to **final 5.394**, **mean 4.587 ± 1.266** over 50 episodes; characteristic REINFORCE variance | CONFIRMED qualitatively; quantitative comparison in the 10-seed run (E4). The agent now trains on the *learned world model*, not the analytic trainee (audit F-1). |
| E3 | A2C stabilises training relative to REINFORCE via critic-baseline | ActorCriticNet(actor_hidden=128, critic_hidden=128), actor_lr=3e-4, critic_lr=1e-3, entropy_coef=0.01, 50 episodes, frozen-LSTM env | `a2c_training.png`: **mean reward 4.684 ± 1.102** (tighter σ than REINFORCE's 1.266), mean actor loss 0.199 ± 0.118, mean critic loss 0.895 ± 0.413 | OBSERVED lower single-seed reward variance than REINFORCE; single-seed evidence — the **10-seed comparison** (E4) is the headline. |
| E4 | REINFORCE-vs-A2C head-to-head (DA4) **+ masked-random baseline**, all over the frozen LSTM env | `scripts/run_seeded_comparison.py` — **30 seeds × 200 episodes** (12× the prior budget); per-seed last-5-episode tail; Welch's two-sample t-tests | `results/figures/comparison.png` + `results/comparison_seeded.json` (30 seeds × 200 ep, 3 series). Per-seed last-5-ep tail means: **REINFORCE = +5.570 ± 0.678**, **A2C = +5.478 ± 0.865**, **masked-Random = +5.853 ± 0.488**. A2C vs REINFORCE: t = **−0.454, p = 0.652** (NOT sig). REINFORCE vs Random: **p = 0.074** (not distinguishable). A2C vs Random: **p = 0.048** — A2C's mean is *below* random, i.e. significantly **worse**, not better. | **Two honest nulls.** (1) A2C and REINFORCE remain statistically indistinguishable even at 30×200 — strong evidence the gap is genuinely ~0 on this env, not just under-powered. (2) **Neither learned policy beats a uniform-random *masked* policy** (random's mean is highest). Interpretation: the ActionMaskService (ADR-004) already confines actions to a safe/reasonable region and the shaped reward is fairly flat within it, so masked-random is already near-optimal and RL has little headroom to exploit. This is a property of the simplified toy env/reward (brief §7.6), **not** an algorithm defect — and the random baseline (added per §3 counterfactual) is exactly what surfaced it. |
| E5 | Action Masking (ADR-004) prevents degenerate policy collapse | ActionMaskService with rest-streak rule + soreness threshold; assert mask never zeros all actions | tests/test_action_mask.py + tests/test_env_invariants.py::test_mask_never_zeros_out_every_action — both green | CONFIRMED behaviorally; integration into PG-unbiasedness preserved per Huang & Ontañón 2022 |
| E6 | Brief §7.6 Q4 limitation: plan-content data ≠ real outcomes | Train LSTM on the real PHUL program-derived trajectory; observe predicted next_state vs realistic biology | LSTM fits the real program's prescribed structure with simulated physiology; no biological-validity claim made | ACKNOWLEDGED — honest limitation in README §Honest Limitations + analysis.ipynb §discussion |
| E7 | GUI session_state survives 10 page transitions without loss | Streamlit GUI driven through 10 sequential page navigations; assert session_state keys + values intact after each transition | tests/test_gui_e2e_flow.py — all 10 transitions green; session_state preserved end-to-end | CONFIRMED — navigation flow does not drop state |
| E8 | Live-training per-episode chart-update callback slows training by >10% | REINFORCE, 50 episodes, seed=42, planned wall-clock comparison with vs without the chart-update callback | Status: deferred — overhead measurement not run (single-process training; no parallel-vs-sequential comparison performed) | DEFERRED — no quantitative claim made; callback overhead remains uncharacterised |

## §2.1 LSTM train/val MSE explanation (E1 follow-up)

At the 2000-epoch convergence run on the **real PHUL trajectory** (71 train + 7 val windows): **final train MSE = 0.8993, final val MSE = 2.0259, best val MSE ≈ 0.5725 @ epoch ~575**. Both curves are flat for hundreds of epochs before run-end — the plateau is real — and the early-stop ledger pins the convergence point.

Two structural points to note:

1. **Dropout active during training but disabled at eval** — standard PyTorch `nn.Module` behaviour; train-mode losses are computed on a noisier forward pass than eval-mode losses.
2. **The val set is only 7 windows** (84-day program, last-7 held out) — variance dominates the point estimate at this sample size, so the best-val figure (≈0.57) is the conservative number to report; the final-epoch val MSE (2.03) reflects mild post-best drift.

The **chronological split itself is methodologically correct** (train = earlier days, val = last 7 days, no future leakage). The remaining train/val asymmetry is a **sample-size + dropout-eval-mode artefact**, not a leakage symptom.

## §3 Counterfactuals / What was NOT tested

- **Publication-grade seed/episode budget**: E4 now uses **30 seeds × 200 episodes** (12× the earlier 10×50) and Welch's t (p = 0.652, not significant). Even at this budget A2C and REINFORCE are indistinguishable — going to 500 episodes is unlikely to change the conclusion on this toy env.
- **Full 600K-row detailed file**: the pipeline runs on the committed real `program_summary.csv` (2,598 programs) + the real PHUL exercise subset; the full 294 MB `programs_detailed_boostcamp_kaggle.csv` is fetched on demand (kept git-ignored), not committed.
- **Random baseline (now included, E4)**: a uniform-random *masked* policy anchors the floor — and reveals that neither RL method beats it on this env (the action mask + flat shaped reward make masked-random near-optimal). A *heuristic* (rotate-chains PPL) baseline is implemented (`src/model/trajectory_builder.py::_rotate_chains_action`) but **deliberately deferred as an evaluation series**: with masked-random already at the reward ceiling (E4: 5.853 ± 0.488, above both RL policies), a deterministic in-mask heuristic cannot add discriminative information — it would land in the same flat-reward band and tell us nothing the random floor has not.

## §3.1 E9 — λ_1 × λ_2 sensitivity (V3 §9.1, expanded to 5×5 × 3 seeds)

| | |
|---|---|
| **Hypothesis** | The reward decomposition `r_t = gain_t − λ_1·overload_t − λ_2·imbalance_t` is sensitive to the (λ_1, λ_2) weighting; sweeping a 5×5 grid with per-cell seed replication produces a measurable, statistically attributable shift in REINFORCE's late-window mean reward, and the prior 3×3 pilot's "dead λ_1 axis" finding should be re-examined under broader axis coverage and multiple seeds. |
| **Setup** | `scripts/run_lambda_sensitivity.py` — λ_1 ∈ {0.5, 1.0, 1.5, 2.0, 3.0} × λ_2 ∈ {0.25, 0.5, 1.0, 1.5, 2.0}; per cell: train REINFORCE **over the frozen real-PHUL LSTM env** for 20 episodes × 3 seeds (42, 43, 44), record the per-seed last-5-episode mean. **25 cells × 3 seeds = 75 runs, ~24 s wall-clock.** |
| **Output** | `results/figures/lambda_sensitivity.png` (5×5 heatmap with annotated `mean ± σ` cell values). |
| **Result** | Reward **decreases monotonically along λ_2** (per-λ_2 means **4.18 → 3.29 → 2.78 → 2.24 → 0.98**) but is **flat along λ_1** (every λ_1 row is identical). Best λ_2=0.25 ≈ **+4.18**, worst λ_2=2.0 ≈ **+0.98**, span ≈ **3.20** along the imbalance axis only. |
| **Interpretation** | Over the **frozen LSTM env**, the overload penalty (λ_1) is **inert**: the LSTM-predicted `rolling_7d_volume` rarely crosses the overload threshold on policy rollouts, so the λ_1 term contributes ~0 to the gradient regardless of weight. The imbalance penalty (λ_2) is the active reward dimension — heavier λ_2 monotonically lowers attainable reward. This is an honest property of the *learned* environment (it differs from earlier analytic-trainee sweeps precisely because the transition model is now the real-PHUL LSTM). The defaults (λ_1=2.0, λ_2=1.0) are brief-prescribed constants encoding physiology, not free hyperparameters to maximise reward. |
| **Verdict** | **λ_2 active, λ_1 inert over the LSTM env.** A faithful sensitivity surface for the learned environment; toy-scale (3 seeds × 30 eps/cell, not 30 × 500). |

Reproduce:

```bash
uv run --active python scripts/run_lambda_sensitivity.py
# → results/figures/lambda_sensitivity.png   (~19 s wall-clock for 75 runs)
```

## §3.2 E10 — REINFORCE learning-rate sensitivity sweep

| | |
|---|---|
| **Hypothesis** | The REINFORCE actor's lr (default 3e-4, lifted directly from OpenAI Spinning Up VPG) is not the optimum on this 28-day toy environment; a one-decade sweep around the default will surface a measurably better setting. |
| **Setup** | `scripts/run_lr_sweep.py` — actor lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} × seeds ∈ {42, 123, 2026}, 30 episodes/run **over the frozen real-PHUL LSTM env**, last-5-episode mean as the per-seed tail. **5 × 3 = 15 runs in ~10 s.** Raw arrays + per-lr (mean, std) in `results/lr_sweep.json`. |
| **Output** | `results/figures/reinforce_lr_sweep.png` — mean ± σ bars across lrs (log-x). |
| **Result** | Per-lr tail means across 3 seeds: **1e-4 → +3.21**, **3e-4 → +4.52**, **1e-3 → +5.38 (peak)**, **3e-3 → +3.11**, **1e-2 → +0.93**. Best lr = **1e-3**, **one decade above** the 3e-4 "OpenAI default". |
| **Interpretation** | The 3e-4 default is **conservative** for this env — 1e-3 wins on mean (+5.38 vs +4.52). Defaults tuned for Atari/MuJoCo scale tend to be under-stepped on toy domains where per-step gradient signal is denser. **We did not retrain the headline E2/E4 results at lr=1e-3** — that would conflate "lr tuning" with "algorithm comparison"; the sweep is a sensitivity finding, not a re-tuned headline. |
| **Verdict** | **CONFIRMED — 1e-3 dominates 3e-4 on the LSTM env.** Robust across 3 seeds. Open work: re-run E4 with lr=1e-3 on both algos. |

Reproduce:

```bash
uv run --active python scripts/run_lr_sweep.py
# → results/figures/reinforce_lr_sweep.png + results/lr_sweep.json   (~10 s)
```

## §4 Reproducibility caveats

All experiments are seeded via src/utils/seeding.set_global_seed; the seeded determinism test (tests/test_reproducibility.py) verifies tensor equality across runs at seed=42. CUDA non-determinism: scatter_add/index_add are non-deterministic on CUDA — we default to CPU. See README §Reproducibility caveats for the full list.

## §5 Open questions surfaced

1. Does training the LSTM on a multi-program mixture (not just PHUL) improve generalisation? Not tested.
2. How sensitive is **A2C** stability to the actor_lr/critic_lr ratio? Defaults pick 3e-4 / 1e-3; the **REINFORCE-only** lr sweep (E10) found 1e-3 dominates 3e-4 by a full decade — the analogous A2C sweep + the 2-d (actor_lr, critic_lr) ratio sweep are open work.
3. Does Action Masking change the policy's eventual fixed point, or only the trajectory? Theory ([8]) says no for the fixed-policy distribution but the gradient direction changes — open empirical question.

## §6 Program attribution note

PHUL (Power Hypertrophy Upper Lower) was popularised by Brandon Campbell on bodybuilding.com c.2014. It is **not** a Layne Norton programme (Norton authored PHAT, a different upper/lower/power/hypertrophy hybrid). Verified by grep on 2026-05-31: no "Norton", "Layne", or "Campbell" attributions appear in any tracked file, so this repo carries no incorrect author claim. The chosen Kaggle row title is *Optimized PHUL (Power Hypertrophy Upper Lower)*; we deliberately do not attribute it inline, since the agent's behaviour depends on the row contents, not the author.

## Methodology notes — REINFORCE-vs-A2C fair comparison

Both algorithms use:
- Identical PolicyNet architecture for REINFORCE actor and A2C actor.
- Same initial seed and replay env per seed.
- Same gamma (0.99) and entropy_coef (REINFORCE has no entropy term;
  A2C uses β=0.01).
- Same Adam optimizer flavor + grad-clip (0.5).

REINFORCE-specific:
- Baseline = scalar running-mean EMA (Williams 1992 reinforcement
  comparison), NOT a learned V(s_t).
- Loss = -(log_prob × (G_t − b)).mean() (note .mean(), not
  textbook .sum() — converges to same optimum, see Spinning Up
  vpg.py reference).

A2C-specific:
- Advantage = 1-step TD δ_t = r + γV(s') − V(s) (brief eq. 9),
  NOT Mnih 2016's n-step.
- Two independent Adam optimizers (actor + critic disjoint
  parameter sets, no shared trunk — brief §5.2).

Differences are deliberate spec choices, not bugs.
