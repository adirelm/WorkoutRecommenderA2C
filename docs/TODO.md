# TODO — phased task list (A3: REINFORCE + A2C on LSTM World Model)

Task tracker (§2.2). **Definition of Done (DoD)** for every build task:
behaviour implemented + a test asserting it + gates green (ruff,
≤150 lines/file, coverage ≥85%, uv-only) + an evidence pointer
(file path / test id / chart / notebook cell / screenshot) recorded +
committed with a §-referenced message.

**Ownership (§1.4 / §2.2.3c).** Solo project. Every row below is owned
end-to-end by the solo developer in the **architect** role (scope,
state/action design, reward design, acceptance criteria, sign-off);
the AI is the **implementer** against an approved PRD/PLAN edit. No
per-task hand-off — ownership is stated once here rather than per row.
The hybrid architecture (Architecture A runtime + ONE analysis
notebook) is itself an architect-decided commitment; see PRD §1.4
and PLAN §Architecture.

**Action space.** The discrete action set is **7** values:
`{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`.
All policy-net, env, masking, and notebook rows below use this 7-action set.

**Honesty stance (§M5).** This TODO contains **no numeric self-grade
prediction**. The self-grade declared on the cover sheet
(`adrl-001-ex03.pdf`) is the only place a numeric self-score lives;
internal docs (PRD / PLAN / TODO / PROMPTS / THEORY / TRACE / README)
describe what was built and its honest limitations, not what grade it
will earn.

## Build phases — status + evidence-pointer template
| # | Phase | Status | Evidence pointer template |
|---|---|---|---|
| **0** | **Must-fix preflight** (theory accuracy, repro plumbing, data-quality contract, action-masking ADR, honesty pass) | ⬜ | `docs/THEORY.md`, `src/utils/seeding.py`, `tests/test_reproducibility.py`, `tests/test_data_quality.py`, `docs/adr/ADR-004-action-masking.md`, commit Phase 0 |
| 1 | Repo bootstrap (pyproject, uv, ruff, pytest, CI, ≤150-line guardrail) | ⬜ | `pyproject.toml`, `.github/workflows/ci.yml`, commit Phase 1 |
| 2 | Theory artefact (`docs/THEORY.md` §1.1–§3.4 + brief equations 1/2/4/7/8/9/10/11/12/15/16/17 + derivations) | ⬜ | `docs/THEORY.md`, commit Phase 2 |
| 3 | State / action design docs + dataset gatekeeper | ⬜ | `docs/STATE_DESIGN.md`, `docs/ACTION_DESIGN.md`, `data/`, Phase 3 |
| 4 | Workout env (terminals, reward = gain − λ₁·overload − λ₂·imbalance) | ⬜ | `src/env/workout_env.py`, tests, Phase 4 |
| 5 | LSTM world model (Kaggle history → next-state predictor) | ⬜ | `src/world_model/lstm_world.py`, loss curve, Phase 5 |
| 6 | Policy net (1-layer FC, 128, Categorical head over 7 actions) | ⬜ | `src/policy/policy_net.py`, tests, Phase 6 |
| 7 | REINFORCE trainer (objective, log-trick, weighted-CE, baseline) | ⬜ | `src/training/reinforce_*`, tests, Phase 7 |
| 8 | A2C trainer (actor-critic, advantage) | ⬜ | `src/training/a2c_*`, tests, Phase 8 |
| 9 | RL-on-frozen-world-model pipeline + action masking service | ⬜ | `src/training/rl_on_world_model.py`, `src/services/action_mask_service.py`, Phase 9 |
| 10 | SDK facade (single entry point per CLAUDE.md §3) + CLI + GUI dashboard | ⬜ | `src/sdk.py`, `src/cli/`, `src/gui/`, Phase 10 |
| 11 | Analysis notebook (LaTeX next to LSTM loss / REINFORCE / A2C / comparison + §7.6 discussion) | ⬜ | `notebooks/analysis.ipynb`, Phase 11 |
| 12 | Docs + COST_ANALYSIS + PROMPTS + submission PDF + collaborator invite + README index | ⬜ | `README.md`, `docs/COST_ANALYSIS.md`, `docs/PROMPTS.md`, `adrl-001-ex03.pdf`, Phase 12 |

Phase gates (all must be green before phase is marked ✅):
ruff zero violations · every `.py` ≤150 LOC · coverage ≥85% · uv-only ·
SDK is single business-logic entry · notebook is a *consumer* of the
SDK (no parallel implementation) · LSTM weights frozen during RL phase ·
seeding deterministic across two consecutive forward passes (M2) ·
data-quality report emitted to `results/data_quality_report.txt` (M3).

## Initial task list

Columns: `id | content (imperative) | activeForm (progressive) | phase | est-LOC-Δ | maps-to-PRD-req-id | acceptance-criterion`

### Phase 0 — Must-fix preflight (apply BEFORE any code in Phases 1+)

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T-MF1 | Author `docs/THEORY.md` transcribing brief equations 1, 2, 4, 7, 8, 9, 10, 11, 12, 15, 16, 17 verbatim in LaTeX; include eq.15 reward `r_t = gain_t − λ₁·overload_t − λ₂·imbalance_t` (λ₁=2.0, λ₂=1.0) with `gain_t = 0.7·progress_t + 0.3·variety_t`, `variety = 1 − JS(muscle_dist, target)`, `overload_t = max(0, (vol_7d − 1.2·baseline)/baseline)^1.5`, `imbalance_t = variance(muscle_share_14d)`; add "how it maps to `src/`" cross-reference table per M1 | Authoring THEORY.md with brief equations + reward decomposition | 0 | +220 docs | M1,1.1,1.2,1.4,2.4,2.5,3.2,3.3,3.4 | All 12 equations render in LaTeX; cross-ref table names exact `src/` module for each; reward equation matches `src/env/workout_env.py` symbols 1:1 |
| T-MF2 | Author `src/utils/seeding.py` exporting `set_global_seed(seed: int)` that seeds `random`, `numpy`, `torch` (CPU+CUDA), `PYTHONHASHSEED`, sets `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`; add `tests/test_reproducibility.py` asserting tensor equality across two seeded LSTM forwards; add README "Reproducibility caveats" subsection naming CUDA non-determinism (scatter_add, index_add), mixed-precision drift, multi-worker DataLoader shuffle order, MPS LSTM kernel determinism caveat per M2 | Authoring seeding utility + reproducibility test + README caveats | 0 | +110 + 60 test + 80 docs | M2 | `pytest tests/test_reproducibility.py` passes; two LSTM forward passes with same seed produce bitwise-identical tensors on CPU; README §Reproducibility names all four caveats verbatim |
| T-MF3 | Author PRD §1.5.0 **Data Quality Contract** + `tests/test_data_quality.py` with cases: `negative_reps_row_dropped`, `plank_reclassified` (time-encoded → seconds/3), `rest_day_inserted` (zero-volume row per brief §7.2.4 week-day gap), `total_volume_nonneg`; cleaning rules: (a) negative reps/sets → drop with logged count to `results/data_quality_report.txt` (NOT impute), (b) time-encoded reps trap (Plank/Hold/Bridge/Wall-sit/L-sit/Hollow-body OR reps>60 AND sets≤1 → `reps_equiv = seconds / 3`, 3s/rep tunable in `config/config.yaml`), (c) rest-day insertion; notebook cell prints `N rows raw → N after each rule` per M3 | Authoring data-quality contract + tests + notebook cleaning report | 0 | +140 docs + 120 test | M3,TR3 | All 4 test cases pass; `results/data_quality_report.txt` lists drop counts per rule; notebook §Data-cleaning cell prints rule-by-rule funnel |
| T-MF4 | Author `docs/adr/ADR-004-action-masking.md` + notebook §7.6.1 outline covering: (1) why mask via logits→−∞ before softmax (preserves policy-gradient unbiasedness vs post-hoc rejection), (2) citation `[Huang & Ontañón 2022, FLAIRS]` using brief's bib ref [8], (3) worked example: trainee did Legs yesterday with `soreness_legs > 0.8` → mask action 3 (Legs) next day, (4) safety-vs-exploration trade-off ("Guardrails as humans in the loop next to optimisation"), (5) `ActionMaskService` masked-softmax code outline; test asserts masked-logit positions → 0 probability per M4 | Authoring ADR-004 + notebook §7.6.1 outline | 0 | +140 docs + 60 nb | M4,TR2,1.1 | ADR has all 5 sections; cited as `[8]`; worked example uses action index 3 = Legs; `test_masked_logits_yield_zero_probability` passes |
| T-MF5 | Strip every "~92", "~93", "demonstrate", "earns the same ~92", "I built it → I understand it" from internal docs (PRD, PLAN, TODO, PROMPTS, THEORY, TRACE, README); replace "demonstrate the progression" with "attempt to reproduce the §7 pipeline on a single synthetic trainee, with explicit limitations"; add "Honest Limitations" section to README §X and PRD §11 listing: (a) plan-content data ≠ real workout outcomes (brief §7.6 Q4 — answered head-on), (b) single synthetic trainee = no population generalisation, (c) LSTM fit on plan-derived sequences may memorise periodisation rather than learn dynamics, (d) reward is hand-designed and may not align with real-world training goals; numeric claims (convergence, comparison) cite seed, episode count, mean ± std — never bare adjectives per M5 | Stripping self-grade predictions + adding Honest Limitations sections | 0 | -40 docs + 100 docs | M5,A1A1,TR10 | `grep -E "~9[0-9]|earns the same|I built it.*I understand"` returns zero matches across `docs/` and `README.md`; PRD §11 and README §X each list all 4 limitations verbatim |

### Phase 1+ — Build tasks

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T01 | Initialise repo skeleton (pyproject, uv.lock, ruff config, pytest config, CI workflow, .gitignore, .env-example) | Initialising repo skeleton | 1 | +200 | TR5,1.3 | `uv run pytest` and `uv run ruff check` both exit 0 on empty suite |
| T02 | Add `tool.coverage` `fail_under=85` and `ruff` line-length config; enforce ≤150 LOC via pre-commit hook script | Adding coverage gate and 150-LOC pre-commit hook | 1 | +40 | (gate) | Hook rejects a deliberately bloated test file in CI dry-run |
| T03 | Write `docs/THEORY.md §1.1` — policy as probability vector π_θ(a\|s) with θ notation | Writing THEORY §1.1 (policy as probability vector) | 2 | +60 docs | 1.1 | Section cites brief verbatim; reviewed-by-architect checkbox ticked |
| T04 | Write `docs/THEORY.md §1.2` — objective J(θ)=E_{τ~π_θ}[Σγ^t r_t] with discount derivation | Writing THEORY §1.2 (discounted objective) | 2 | +60 docs | 1.2 | LaTeX renders in notebook + GitHub; matches `src/training/objective.py` symbols |
| T05 | Write `docs/THEORY.md §1.3` — state vector spec referencing the Cart 4-tuple analogy and the workout-trainee state vector | Writing THEORY §1.3 (state vector) | 2 | +50 docs | 1.3 | Cross-links to `docs/STATE_DESIGN.md` |
| T06 | Write `docs/THEORY.md §1.4` — POMDP, history h_t, why we need RNN/LSTM "world model" | Writing THEORY §1.4 (POMDP + history) | 2 | +60 docs | 1.4 | Forward-references §7 world-model section |
| T07 | Write `docs/THEORY.md §2.5` — log-derivative trick ∇p = p∇log p with model-free estimator justification | Writing THEORY §2.5 (log-derivative trick) | 2 | +50 docs | 2.5 | Includes step-by-step derivation, not just final form |
| T08 | Write `docs/THEORY.md §3.3` — baseline b(s) independence from a → unbiased gradient (control variate proof) | Writing THEORY §3.3 (unbiased baseline proof) | 2 | +50 docs | 3.3 | Proof closes with E_a[∇log π · b(s)] = b(s)·∇Σ π = 0 |
| T09 | Write `docs/THEORY.md §3.4` — bias-variance trade-off framing, baseline reduces variance with zero bias | Writing THEORY §3.4 (bias-variance) | 2 | +40 docs | 3.4 | Forward-references `results/variance_comparison.png` |
| T10 | Author `docs/STATE_DESIGN.md` — justify which Kaggle dataset columns become state features (volume normalisation, 7d/14d lookback windows, muscle-distribution share, soreness proxy, week_index, day_in_cycle) | Authoring STATE_DESIGN.md | 3 | +120 docs | 1.3,TR2 | Each chosen column has a 1-line rationale; rejected columns also listed |
| T11 | Author `docs/ACTION_DESIGN.md` — define the **7-action** discrete set `{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}` with grading-defensible rationale; reference brief's "one-hot of sampled action" framing and ADR-004 masking | Authoring ACTION_DESIGN.md (7 actions) | 3 | +90 docs | TR2,M4 | Action space documented as 7 entries; one-hot dim = 7; references ADR-004 |
| T12 | Build `src/data/kaggle_loader.py` — fetch + cache Kaggle workout dataset (Adnane Louardi 600K+) deterministically | Building Kaggle data loader | 3 | +120 | TR3 | `test_loader_returns_dataframe_with_expected_columns` |
| T13 | Add `src/data/gatekeeper.py` — schema/NaN/date-monotonic + data-quality contract enforcement (delegates to M3 rules); rejects unclean data before training touches it | Adding data gatekeeper | 3 | +90 | TR3,M3 | `test_gatekeeper_rejects_nan_and_unordered_dates` + `test_gatekeeper_invokes_quality_contract` |
| T14 | Implement `src/env/workout_env.py` — 12-d state vector, reset, step (7-action set), reward eq.15 (`gain − λ₁·overload − λ₂·imbalance`), terminal conditions (28-day horizon / failure / dataset exhausted) | Implementing workout env | 4 | +140 | 1.3,2.2,M1 | `test_episode_termination_three_cases` + `test_reward_decomposition_matches_eq15` |
| T15 | Test: episode loop random-init → sample action → update per brief §2 | Testing episode loop (init/sample/update) | 4 | +50 test | 2.1 | `test_episode_loop_random_init_sample_update` passes |
| T16 | Implement `src/world_model/lstm_world.py` — LSTM next-state predictor on Kaggle history with configurable window | Implementing LSTM world model | 5 | +140 | 1.4,TR3 | `test_lstm_history_window` + loss curve saved to `results/lstm_loss.png` |
| T17 | Train LSTM world model end-to-end; emit `results/lstm_loss.png` and persist weights (cite seed + epoch count + final val-loss mean±std in PRD) | Training LSTM world model | 5 | +60 | TR3,M5 | Loss curve recorded; final val-loss reported with seed + mean±std, never bare adjective |
| T18 | Add `src/training/rl_on_world_model.py` — wires policy onto frozen LSTM (sets `requires_grad=False` on every LSTM param); integrates `ActionMaskService` from ADR-004 | Wiring RL onto frozen LSTM + action masking | 9 | +130 | TR3,TR4,M4 | `test_lstm_params_frozen_during_rl` + `test_pipeline_lstm_then_rl` + `test_masked_logits_yield_zero_probability` |
| T19 | Implement `src/policy/policy_net.py` — 1-layer FC, 128 hidden units, Categorical head over **7 actions** (architecture from brief, TR5) | Implementing policy net (7-action head) | 6 | +90 | 1.1,TR5 | `test_policy_net_hidden_size_128_single_layer` + `test_policy_outputs_probability_distribution_over_7_actions` |
| T20 | Implement `src/training/objective.py` — discounted return Σγ^t r_t helper used by both REINFORCE and A2C | Implementing discounted-return helper | 7 | +60 | 1.2 | `test_discounted_return_formula` checks against hand-computed example |
| T21 | Implement `src/training/credit_assignment.py` — propagate episode return to every visited (s,a) | Implementing credit assignment | 7 | +60 | 2.3 | `test_return_propagated_to_all_steps` |
| T22 | Implement `src/training/reinforce_update.py` — θ ← θ + α Σ_t ∇log π(a_t\|s_t) G_t | Implementing REINFORCE update | 7 | +90 | 2.4 | `test_reinforce_gradient_update_matches_formula` (compares to autograd reference) |
| T23 | Implement `src/training/loss.py` — REINFORCE as one-hot-weighted cross-entropy (×G_t) using `torch.nn.functional.cross_entropy` over 7-action one-hot | Implementing weighted-CE loss | 7 | +70 | 2.6,2.7 | `test_reinforce_loss_equals_weighted_cross_entropy` + `test_uses_torch_crossentropy_backend` |
| T24 | Implement `src/training/baseline.py` — running-mean baseline; subtract b from G_t before weighting | Implementing baseline subtraction | 7 | +60 | 3.2 | `test_baseline_subtraction_in_update` |
| T25 | Test (numerical): log-derivative identity ∇p = p·∇log p holds for a small softmax over 7 logits | Testing log-derivative identity numerically | 7 | +40 test | 2.5 | `test_log_derivative_identity_numerically` within 1e-5 |
| T26 | Test (statistical): baseline does not bias the gradient (Monte-Carlo over many seeds) | Testing baseline unbiasedness | 7 | +50 test | 3.3 | `test_baseline_unbiased_gradient` 95% CI brackets the no-baseline mean |
| T27 | Implement `src/training/reinforce_trainer.py` — episode loop, sampler (Categorical over 7 actions), end-of-episode update, history logger | Implementing REINFORCE trainer | 7 | +140 | 2.1,2.4,TR6 | Runs to completion on env; reward history reported with seed + mean±std (no bare "converges") |
| T28 | Worked-example artefact: the "10-step trajectory with one bad action at step 8" from brief §3.1 — render in notebook | Authoring §3.1 worked-example artefact | 11 | +60 nb | 3.1 | `test_noise_example_unfair_punishment` |
| T29 | Empirical variance-reduction chart: REINFORCE vs REINFORCE+baseline over N seeds (report mean±std, not bare adjectives) | Producing variance-reduction chart | 11 | +60 | 3.4,M5 | `results/variance_comparison.png` exists; `test_variance_reduction_empirical`; caption cites seed + N + mean±std |
| T30 | Implement `src/policy/value_head.py` — critic head V_φ(s) sharing torso with actor (or separate FC) | Implementing critic value head | 8 | +80 | (A2C) | Outputs scalar; gradient flows |
| T31 | Implement `src/training/a2c_update.py` — advantage Â_t = G_t − V_φ(s_t); actor uses log π · Â over 7-action one-hot, critic minimises (G_t − V)^2 | Implementing A2C update | 8 | +110 | (A2C) | `test_a2c_advantage_formula` + `test_a2c_critic_mse` |
| T32 | Implement `src/training/a2c_trainer.py` — synchronous A2C training loop on frozen world model | Implementing A2C trainer | 8 | +140 | (A2C),TR1 | `test_both_agents_train_to_completion` (both REINFORCE and A2C reach end) |
| T33 | Implement `src/services/action_mask_service.py` per ADR-004 — masked-softmax over 7 logits, soreness/recency rule (mask action 3 Legs if `soreness_legs > 0.8`); plug into both REINFORCE and A2C samplers | Implementing ActionMaskService (7-action masked softmax) | 9 | +100 | M4,TR2 | `test_masked_logits_yield_zero_probability` + `test_legs_masked_when_soreness_high` |
| T34 | Implement `src/sdk.py` — single facade exposing `train_reinforce()`, `train_a2c()`, `train_world_model()`, `load_world_model()`, `get_metrics()`, `save_run()`, `clean_data()` | Implementing SDK facade | 10 | +140 | (§3 contract) | `test_sdk_is_single_entry_point` (CLI, GUI, notebook all import only SDK) |
| T35 | Implement `src/cli/menu.py` + `main.py` — terminal-driven training & inspection (≤20 LOC `main.py`) | Implementing CLI menu | 10 | +130 | (UX) | `uv run main.py` boots; menu lists clean-data / train-world-model / train-reinforce / train-a2c / compare |
| T36 | Implement `src/gui/dashboard.py` — live loss/reward/advantage curves, 7-bin action histogram, muscle-share donut | Implementing GUI dashboard | 10 | +140 | (UX) | Headless render test passes; histogram has 7 bins |
| T37 | Generate `results/comparison/reinforce_vs_a2c.png` — both reward curves overlaid, identical seed schedule, mean±std bands | Generating REINFORCE-vs-A2C comparison chart | 11 | +60 | TR1,M5 | Chart exists, axes labelled, legend present, caption cites seeds + episode count + mean±std |
| T38 | Author `notebooks/analysis.ipynb` §LSTM — LSTM loss section with LaTeX next to plot; consumes SDK only | Authoring notebook §LSTM | 11 | +200 nb | TR3 | Notebook imports SDK only (no model code re-defined) |
| T39 | Author notebook §REINFORCE — derivation of θ ← θ + α ∇log π · G_t alongside reward graph | Authoring notebook §REINFORCE | 11 | +200 nb | 2.4,TR1 | LaTeX block precedes the plot cell |
| T40 | Author notebook §A2C — derivation of advantage Â_t and joint actor/critic loss alongside training graph | Authoring notebook §A2C | 11 | +200 nb | (A2C),TR1 | LaTeX block precedes the plot cell |
| T41 | Author notebook §7.6 Comparison + Discussion — narrative answering the brief's 5 questions (LSTM realism, policy collapse, A2C-vs-REINFORCE stability, dataset limitations, hypothetical physiological signals) + §7.6.1 action-masking sub-section per M4 | Authoring notebook §Comparison/Discussion + §7.6.1 | 11 | +200 nb | TR1,TR10,M4 | All 5 brief questions answered with seed+mean±std evidence; §7.6.1 cites ADR-004 |
| T42 | Add `docs/COST_ANALYSIS.md` — token budget envelope, prompts run, $ spent, lessons (with AI-tooling cost section per A1A3) | Authoring COST_ANALYSIS.md | 12 | +120 docs | (§11),A1A3 | Includes table of prompt-bundles × tokens × cost |
| T43 | Author `docs/PROMPTS.md` — verbatim prompts used (architect→implementer trail per §1.4) with human judgment / pushback / caught-mistakes annotations (A1A8) | Authoring PROMPTS.md | 12 | +200 docs | (§1.4),A1A8 | Every prompt mapped to a commit hash; decisions annotated |
| T44 | Run final gate sweep — ruff clean, all `.py` ≤150 LOC, coverage ≥85% (statement + branch), notebook executes top-to-bottom, reproducibility test green | Running final gate sweep | 12 | 0 | (gates),M2 | `make check` (or scripted equivalent) exits 0 |
| T45 | Tag submission commit `assignment-3`, push branch | Tagging submission and pushing branch | 12 | +0 | (submission) | Tag pushed |

### Phase 12 — Submission package

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T-SUB1 | Export submission PDF from `biu-rl07-ex01-template.docx` → `adrl-001-ex03.pdf` at repo root; cover sheet is the **only** place the numeric self-grade lives (per §M5 / approved decision 5) | Exporting submission PDF (cover-sheet self-grade) | 12 | +0 | R4,R5,M5 | `adrl-001-ex03.pdf` exists at repo root with filled template fields and cover-sheet self-grade; no other doc carries a numeric self-grade prediction |
| T-SUB2 | Invite `rmisegal` as read-only collaborator on the A3 GitHub repo (`WorkoutRecommenderA2C`); record the GitHub invitation ID in `docs/SUBMISSION.md` | Inviting rmisegal as read-only collaborator | 12 | +0 | R1 | Invitation sent with **Read** role; invitation ID recorded in `docs/SUBMISSION.md`; screenshot attached |
| T-SUB3 | `README.md` at repo root links to `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, `docs/THEORY.md`, `docs/PROMPTS.md`, `docs/COST_ANALYSIS.md`, `docs/TRACE.md`, `notebooks/analysis.ipynb`, `adrl-001-ex03.pdf`, and the Kaggle dataset URL | Adding README index of all submission artefacts | 12 | +60 docs | R6,TR7 | `grep -E "PRD\.md\|PLAN\.md\|TODO\.md\|THEORY\.md\|PROMPTS\.md\|COST_ANALYSIS\.md\|TRACE\.md\|analysis\.ipynb\|adrl-001-ex03\.pdf\|kaggle\.com"` against README.md returns 10 matches |
