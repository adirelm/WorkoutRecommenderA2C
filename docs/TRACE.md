# TRACE — Requirement Traceability Matrix

This document is the single source of truth that ties every requirement we are
working against back to a binding source, a planned deliverable, and a planned
test. Four source streams feed it: (1) the assignment-3 brief by Dr. Yoram Segal
(Bar-Ilan Vibe-Coding Workshop, L07, May 2026) — sections §1 through §3 are
covered by binding text in this matrix and §4–§7 (Actor-Critic, A2C, dataset,
analysis) appear via transcript-derived rows TR1–TR10 until the full §4–§7
verbatim text is loaded; (2) the lecturer's live walkthrough transcript and the
explicit grading hints he gave verbally (e.g. "REINFORCE vs A2C side-by-side
with graphs", "freeze the LSTM during RL", "1-layer FC 128 neurons", "sample
from Categorical not argmax"); (3) the standing-rule set R1–R10 carried over
across all workshop assignments (group code `adrl-001`, submission file name
`adrl-001-ex03.pdf`, deadline 2026-06-10 23:59, GitHub read-only collaborator,
late penalty 5pt/24h); and (4) the seven CLAUDE.md hard constraints (CL1–CL7)
that this repository inherits from A1/A2 — 150-LOC file ceiling, TDD with 85%
coverage, OOP base/subclass with no duplication, all algorithm-relevant
parameters in `config/config.yaml`, zero Ruff violations, `uv`-only package
manager, single repository with assignment-specific branch.

Each row gives the requirement id, the source stream, the binding text (kept
in the original language where the brief is in Hebrew, so the wording cannot
drift), the deliverable file or document section where the requirement is
satisfied, the test id planned to enforce it, and a priority of MUST / SHOULD /
NICE. The action space throughout this trace is the 7-action discrete set
`{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`
fixed by ADR-002 and ADR-004; any deliverable that mentions an "action count"
refers to this 7-action set. The matrix below carries 78 rows across eight
source streams (Brief §1–§3, Brief §7.2 dataset, Brief §7.6 analysis Q1–Q5
plus §7.6.1 action-masking, Brief §7.7 deliverable artefacts, transcript
grading hints TR1–TR10, A1 lecturer-feedback avoid/repeat A1A1–A1A8 / A1R1–A1R3,
standing rules R1–R10, and CLAUDE.md hard constraints CL1–CL7). Open items
that the matrix flags but does not yet bind are listed at the bottom under
"Known gaps to revisit" — those rows must be closed before any submission tag
is cut.

| req_id | source | binding_text | deliverable_file_or_section | test_id_planned | priority |
|---|---|---|---|---|---|
| 1.1 | Brief §1 | מהי מדיניות: וקטור של הסתברויות — π_θ(a\|s) מתאר וקטור הסתברויות; הפרמטרים של המדיניות מסומנים ב-θ. | docs/THEORY.md §1.1 + src/policy/policy_net.py (Categorical head over 7 actions) | test_policy_outputs_probability_distribution | MUST |
| 1.2 | Brief §1 | פונקציית המטרה J(θ) = E_{τ~π_θ}[ Σ γ^t r_t ]. | docs/THEORY.md §1.2 + src/training/objective.py | test_discounted_return_formula | MUST |
| 1.3 | Brief §1 | המצב כווקטור של מספרים: דוגמת ה-Cart — ארבעה ערכים מתארים את המצב. | docs/THEORY.md §1.3 + docs/STATE_DESIGN.md (pointer to PRD §1.5 + ADR-002, 12-d state vector spec) | test_state_vector_shape | MUST |
| 1.4 | Brief §1 | מקומי מול מצטבר: RNN/LSTM/Transformer; POMDP; רצף תצפיות h_t; "מודלי עולם" שאליהם נחזור בפרק 7. | docs/THEORY.md §1.4 + src/model/lstm_world.py | tests/test_lstm_world.py::test_forward_output_shape | MUST |
| 2.1 | Brief §2 | תהליך האימון: אפיזודה מתחילה ברשת מאותחלת אקראית; דגימת פעולה לפי הסתברויות; שיפור עד סוף האפיזודה. | src/services/reinforce_trainer.py (episode loop) | test_episode_loop_random_init_sample_update | MUST |
| 2.2 | Brief §2 | מתי נגמרת אפיזודה: (1) הגעתי לפרס, (2) נכשלתי, (3) נגמרו המצבים. MDP timeout only — cases (1) and (2) collapsed per ADR-005 (no scalar reward target; action-masking in ADR-004 prevents injury-equivalent failure events). | src/env/workout_env.py (case 3 only) + docs/adr/ADR-005-terminal-conditions.md | test_done_after_episode_length_steps + test_action_mask_prevents_injury_event (case-1 N/A — see ADR-005) | MUST |
| 2.3 | Brief §2 | קרדיט משותף — סך הניקוד מהאפיזודה מחולק לכל הפעולות שלאורכה. | src/services/reinforce_helpers.py (compute_returns) | test_return_propagated_to_all_steps | MUST |
| 2.4 | Brief §2 | θ ← θ + α Σ_t ∇_θ log π_θ(a_t\|s_t) G_t — REINFORCE update. | src/services/reinforce_helpers.py + src/services/reinforce_trainer.py | test_reinforce_gradient_update_matches_formula | MUST |
| 2.5 | Brief §2 | טריק נגזרת הלוג: ∇p = p ∇log p; model-free sample-based estimator. | docs/THEORY.md §2.5 (derivation) | test_log_derivative_identity_numerically | MUST |
| 2.6 | Brief §2 | הקשר ל-Cross Entropy: One-Hot של הפעולה שנדגמה; ההפרש מוכפל ב-G_t. | src/services/reinforce_helpers.py (weighted CE) | test_reinforce_loss_equals_weighted_cross_entropy | MUST |
| 2.7 | Brief §2 | שקילות מתמטית ל-Cross Entropy: implementable via PyTorch CE on GPU, ללא אופטימייזר מיוחד. | src/services/reinforce_helpers.py + README §Implementation Notes | test_uses_torch_crossentropy_backend | MUST |
| 3.1 | Brief §3 | מסלול של 10 צעדים, פעולה חריגה אחת הפילה את הניקוד למינוס 10; הראשונות נענשות. | docs/THEORY.md §3.1 (worked example) | test_noise_example_unfair_punishment | SHOULD |
| 3.2 | Brief §3 | הפתרון: החסרת הממוצע — קו בסיס b: θ ← θ + α Σ_t ∇_θ log π_θ(a_t\|s_t)(G_t − b). | src/services/baseline.py | test_baseline_subtraction_in_update | MUST |
| 3.3 (inferred) | Brief §3 | קו בסיס b(s) לא תלוי ב-a → לא מטה את הגרדיאנט (control variate). | docs/THEORY.md §3.3 (proof) + tests | test_baseline_unbiased_gradient | MUST |
| 3.4 (inferred) | Brief §3 | Bias-Variance Trade-off: כאן צמצום שונות ללא הטיה. | docs/THEORY.md §3.4 (variance_comparison.png deferred — not generated) | test_variance_reduction_empirical | SHOULD |
| D1 | Brief §7.2 | Kaggle dataset adnanelouardi/600k-fitness-exercise-and-workout-program-dataset; ODbL 1.0 license; cite in submission | docs/PRD.md §1.3 + docs/adr/ADR-002 + README §References | test_dataset_slug_in_config | MUST |
| D2 | Brief §7.2 | Download procedure: kaggle CLI + cached parquet under data/raw/ (instructions/-only; not in repo) | src/data/kaggle_client.py + docs/PRD.md §3.1 F1 | test_kaggle_client_caches_to_parquet | MUST |
| D3 | Brief §7.2 | Two CSV schemas: program_summary.csv (2598 rows) + fitness_exercises.csv (605k rows) joinable on 'title' | src/data/preprocessor.py | test_csv_schema_columns_present | MUST |
| D4 | Brief §7.2.3 | Negative reps/sets caveat: NOT corruption — encodes seconds for timed exercises; convert via seconds_per_rep=3.0 (config) | src/data/preprocessor.py (apply_data_quality_contract) + config/config.yaml data_quality | test_negative_sets_reps_treated_as_seconds | MUST |
| D5 | Brief §7.2.4 (eq. 13) | Single-trainee trajectory builder: filter chosen_program (≥8wk, Full Gym, 45-120min); daily aggregation total_volume_t = Σ(sets·reps); REST DAY insertion; trajectory (s_1..s_T) | src/data/aggregator.py | test_daily_aggregation_eq13 + test_rest_day_inserted | MUST |
| D6 | Brief §7.7 | Submission cites dataset URL + key files + license + chosen program (PHUL primary) | README §Dataset + docs/PRD.md §10 | test_readme_cites_dataset_url | MUST |
| DA1 | Brief §7.1.2 Part A | Formal written MDP definition (state/action/reward/transition) + pipeline pseudocode | docs/THEORY.md §1.1-1.4 + docs/PRD.md §1.5-1.7 | test_part_a_mdp_writeup_present | MUST |
| DA2 | Brief §7.3.1 Part C | LSTM loss curves (train + val) + temporal-pattern discussion | results/figures/lstm_loss.png + notebooks/analysis.ipynb cell 3 | test_lstm_loss_chart_exists | MUST |
| DA3 | Brief §7.4.3 Part D | REINFORCE training Loss curve + avg-return graph + weekly-load variance diagnostic | results/figures/reinforce_rewards.png + notebooks/analysis.ipynb cell 5 (reinforce_variance.png deferred — not generated) | test_reinforce_reward_chart_exists | MUST |
| DA4 | Brief §7.5.1 Part E | A2C actor+critic Loss curves + side-by-side REINFORCE vs A2C comparison | results/figures/a2c_training.png + results/figures/comparison.png + notebooks/analysis.ipynb cells 7-8 | test_a2c_and_comparison_charts_exist | MUST |
| DA5 | Brief §7.7 | Submission-list roll-up: preprocessing code + trainee trajectory + LSTM impl + REINFORCE impl + A2C impl + summary discussion | tests/test_submission_manifest.py asserts each artefact path | test_submission_manifest_all_present | MUST |
| DA6 | Brief eq. 15 + eq. 17 | Reward eq.15 unit-test + Advantage eq.17 unit-test (first-class binding, not buried in ADR) | src/env/reward.py + src/services/a2c_trainer.py | test_spec_eq15_reward_decomposition + test_spec_eq17_advantage | MUST |
| Q1 | Brief §7.6 Q1 | Did the LSTM learn realistic temporal structure? | notebooks/analysis.ipynb §discussion-q1 + docs/EXPERIMENTS.md | n/a (doc artifact) | MUST |
| Q2 | Brief §7.6 Q2 | Does the policy collapse to a small action subset? | notebooks/analysis.ipynb §discussion-q2 — action histogram | test_action_diversity_above_floor | MUST |
| Q3 | Brief §7.6 Q3 | Did A2C outperform REINFORCE (stability/convergence)? | notebooks/analysis.ipynb §discussion-q3 (REINFORCE vs A2C side-by-side) | n/a | MUST |
| Q4 | Brief §7.6 Q4 | Limitations: training-program data ≠ real outcomes — answer head-on | notebooks/analysis.ipynb §discussion-q4 + README §Honest Limitations (a) | n/a | MUST |
| Q5 | Brief §7.6 Q5 | Which physiological measurements (HR, soreness, strength logs) would improve the system? | notebooks/analysis.ipynb §discussion-q5 | n/a | MUST |
| AM1 | Brief §7.6.1 | Action Masking proposal: logits→−∞ before softmax; Huang & Ontañón 2022 [8]; worked example (legs-day → mask Legs next day) | docs/adr/ADR-004-action-masking.md + src/env/action_mask.py + notebooks/analysis.ipynb §action-masking | test_action_mask_zeros_prob_for_invalid | MUST |
| AM2 | Brief §7.6.1 | Simplified-reward acknowledgement + hierarchical-decision limitation discussion | notebooks/analysis.ipynb §reward-simplification + README §Honest Limitations (d) | n/a | MUST |
| TR1 | Transcript / grading_hints | MUST DELIVER: explicit REINFORCE vs A2C side-by-side comparison with graphs (01:20:34). | results/figures/comparison.png + docs/ANALYSIS.md | test_both_agents_train_to_completion | MUST |
| TR2 | Transcript / grading_hints | STATE/ACTION DESIGN IS GRADED: justify which dataset columns become actions vs states (01:18:27). | docs/STATE_DESIGN.md (pointer to PRD §1.5 + ADR-002) + docs/ACTION_DESIGN.md (pointer to PRD §1.6 + ADR-004, 7-action set) | n/a (doc artifact) | MUST |
| TR3 | Transcript / grading_hints | TWO-STAGE PIPELINE: (a) LSTM world model on Kaggle historical data; (b) REINFORCE/A2C on top. | src/model/lstm_world.py + src/services/rl_on_world_model.py — BOUND — src/services/reinforce_trainer.py wires PolicyNet onto frozen LSTMWorldModel via LSTMEnvAdapter (Phase 3+4 join) | test_pipeline_lstm_then_rl | MUST |
| TR4 | Transcript / grading_hints | FREEZE THE LSTM during RL phase (01:20:14). | src/services/rl_on_world_model.py (requires_grad=False) — BOUND — model.freeze() called pre-RL in Phase 4 trainer path; LSTMEnvAdapter __init__ asserts is_frozen | test_lstm_params_frozen_during_rl | MUST |
| TR5 | Transcript / grading_hints | SIMPLE NET: 1-layer FC, 128 neurons (01:06:59). | src/model/policy_net.py (architecture) + config/config.yaml | test_policy_net_hidden_size_128_single_layer | MUST |
| TR6 | Transcript / grading_hints | STOCHASTIC POLICY DURING TRAINING: sample (Categorical), not argmax. | src/model/policy_net.py (sample at train) | test_training_uses_categorical_sample_not_argmax | MUST |
| TR7 | Transcript / grading_hints | Check latest assignment PDF before starting (uploaded late). | docs/SOURCES.md (assignment PDF hash + date) | n/a (process artifact) | SHOULD |
| TR8 | Transcript / grading_hints | DEADLINE FLEXIBILITY: lecturer is open if asked; late penalties not fixed. | docs/SUBMISSION.md note + email-trail evidence if requested | n/a | NICE |
| TR9 | Transcript / grading_hints | Future lectures workshop-style — schedule accordingly. | docs/PLAN.md (schedule section) | n/a | NICE |
| TR10 | Transcript / grading_hints | ANALYSIS SECTION (§7.6) matters — graphs + comparison beyond "does it run". | docs/ANALYSIS.md + results/figures/*.png | test_analysis_artifacts_exist | MUST |
| A1A1 | A1 avoid | Do NOT express strong/unqualified confidence in self-assessment — be honest about limitations. | README §Self-Assessment + docs/REFLECTION.md (limitations) | n/a | MUST |
| A1A2 | A1 avoid | Do NOT submit only the polished final — include thinking process, abandoned approaches, negative results. | docs/shared/PROMPTS.md (multi-pass narrative + dead-ends) | n/a | MUST |
| A1A3 | A1 avoid | Include AI tooling cost (tokens, subscription share, dev hours, AI-rework tax) not just cloud. | docs/COST_ANALYSIS.md | test_cost_analysis_has_ai_tooling_section | MUST |
| A1A4 | A1 avoid | Show automated enforcement (CI, pre-commit, lint gates, coverage gates) — not manual review. | .github/workflows/ci.yml + .pre-commit-config.yaml | test_ci_runs_lint_and_coverage_gate | MUST |
| A1A5 | A1 avoid | Explicitly demonstrate separation of concerns + extensibility for a new dev. | docs/ARCHITECTURE.md (layer diagram) + ADRs | test_module_boundaries_no_crosslayer_imports | MUST |
| A1A6 | A1 avoid | Answer each generic rubric bullet with concrete project-specific evidence. | docs/RUBRIC_SELFCHECK.md (per-bullet evidence) | n/a | MUST |
| A1A7 | A1 avoid | Do NOT erode the six praised areas (planning, docs, config/security, testing, UI/UX, version mgmt). | README + docs/PLAN.md + config/config.yaml + tests/ + ui/ + CHANGELOG.md | test_six_praised_areas_smoke | MUST |
| A1A8 | A1 avoid | PROMPTS.md must show human judgment, pushback, caught mistakes, redesigns — not passive transcript. | docs/shared/PROMPTS.md (annotated decisions) | n/a | MUST |
| A1R1 | A1 repeat | Keep PRD/ADR-style artifacts enabling independent onboarding. | docs/PRD.md + docs/adr/*.md | n/a | MUST |
| A1R2 | A1 repeat | Maintain professional README/setup/usage docs. | README.md (setup, usage, results) | n/a | MUST |
| A1R3 | A1 repeat | Externalized config (config.yaml), secrets discipline, multi-environment. | config/config.yaml + .env.example + docs/SECURITY.md | test_no_secrets_in_repo | MUST |
| R1 | Standing rule R1 | Submit to GitHub; share with rmisegal@gmail.com (read-only collaborator). | docs/SUBMISSION.md + GitHub repo settings screenshot | n/a (process) | MUST |
| R2 | Standing rule R2 | Each member submits separately on Moodle; same repo URL; per-person timing. | docs/SUBMISSION.md (per-member checklist) | n/a | MUST |
| R3 | Standing rule R3 | 8-char group code adrl-001 (semester-long). | README header + cover sheet + adrl-001-ex03.pdf filename | test_group_code_present_in_readme | MUST |
| R4 | Standing rule R4 | Submission PDF = lecturer template filled → adrl-001-ex03.pdf (no edits to template fields). | docs/submission/adrl-001-ex03.pdf | test_pdf_filename_and_template_integrity | MUST |
| R5 | Standing rule R5 | Self-grade declared in cover sheet (the cover sheet `adrl-001-ex03.pdf` is the only place the numeric self-grade is stated). | adrl-001-ex03.pdf cover sheet only | test_cover_sheet_self_grade_present | MUST |
| R6 | Standing rule R6 | README at repo root + docs/PRD.md + docs/PLAN.md + docs/TODO.md. | README.md + docs/PRD.md + docs/PLAN.md + docs/TODO.md | test_required_root_docs_exist | MUST |
| R7 | Standing rule R7 | §1.4 architect/implementer contract visible in CLAUDE.md + per-commit message. | CLAUDE.md §1.4 + git log (§-tagged commits) | test_claude_md_has_section_1_4 + test_commits_reference_sections | MUST |
| R8 | Standing rule R8 | Deadline 2026-06-10 23:59. | docs/PLAN.md (milestone) + docs/SUBMISSION.md | n/a (calendar) | MUST |
| R9 | Standing rule R9 | Lecturer scoring weight: principle UNDERSTANDING > training quality. | docs/THEORY.md + docs/ANALYSIS.md (concept depth) | n/a | MUST |
| R10 | Standing rule R10 | Late penalty 5pts/24h. | docs/SUBMISSION.md (risk register) | n/a | SHOULD |
| CL1 | CLAUDE.md hard constraint | File size ≤ 150 LOC per .py file. | All src/*.py + tests/*.py | test_no_file_exceeds_150_loc | MUST |
| CL2 | CLAUDE.md hard constraint | TDD RED→GREEN→REFACTOR; 85%+ coverage. | tests/ + coverage report | test_coverage_at_least_85_percent | MUST |
| CL3 | CLAUDE.md hard constraint | OOP: BaseAgent → REINFORCEAgent / A2CAgent; no duplication. | src/agents/base_agent.py + reinforce_agent.py + a2c_agent.py | test_agents_inherit_base_no_dup_logic | MUST |
| CL4 | CLAUDE.md hard constraint | No hardcoded values: all algorithm-relevant params in config/config.yaml. | config/config.yaml + src/config/loader.py | test_no_hardcoded_hyperparams_in_src | MUST |
| CL5 | CLAUDE.md hard constraint | Zero Ruff violations on src/ tests/ main.py. | CI ruff step | test_ruff_clean | MUST |
| CL6 | CLAUDE.md hard constraint | UV only — no pip/conda. | pyproject.toml + uv.lock + README run instructions | test_uv_lock_present_no_requirements_txt | MUST |
| CL7 | CLAUDE.md version control | Same repo as A1/A2; branch assignment-3; version 1.2.x. | git branch + CHANGELOG.md | test_branch_and_version_consistent | MUST |

## Known gaps to revisit

These items were flagged during traceability assembly and are not yet bound to
a final binding text or a deliverable in the matrix above. Each one must be
closed (either by adding a row or by an explicit "out of scope" note in the
PRD) before the submission tag is cut.

- Brief §3 transcript was truncated mid-row (req 3.2 source text cuts off after
  the formula). The unbiased-baseline proof (3.3) and bias-variance trade-off
  (3.4) had to be inferred from the lecture; confirm against the full brief
  before locking the 3.3 / 3.4 ids.
- Brief sections §4–§7 (Actor-Critic / A2C, advantage A(s,a)=Q−V, value head,
  entropy bonus, dataset/Kaggle spec, §7.6 analysis deliverables) were not in
  the initial ingest. The matrix covers the §1–§3 explicit requirements plus
  the transcript A2C ask (TR1), but per-equation A2C requirements (advantage
  formula, critic loss, entropy regularisation, GAE if mentioned) cannot be
  traced until §4+ are supplied verbatim. The five must-fix corrections (M1–M5)
  applied at planning time partially close this by binding eq. 15 / 16 / 17
  through ADR-003 and docs/THEORY.md, but the §7.6 analysis questions still
  need explicit per-question rows.
- The Kaggle dataset identity (Adnane Louardi 600K+ Fitness Exercise & Workout
  Program Dataset, two files `fitness_exercises.csv` and `program_summary.csv`)
  is referenced in transcript hints but not yet bound by a verbatim row here;
  docs/STATE_DESIGN.md must cite the exact dataset link, license (ODbL 1.0),
  and the chosen primary trainee program (PHUL, with GZCLP and nSuns 5/3/1 as
  fallbacks) from the assignment PDF.
- Transcript "nuances" field in the input was truncated mid-string. Could not
  extract nuance-derived requirements (e.g. normalisation, reward-shaping
  warnings, episode-length choice). Re-pull nuances before final lock.
- A1-feedback "repeat" list was truncated mid-sentence. Items beyond R3
  (testing discipline, UI/UX, version mgmt explicit asks) are not yet mapped
  to rows; expand once the full A1 lecturer feedback is available.
- No explicit requirement yet for environment / reward design of the workout
  env itself beyond the M1 binding (action-space cardinality is fixed at 7 via
  ADR-002 and ADR-004, reward weights via ADR-003). Confirm episode length
  (planned: 28 days), termination conditions, and reset distribution against
  the brief before TR2 is closed.
- No explicit requirement for the seed / reproducibility policy in inputs
  (carried over from A1 / A2 norms but not bound here). The M2 must-fix adds
  `src/utils/seeding.py` and `tests/test_reproducibility.py`; add a REPRO row
  once the binding text is finalised in the PRD.
- No explicit requirement for the evaluation protocol (train / val / test split
  of the synthetic trainee trajectory, walk-forward windowing, out-of-sample
  evaluation). Critical for TR3 (two-stage pipeline) credibility; flag for
  clarification before notebook §7.7 is written.
- Hyper-parameter sweep / sensitivity analysis is implied by TR1 (comparison
  graphs) but not bound. Clarify whether single-seed comparison suffices or
  multi-seed mean ± std is required by §7.6.
- Compute / runtime budget is not specified. Affects whether A2C can use
  parallel workers; needs a self-decision before locking the TR1 deliverable
  scope.

## Phase-3 freshness sweep (2026-05-31)

Path: TRACE planned `src/world_model/` but as-built layout is `src/model/` for all LSTM modules. Updated rows: 1.4, F4, F6, DA6-eq.17, TR3, TR4. Underlying tests pass identically; only the path/test-id strings drifted between the planning workflow and Phase-3 implementation.

## Phase-4 freshness sweep (2026-05-31)

The Phase-4 build placed PolicyNet under `src/model/` and the REINFORCE trainer + helpers under `src/services/` (rather than the planned `src/policy/` + `src/training/` paths). Updated rows: TR5, TR6, 2.1, 2.4, 2.6, 2.7, 3.2. Three filename collapses: `reinforce_update.py` + `loss.py` → `src/services/reinforce_helpers.py` (rows 2.4, 2.6, 2.7). DA3 explicitly deferred to Phase 7. TR3/TR4 closed.

## Phase-5 freshness sweep (2026-05-31)

Phase 5 placed Actor-Critic under `src/model/actor_critic.py` and A2C trainer/helpers/comparator under `src/services/` (not the planned `src/training/` paths). Updated rows: 2.3 (credit assignment now in reinforce_helpers + a2c_helpers), DA6 (eq.17 advantage now in src/services/a2c_helpers.py::compute_advantages_td), TR1 (REINFORCE-vs-A2C side-by-side now in src/services/comparator.py with ComparisonResult dataclass). DA4 (REINFORCE-vs-A2C comparison chart + notebook cell 7-8) explicitly DEFERRED TO PHASE 7 (analysis notebook).

## Phase-6 freshness sweep (2026-05-31)

Phase 6 placed the SDK facade at `src/sdk/sdk.py` (class `WorkoutSDK`, not `TrainingSDK` as planning text used). Updated rows: F15, F16, F17. F16 entry-point invariant test moved from planned `tests/test_architecture.py` to as-built `tests/test_sdk_facade.py`. F17 CLI verb test moved from `tests/test_cli.py` to `tests/test_cli_menu.py`. CLI is a numeric stdin menu (not argparse), so "verbs" are routed by integer choice; one test asserts all six verbs dispatch to the SDK.

## Phase-9 freshness sweep (2026-05-31)

Phase 9 added a Streamlit GUI surface (10 pages under src/gui/). New TRACE rows:
| G1 | Architect pivot from PRD §1.3 "no GUI" | docs/adr/ADR-006-streamlit-gui-framework.md | n/a (architect decision) | MUST |
| G2 | 10 GUI pages bound to brief §7.x and SDK verbs | src/gui/pages/*.py | tests/test_gui_pages_*.py | MUST |
| G3 | Bar-Ilan blue theme | src/gui/theme.py | tests/test_gui_accessibility.py (WCAG AA contrast) | SHOULD |
| G4 | Live training charts via observer callbacks | src/gui/callbacks.py | tests/test_gui_callbacks.py | MUST |
| G5 | Action Masking demo (bonus beyond brief minimum) | src/gui/pages/08_action_masking.py | tests/test_gui_page_action_masking.py | NICE |
| G6 | CLI verb 7 launch-gui | src/cli/menu.py | tests/test_cli_menu.py | MUST |
| G7 | Screenshot capture | scripts/capture_gui_screenshots.py + docs/assets/gui_*.png | tests/test_gui_screenshots_exist.py | NICE |
