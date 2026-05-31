# PRD — WorkoutRecommenderA2C (Assignment 3 product requirements)

The project-wide requirements document for Assignment 3 of the Bar-Ilan
Vibe Coding Workshop. Per-component design lives in `PLAN.md`; the literal
prompts and AI workflow live in `shared/PROMPTS.md`; the phased task list with
definition-of-done lives in `TODO.md`.

> Teaching artefact, **not medical or training advice**. Synthetic trainee, offline
> simulator, no biometric integration. See §11 for the full honest-limitations list.

---

## 1. Objective & scope

### 1.1 Headline goal

Attempt to reproduce the §7 pipeline of Assignment 3 — **dynamics model (LSTM
world model) → policy-gradient control (REINFORCE) → actor-critic control
(A2C)** — on a single synthetic trainee, with explicit limitations (§11). The
deliverable is a reproducible offline pipeline that fits the trainee's
day-to-day response to training actions, then trains two policies that
recommend the next day's session under a reward that blends weekly progress,
muscle-distribution variety, overload safety, and long-horizon balance.

### 1.2 Three artefacts, one SDK

The submission ships three trainable components behind one Python facade
(`TrainingSDK`):

1. **LSTM transition model** — fits `p(s_{t+1} | s_t, a_t)` from a synthetic
   logbook (brief §7.2). This is the "world model" the policies roll out against.
2. **REINFORCE agent** (Williams 1992) — episodic Monte-Carlo policy gradient
   over the LSTM-rolled-out environment (brief §7.4).
3. **A2C agent** (Mnih et al. 2016, synchronous variant) — actor + value
   critic, advantage `A_t = r_t + γV_ψ(s_{t+1}) − V_ψ(s_t)` (brief §7.5).

Plus one head-to-head comparison artefact (brief §7.6) and a recommendation
endpoint (brief §7.7).

### 1.3 In scope
- Synthetic trainee data generator (28-day logbook, seeded).
- 12-dimensional state vector (see §1.5 — encoding **C2_moderate_12d**).
- **7 discrete actions** (see §1.6 — Rest / Push / Pull / Legs / Full-body /
  Conditioning / Mobility).
- LSTM transition model + offline rollout (no live trainee in the loop).
- REINFORCE + A2C trained against the LSTM rollout.
- Action masking via masked softmax (logits → −∞), per ADR-004.
- LaTeX-annotated analysis notebook (brief §7.7 deliverables).
- Public SDK + thin CLI; **plus a beautiful Streamlit GUI surface (Phase 9, ADR-006) covering all 6 SDK verbs + Theory & Discussion pages**.

### 1.4 Out of scope
- Live heart-rate / wearables / sensor integration (see §11).
- Multi-trainee population models (single trainee, seeded).
- Online learning / on-policy data collection from a real human.
- Mobile / web frontend (deferred — A4 candidate).
- Efficacy claims against real strength-training literature.

### 1.5.0 Data Quality Contract

The Kaggle dataset (Adnane Louardi 600K+ Fitness Exercise & Workout Program)
contains structured noise the brief warns about in §7.2.3–§7.2.4. The
contract below is enforced in `src/data/cleaner.py` before any state is
ever computed; the notebook prints a cleaning report (N rows raw → N after
each rule), and `results/data_quality_report.txt` is committed alongside
training results so the cleaning is auditable.

**(a) Negative reps/sets** — **drop** the row with a logged count; do **not**
impute. Imputation would mask a real upstream parsing bug. Failures land in
`results/data_quality_report.txt` keyed by `(title, week, day, exercise)`.

**(b) Time-encoded reps (the §7.2.3 trap).** Some rows encode duration
(seconds) in the `reps` field for isometric / static exercises. The rule:

> If `exercise_name ∈ {Plank, Hold, Bridge, Wall-sit, L-sit, Hollow-body, …}`
> **OR** (`reps > 60` **AND** `sets ≤ 1`) → treat `reps` as seconds and
> convert via `reps_equiv = seconds / SECONDS_PER_REP`.

`SECONDS_PER_REP` defaults to `3.0` and lives in `config/config.yaml` so the
assumption is tunable — not buried in source.

**(c) Rest days (brief §7.2.4)** — insert **zero-volume rows** at every
`(week, day)` gap in the selected program. A rest day is a first-class
observation (`fatigue` decays, `streak_days_trained` resets) — silently
omitting it would teach the LSTM that days never break.

**Verification.** `tests/test_data_quality.py` carries the four named cases:
`negative_reps_row_dropped`, `plank_reclassified`, `rest_day_inserted`,
`total_volume_nonneg`.

### 1.5 State design (decision: **C2_moderate_12d**)

A 12-dimensional continuous vector chosen as the moderate-information design
point — rich enough to express soreness asymmetry and recovery dynamics, narrow
enough that the LSTM can fit it in the available training budget. The vector is:

1. `fatigue` ∈ [0, 1] — aggregate systemic fatigue.
2. `soreness_push` ∈ [0, 1] — chest / anterior delts / triceps.
3. `soreness_pull` ∈ [0, 1] — lats / rhomboids / biceps / rear delts.
4. `soreness_legs` ∈ [0, 1] — quads / hamstrings / glutes / calves.
5. `soreness_core` ∈ [0, 1] — abs / obliques / spinal erectors.
6. `readiness` ∈ [0, 1] — self-reported subjective readiness proxy.
7. `rolling_7d_volume` ∈ ℝ⁺ — sum of session volumes over trailing 7 days.
8. `streak_days_trained` ∈ ℤ⁺ — consecutive non-rest days.
9. `days_since_last_rest` ∈ ℤ⁺ — recovery-debt proxy.
10. `muscle_balance_push_vs_pull` ∈ [-1, 1] — 14-day exposure asymmetry.
11. `adherence_signal` ∈ [-1, 1] — planned-vs-actual residual.
12. `weekly_progress` ∈ [0, 1.2] — `weekly_volume_so_far / weekly_target` (capped).

### 1.6 Action design

**Seven discrete actions** — the cardinality is fixed and load-bearing for
ADR-004 (Action Masking) and the policy network output dimension. Each
action produces a deterministic-mean stochastic state delta on the LSTM
transition. Summarised (full deltas in `PLAN.md` §6):

| id | name         | primary muscles                                  | min  | intensity |
|----|--------------|--------------------------------------------------|------|-----------|
| 0  | Rest         | none (systemic recovery)                          | 0    | low       |
| 1  | Push         | chest, anterior delt, triceps                     | 40-60| med       |
| 2  | Pull         | lats, rhomboids, biceps, posterior delt, traps    | 40-60| med       |
| 3  | Legs         | quads, hamstrings, glutes, calves                 | 50-75| high      |
| 4  | Full-body    | compound multi-chain                              | 45-70| med-high  |
| 5  | Conditioning | systemic cardio (HR-driven)                       | 25-45| med       |
| 6  | Mobility     | low-load, range-of-motion                         | 20-30| low       |

Action `Rest` resets `streak_days_trained` to 0 and decays `rolling_7d_volume`
by `1/7` of itself. Training actions add session volume and elevate soreness on
the targeted chain, with peak soreness offset to `t+24h` in the LSTM target.

**Masking** (ADR-004). Where domain knowledge says a candidate action is
unsafe (e.g. `soreness_legs > 0.8` with `Legs` trained the previous day),
the corresponding logit is set to `−∞` *before* the softmax — preserving
the policy-gradient unbiasedness result (Huang & Ontañón 2022, FLAIRS).
Worked example and code outline live in `docs/adr/ADR-004-action-masking.md`
and notebook §7.6.1.

### 1.7 Reward design

Reward at step `t` (brief eq. 15):

> **r_t = gain_t − λ_1 · overload_penalty_t − λ_2 · imbalance_penalty_t**

with `λ_1 = 2.0` (overload weighted heavier — keep policy safe) and
`λ_2 = 1.0` (imbalance weighted lighter — let policy explore variety).

**Gain** (brief eq. 16):

> `gain_t = w_p · progress_t + w_v · variety_t`
>
> `progress_t = clip(weekly_volume_so_far / weekly_target, 0, 1.2) − clip(prev, 0, 1.2)`
>
> `variety_t = 1 − JS(muscle_share_14d, target_muscle_dist)`
>
> Defaults `w_p = 0.7`, `w_v = 0.3`.

Pure prev-day volume delta would create a perverse incentive to ramp volume
monotonically and ignore recovery — exactly the behaviour the overload penalty
fights, producing a high-variance tug-of-war. We pick a blend of
**(b) progress-toward-weekly-target** and **(c) variety-bonus**: (b) is
bounded, smooth, and aligns with the brief's prescribed weekly stimulus;
(c) directly rewards covering the muscle distribution the trainer prescribed.
Capping progress at 1.2× target prevents the policy from farming reward by
overshooting volume — overshoot becomes a net negative once the overload term
activates. Jensen-Shannon (symmetric, bounded in [0, 1]) keeps the variety
term on the same scale as progress so weights are interpretable.

**Overload penalty** (brief eq. 17):

> `overload_penalty_t = max(0, (vol_7d − 1.2 · baseline) / baseline) ** 1.5`
>
> `baseline = user_baseline_7d_volume`, a 28-day trailing **median** (robust
> to outliers), recomputed weekly.

The 1.5 exponent makes overload superlinear past the threshold so the policy
cannot trade a tiny gain bump for a large overload risk.

**Imbalance penalty** (brief eq. 15 second penalty term):

> `imbalance_penalty_t = variance(muscle_share_14d)`

We pick **variance** (over the symmetric KL-divergence-from-uniform
alternative) because variance is smoothly differentiable everywhere in the
simplex and gives a clean autograd path through the policy; KL would
introduce log-zero singularities when a muscle share is zero, which is the
modal value in early episodes. λ_2 = 1.0 makes the imbalance term a soft
nudge rather than a constraint — the policy is allowed to specialise
within a week, only paying when the 14-day window stays lopsided.

### 1.8 GUI Surface (Phase 9 — Streamlit)

Per the brief §1.4 architect decision recorded in
[ADR-006](adr/ADR-006-streamlit-gui-surface.md), Assignment 3 ships a
**Streamlit GUI** on top of the existing `TrainingSDK` facade. The GUI is
**purely a consumer** of `src.sdk.sdk` (CLAUDE.md §3) — every page imports
SDK verbs only; no direct `src.env` / `src.model` / `src.services` reach-in.
This keeps the §5 hybrid architecture intact: the runtime layered core
(`data/env/model/services/sdk/cli`) stays gradeable, ruff-clean,
coverage-gated, and 150-LOC-bounded; the GUI is one more *surface* on the
same SDK, sibling to the analysis notebook.

**Ten pages** (each a separate file under `src/gui/pages/`, ≤ 150 LOC):

1. **Home** — project overview, brief §7 alignment, navigation hub.
2. **Data** — invokes `prepare_data()`, shows the 28-day logbook, cleaning
   report, and the §1.5.0 Data Quality Contract verification badges.
3. **LSTM** — `train_world_model()` with **live per-epoch loss curves**
   (Plotly, refreshed via `st.empty()` + a `src/gui/callbacks.py` observer).
4. **REINFORCE** — `train_reinforce()` with **live per-episode reward** +
   baseline trace.
5. **A2C** — `train_a2c()` with live actor/critic loss + entropy + advantage.
6. **Compare** — `compare()` REINFORCE-vs-A2C plot (mean ± 1σ over N seeds).
7. **Recommend** — interactive `recommend()` endpoint with **12 state
   sliders** (one per state dim from §1.5); shows the chosen action,
   per-action probability vector (length 7), predicted next state, and
   expected reward.
8. **ActionMasking** — visualises `action_mask(state, history)` (ADR-004):
   which of the 7 actions are masked, *why* (the violated safety rule),
   and the resulting masked-softmax distribution.
9. **Theory** — brief §7.6 / §7.6.1 derivations (policy gradient,
   advantage estimator, masked-softmax unbiasedness) **rendered with
   KaTeX-rendered LaTeX** via `st.latex(...)`, sourced from
   `docs/THEORY.md` so the equations stay single-source-of-truth.
10. **Discussion** — brief §7.6 honest-limitations narrative (§11 of this
    PRD), the REINFORCE-vs-A2C reading, and the ablation summary.

**Theme.** Bar-Ilan blue palette (primary `#003D7A`, accent `#FFCD00`,
light bg `#F5F7FA`); dark mode via Streamlit's built-in toggle.
`streamlit-extras` for optional polish; Plotly ≥ 5.20 for interactive
charts; matplotlib charts (when used) go through `src/gui/charts.py`
factories — never inline `plt` calls.

**Live updates.** Trainer code is **not modified** for the GUI — the
per-epoch / per-episode refresh uses the observer pattern via
`src/gui/callbacks.py` + Streamlit's `st.empty()` placeholders. Session
state keys are namespaced `gui.<page>.<key>` and accessed through the
typed `src/gui/state.py` helper.

**Testing.** All pages tested headlessly with `streamlit.testing.v1.AppTest`
(Streamlit ≥ 1.40) — same `≥ 85 %` coverage gate (N2) and `≤ 150 LOC`
ceiling (N1) as the rest of the runtime code.

---

## 2. Stakeholders & responsibilities

Per `CLAUDE.md §1.4` (architect/implementer contract), this section makes the
Human ↔ AI boundary explicit for A3.

| Concern                                       | Human (architect) | AI (implementer) |
|-----------------------------------------------|-------------------|------------------|
| PRD scope, KPIs, brief §7 alignment           | ✅                 | —                |
| Architecture decision (A vs B vs C → hybrid)  | ✅                 | —                |
| State / action / reward design (§1.5–§1.7)    | ✅                 | —                |
| Trace-matrix MUSTs (§7 acceptance criteria)   | ✅                 | —                |
| Code generation against approved spec         | —                 | ✅                |
| Refactor within `TrainingSDK` public API      | —                 | ✅                |
| Test scaffolding from written acceptance      | —                 | ✅                |
| Docstring + analysis-notebook prose drafts    | —                 | ✅                |
| Lint / format fixes                           | —                 | ✅                |

Stakeholders:
- **Student (Adir)** — architect, final commit author, rubric self-scorer.
- **Lecturer** — grader; scores against the published rubric (UNDERSTANDING
  weighted above training quality — see architecture justification in §5).
- **Claude (AI subagent fleet)** — implementer under §1.4 contract.

---

## 3. Functional requirements

Each `F#` is traceable to brief §7.x. The trace matrix (`docs/TRACE.md`)
carries 53 rows and 46 MUSTs.

### 3.1 Data
- **F1** (brief §7.2). Generate a 28-day synthetic logbook of `(state, action,
  next_state)` triples from a seeded stochastic trainee simulator built on
  one selected Kaggle program (PHUL primary; GZCLP / nSuns 5/3/1 fallback).
- **F2**. Persist the logbook as parquet under `data/synthetic/` with a seed
  manifest so any run is bit-reproducible from `seed → logbook → models`.
  **Status: DEFERRED — synthetic trainee is generated in-memory per run
  (deterministic via seed), no parquet cache needed for the 28-day rollout
  horizon. Implement only if rollout horizons grow to weeks.**
- **F3**. Provide a chronological train/val split (last 7 days held out for
  LSTM validation).
- **F3a**. Apply the §1.5.0 Data Quality Contract to the raw Kaggle CSVs
  before state computation; emit `results/data_quality_report.txt`.
  **Status: DEFERRED — data ingest validation lives inline in
  `src/data/program_filter.py` (raises on no match) +
  `tests/unit/test_program_filter.py` asserts the contract. Separate `cleaner.py`
  + `data_quality_report` would be needed only for multi-source ingest.**

### 3.2 World model
- **F4** (brief §7.3). Train an LSTM `f_φ: (s_t, a_t) → s_{t+1}` on the 21-day
  training window with teacher forcing.
- **F5**. Report training and validation MSE loss curves on the 12-d state.
- **F6**. Expose `world_model.rollout(s0, policy, horizon)` returning a
  trajectory of length `horizon` for downstream RL training.

### 3.3 REINFORCE
- **F7** (brief §7.4). Implement REINFORCE (Williams 1992) with a softmax
  policy over the 7 discrete actions and a learned-baseline variant
  (running mean of episodic returns) to reduce gradient variance.
- **F8**. Train against the LSTM rollout; log per-episode return.
- **F9**. Emit the REINFORCE reward curve required by brief §7.7.

### 3.4 A2C
- **F10** (brief §7.5). Implement synchronous A2C: shared trunk, actor head
  (softmax) + critic head (scalar V), advantage `A_t = r_t + γV_ψ(s_{t+1}) − V_ψ(s_t)`,
  entropy bonus on the actor loss, critic minimises ½δ²_t MSE.
- **F11**. Train against the LSTM rollout with the same seed schedule as
  REINFORCE (so the comparison is paired).
- **F12**. Emit the A2C training graph required by brief §7.7.

### 3.5 Comparison & recommendation
- **F13** (brief §7.6). Produce the REINFORCE-vs-A2C comparison plot
  (mean return ± 1σ across N seeds) on a held-out start-state distribution.
- **F14** (brief §7.7). Produce the LaTeX-annotated analysis notebook
  (`notebooks/analysis.ipynb`) that imports `TrainingSDK`, renders the four
  required plots (LSTM losses, REINFORCE rewards, A2C training, REINFORCE vs
  A2C comparison), and discusses the policy-gradient and advantage updates
  next to the plots.
- **F15**. `TrainingSDK.recommend(s_t)` returns a `WorkoutRecommendation`
  containing the chosen action, the policy's per-action probability vector
  (length 7), the predicted next state from the LSTM, and the expected reward.

### 3.6 SDK + CLI
- **F16**. All training, comparison, and inference flow through `TrainingSDK`
  — no notebook or CLI bypasses the facade (`CLAUDE.md §3`).
- **F17**. CLI verbs: `prepare-data`, `train-world-model`, `train-reinforce`,
  `train-a2c`, `compare`, `recommend`.

---

## 4. Non-functional requirements

Inherited from `CLAUDE.md` Hard Constraints, plus A3-specific additions.

- **N1**. Every `.py` file ≤ 150 lines (hard, enforced by a CI check).
- **N2**. TDD: tests written before implementation; coverage **≥ 85 %** on the
  `src/` package (run: `uv run pytest tests/ --cov=src --cov-report=term-missing`).
- **N3**. Zero ruff violations on `src/`, `tests/`, `main.py`, `scripts/`.
- **N4**. **uv-only** dependency management; `uv sync --dev` reproduces the
  environment from `uv.lock`.
- **N5**. Deterministic seeds. `src/utils/seeding.py` exposes
  `set_global_seed(seed: int)` that seeds Python `random`, numpy, torch (CPU
  + CUDA), sets `PYTHONHASHSEED`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True,
  warn_only=True)`. `tests/test_reproducibility.py` asserts tensor equality
  across two seeded LSTM forwards. Reproducibility caveats (CUDA
  `scatter_add` / `index_add` non-determinism, mixed-precision drift,
  multi-worker DataLoader shuffle order, MPS LSTM kernel) are named
  explicitly in the README.
- **N6**. No hardcoded algorithm parameters; all hyperparameters live in
  `config/config.yaml` (§4 of `CLAUDE.md`), including `seconds_per_rep`,
  `lambda_1`, `lambda_2`, `weekly_target`, `overload_exponent`.
- **N7**. The analysis notebook is a **consumer** of `TrainingSDK` — it must
  not redefine training logic, dataclasses, or losses. Verified by a test
  that grep-asserts the absence of `class .*(nn.Module)` and
  `def .*loss` inside `notebooks/`.
- **N8**. Total notebook execution time ≤ 10 minutes on a 2024 MacBook Air
  M-series CPU (no GPU required).

---

## 5. Architecture overview

### 5.1 Decision: **hybrid (layered SDK + analysis notebook)**

A hybrid wins because the lecturer scores UNDERSTANDING > training quality.
Architecture A (the layered-SDK candidate) gives the rubric the engineering
spine it expects (SDK facade, OOP inheritance, DRY, 150-LOC ceiling, 85 %
coverage gate, ruff-clean, uv-only) and is the only candidate that honestly
satisfies `CLAUDE.md §3` (SDK as single entry point) and `§1.4`
(architect/implementer contract with clear public API). But A alone misses
the LaTeX-next-to-results pedagogical surface that lets the reader connect
the equations in the brief to the curves in `results/`. Candidate C
(notebook-first) alone fails the OOP, coverage, ruff, and 150-LOC gates.
Candidate B (monolithic) alone collapses the layer boundaries and risks a
god-module `sdk.py`.

The hybrid keeps A's runtime code (`data / env / model / services / sdk / cli`)
as the gradeable, testable, CI-enforced core, and adds **one** analysis
notebook (`notebooks/analysis.ipynb`) that imports `TrainingSDK` and renders
the §7.7 deliverables (LSTM loss curves, REINFORCE reward graph, A2C training
graph, REINFORCE-vs-A2C comparison, discussion) with LaTeX derivations of the
policy-gradient and advantage updates next to the plots. The notebook is a
*consumer* of the SDK, not a parallel implementation — so coverage, ruff, and
the 150-LOC gate continue to apply to the runtime code unchanged.

### 5.2 Pointer

Full layered design + ADRs (incl. ADR-004 Action Masking) + dependency
diagram → [PLAN.md](PLAN.md).

---

## 6. Public API (`TrainingSDK`)

The single facade. UI / notebook / CLI depend **only** on this surface.

```python
class TrainingSDK:
    def __init__(self, config: Config, seed: int) -> None: ...

    # F1, F3 (F2 + F3a DEFERRED — see §3.1)
    def prepare_data(self) -> LogbookHandle: ...

    # F4–F6
    def train_world_model(
        self, logbook: LogbookHandle
    ) -> WorldModelResult: ...

    # F7–F9
    def train_reinforce(
        self, world_model: WorldModelHandle, episodes: int
    ) -> ReinforceResult: ...

    # F10–F12
    def train_a2c(
        self, world_model: WorldModelHandle, episodes: int
    ) -> A2CResult: ...

    # F13
    def compare(
        self, reinforce: ReinforceResult, a2c: A2CResult, seeds: int = 5
    ) -> ComparisonReport: ...

    # F15
    def recommend(
        self, state: State, policy: PolicyHandle, world_model: WorldModelHandle
    ) -> WorkoutRecommendation:
        """next_state in the recommendation comes from world_model.rollout(state, ...)"""

    # Action masking surface (ADR-004)
    def action_mask(self, state: State, history: list[int]) -> np.ndarray:
        """Returns a bool array of shape (ACTION_COUNT,) — UIs need this to
        display "valid recommendations only"."""


@dataclass(frozen=True)
class WorldModelResult:
    handle: WorldModelHandle
    history: LSTMTrainHistory   # train_loss, val_loss, best_epoch — consumed by notebook Phase 7


@dataclass(frozen=True)
class ReinforceResult:
    handle: PolicyHandle
    history: REINFORCEHistory   # episodes_run, rewards, losses, baseline, seed — consumed by §7.4 comparison plots
    config: REINFORCEConfig     # hyperparameters used (lr, gamma, episodes, baseline_alpha, ...)


class WorldModelHandle:
    def rollout(
        self, initial_state: State, policy: Callable[[State], int], horizon: int
    ) -> np.ndarray:
        """Returns (horizon, STATE_DIM) array of predicted states. Used by
        analysis notebook (Phase 7) for §7.7 LSTM-loss-curve and
        rollout-trajectory plots."""
```

All return types are `@dataclass(frozen=True)`. No method exposes a `nn.Module`
to callers — handles are opaque and serializable. The action-probability
vector inside `WorkoutRecommendation` is fixed-length 7.

PRD §6 updated 2026-05-31 (Phase 3 validation) to expose `LSTMTrainHistory`,
world-model rollout, action mask, and the `world_model` argument to
`recommend()`. Phase 6 (SDK facade) implements these signatures.

PRD §6 updated 2026-05-31 (Phase 4 REINFORCE landing) to inline `ReinforceResult`
with its `REINFORCEHistory` + `REINFORCEConfig` fields — symmetric with the
`WorldModelResult` / `LSTMTrainHistory` pattern above. The dataclasses ship
in `src/services/types.py` and are the analysis-notebook contract for §7.4
reward-curve and baseline-trace plots. `REINFORCETrainer` itself stays an
implementation detail behind `TrainingSDK.train_reinforce()`.

---

## 7. Acceptance criteria & DoD

### 7.1 Per-requirement DoD

| Req  | Acceptance criterion                                                          | Evidence pointer            |
|------|-------------------------------------------------------------------------------|-----------------------------|
| F1   | `prepare_data` produces a 28-row parquet with the 12-d state schema           | `tests/test_data.py::test_logbook_shape` |
| F2   | **DEFERRED** — synthetic trainee is in-memory per run (seeded); no parquet cache needed at 28-day horizon | n/a (deferred) |
| F3   | 21/7 chronological split returned in order                                    | `tests/test_data.py::test_chrono_split` |
| F3a  | **DEFERRED** — ingest validation inline in `src/data/program_filter.py` (raises on no match); separate `cleaner.py` + report only needed for multi-source ingest | `tests/unit/test_program_filter.py` asserts the contract |
| F4   | LSTM train MSE on the 21-day window decreases monotonically (5-epoch median)  | `tests/test_world_model.py::test_train_loss_decreases` |
| F5   | Validation loss curve emitted and saved as `results/lstm_losses.png`          | `notebooks/analysis.ipynb` cell 3 |
| F6   | `rollout(s0, π_uniform, 14)` returns a `(14, 12)` array, no NaNs              | `tests/test_world_model.py::test_rollout_shape` |
| F7   | REINFORCE update equals `∇log π(a|s) · (G_t − b)` symbolically (unit test)    | `tests/test_reinforce.py::test_grad_form` |
| F8   | Mean episodic return over last 20 % of training > first 20 % (with seed+std reported) | `tests/test_reinforce.py::test_learning_progress` |
| F9   | `results/reinforce_rewards.png` emitted                                        | `notebooks/analysis.ipynb` cell 5 |
| F10  | A2C advantage equals `r + γV(s') − V(s)` symbolically                          | `tests/test_a2c.py::test_advantage_form` |
| F11  | A2C trained with same seed schedule as REINFORCE                               | `tests/test_a2c.py::test_paired_seeds` |
| F12  | `results/a2c_training.png` emitted                                             | `notebooks/analysis.ipynb` cell 7 |
| F13  | `results/comparison.png` shows mean ± 1σ over ≥ 5 seeds                         | `tests/test_compare.py::test_comparison_shape` |
| F14  | Notebook renders end-to-end and contains ≥ 4 LaTeX equation blocks             | `tests/test_notebook.py::test_notebook_runs` |
| F15  | `recommend(s)` returns dataclass with `action, probs[7], next_state, reward`   | `tests/test_sdk.py::test_recommend_contract` |
| F16  | No public callable outside `TrainingSDK` is imported by `notebooks/` or `cli/` | `tests/test_architecture.py::test_sdk_is_only_entry` |
| F17  | All six CLI verbs invokable; each exits 0 on the happy path                    | `tests/test_cli.py::test_all_verbs` |
| Mask | Masked-logit positions → 0 probability after softmax                           | `tests/test_action_mask.py::test_masked_softmax` |

### 7.2 Project-level DoD

- Clean checkout: `uv sync --dev && uv run pytest` is green, ≥ 85 % coverage.
- `uv run ruff check src/ tests/ main.py scripts/` → zero violations.
- `uv run python scripts/generate_results.py` reproduces all four §7.7 plots.
- Every `F#` row in §7.1 has an evidence pointer.
- The trace matrix (`docs/TRACE.md`) has 53 rows, 46 MUSTs, all linked.
- All numeric claims in `docs/ANALYSIS.md` cite **seed**, **episode count**,
  and **mean ± std** — no bare adjectives like "converges fast" or
  "outperforms".

---

## 8. Out of scope

- **Live biometrics.** No heart-rate, HRV, sleep, or wearables. The simulator
  is the only source of truth for trainee state.
- **Multi-trainee population models.** Single seeded trainee.
- **Online learning.** No real-time on-policy data collection from a human.
- **Real-world efficacy claim.** This is a pedagogical artefact (see §11).
- **Mobile / web UI.** A4 candidate. The CLI + notebook is the A3 surface.
- **Continuous action space.** 7 discrete actions only.

---

## 9. Risks & mitigations

| #  | Risk                                                                 | Mitigation                                                                                                |
|----|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| R1 | LSTM overfits the 28-day window → policies learn its hallucinations  | Hold last 7 days as validation; abort training if val loss diverges 3 consecutive epochs.                  |
| R2 | REINFORCE high gradient variance fails to learn in budget            | Learned-baseline variant + reward normalisation; budget ≥ 500 episodes.                                    |
| R3 | A2C critic collapses to a constant                                    | Entropy bonus on actor, gradient clipping at norm 0.5, separate LR for critic.                             |
| R4 | Reward farming via volume overshoot                                   | `clip(progress, 0, 1.2)` + superlinear overload exponent 1.5 + λ_1 = 2.0 weight on overload (§1.7).        |
| R5 | Notebook hides logic from CI                                          | N7 architectural test rejects `nn.Module` / `loss` definitions inside `notebooks/`.                        |
| R6 | Hidden non-determinism (torch CUDA, Python set ordering)              | CPU-only training; seed manifest covers numpy / torch / random; CI runs the seed-reproducibility test.     |
| R7 | LSTM memorises periodisation rather than learns dynamics              | Held-out validation + variance-of-prediction check; explicitly named as a limitation (§11c).               |
| R8 | 150-LOC ceiling pressure pushes logic into utility dumping grounds    | Per-file responsibility documented in `PLAN.md`; ruff `mccabe` complexity guard.                            |
| R9 | Reward function is hand-designed, not learned                         | Cited as a limitation (§11d); ablation table compares λ_1 ∈ {1, 2, 4} so the choice is visible.            |

---

## 10. References

### Brief
- Assignment 3 brief §7 (state/action/reward, LSTM, REINFORCE, A2C, comparison,
  recommendation). Equations 1, 2, 4, 7, 8, 9, 10, 11, 12, 15, 16, 17 are
  transcribed verbatim in LaTeX in `docs/THEORY.md`.

### Lecture timestamps
- Lecture on policy gradients — REINFORCE derivation (Williams 1992).
- Lecture on actor-critic — advantage estimator and entropy regularisation.
- Lecture on world models — teacher-forcing vs rollout.

### Papers
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for
  connectionist reinforcement learning.* Machine Learning 8(3–4).
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000).
  *Policy Gradient Methods for Reinforcement Learning with Function
  Approximation.* NeurIPS.
- Mnih, V. et al. (2016). *Asynchronous methods for deep reinforcement
  learning.* ICML.
- Ha, D. & Schmidhuber, J. (2018). *World Models.* NeurIPS / arXiv:1803.10122.
- Huang, S. & Ontañón, S. (2022). *A closer look at invalid action masking in
  policy gradient algorithms.* FLAIRS.

### Project documents
- `CLAUDE.md` — global coding standards + §1.4 architect/implementer contract.
- `PLAN.md` — architecture, ADRs, module-level design.
- `shared/PROMPTS.md` — the literal prompts and AI-workflow narrative.
- `TODO.md` — phased task list with definition-of-done.
- `docs/THEORY.md` — verbatim LaTeX equations 1, 2, 4, 7, 8, 9, 10, 11, 12,
  15, 16, 17 from the brief.
- `docs/ANALYSIS.md` — §7.7 analysis (seed-cited, mean ± std).
- `docs/adr/ADR-004-action-masking.md` — masked-softmax design.
- `docs/TRACE.md` — 53-row trace matrix (46 MUSTs) from brief → tests.

---

## 11. Honest Limitations

The submission ships as a pedagogical artefact. The following caveats are
called out head-on (rather than buried) so the analysis in §7.7 is
interpreted in the right frame:

**(a) Plan-content data ≠ real workout outcomes** (brief §7.6 Q4, answered
head-on). The Kaggle dataset captures *prescribed* programs (what a coach
wrote), not *executed* sessions with measured performance. The LSTM learns
the rhythm of how coaches periodise — not how trainees actually respond.
Any claim about "trainee dynamics" is therefore a claim about prescribed
plan dynamics; the simulator's noise model is the only stand-in for
real-world variability.

**(b) Single synthetic trainee = no population generalisation.** One
seeded trainee instance is trained against. Different demographics,
training ages, and recovery profiles are out of scope. Results do not
generalise across trainees, and we make no claim that they do.

**(c) LSTM fit on plan-derived sequences may memorise periodisation
rather than learn dynamics.** Because the source data is prescription
(weekly templates that repeat), the LSTM can score well on validation by
predicting "the next day in the template" instead of capturing causal
state transitions. R7 in §9 names the held-out variance check that surfaces
this; the notebook reports the result rather than hiding it.

**(d) The reward function is hand-designed and may not align with
real-world training goals.** `gain_t = 0.7·progress + 0.3·variety`,
`λ_1 = 2.0`, `λ_2 = 1.0`, `overload_exponent = 1.5`, `seconds_per_rep = 3`
are all author choices grounded in the brief's prescribed weekly stimulus
— not in physiological measurement. An ablation table over `λ_1 ∈ {1, 2, 4}`
keeps the choice visible and falsifiable.

**(e) No biometric ground truth.** Soreness, fatigue, and readiness are
modelled state variables, not measurements. A real deployment would need
HRV, sleep, and self-report integration before any recommendation could
be trusted.

**(f) Reproducibility has known gaps.** CPU determinism is enforced; CUDA
operations (`scatter_add`, `index_add`), mixed-precision training,
multi-worker DataLoader shuffle order, and MPS LSTM kernels can still
introduce run-to-run drift. These are named in the README's
"Reproducibility caveats" subsection so a grader knows where the floor is.

Numeric claims about convergence and the REINFORCE-vs-A2C comparison in
`docs/ANALYSIS.md` are always reported as **mean ± std over N seeds**
together with the exact episode count — no bare adjectives.
