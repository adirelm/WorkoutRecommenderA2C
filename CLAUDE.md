# CLAUDE.md — Global Coding Standards for WorkoutRecommenderA2C

## Project Context

WorkoutRecommenderA2C is the Assignment-3 deliverable for the Bar-Ilan University
*Vibe Coding & Reinforcement Learning* workshop. The pipeline is a three-stage
attempt to reproduce brief §7 on a single synthetic trainee:

1. **LSTM transition model** — supervised next-state predictor over a Kaggle
   workout-program trajectory (acts as a world model in the Ha & Schmidhuber
   sense).
2. **REINFORCE policy** (Williams 1992) — Monte-Carlo policy gradient on top of
   the (frozen) LSTM rollout.
3. **A2C** (synchronous Advantage Actor-Critic) — actor + critic on the same env.

See `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, and `docs/THEORY.md` for the
binding specification.

## Human ↔ AI Responsibility Contract (§1.4)

§1.4 of the submission guidelines frames the developer as **architect** and the
AI as **implementer**. This file is the contract that boundary makes explicit.
Each row says **who decides** before any code is generated.

| Concern | Human-decided (non-delegable) | AI-delegated |
|---|---|---|
| Requirements (PRD, scope, success criteria, brief §7 alignment) | ✅ | — |
| Architecture (ADRs, layer boundaries, public API shape) | ✅ | — |
| State / action / reward design choices | ✅ | — |
| Test acceptance criteria + the assertions that must hold | ✅ | — |
| Final code-review sign-off + commit-message intent | ✅ | — |
| Self-score / grade claim against the rubric | ✅ | — |
| Code generation against an approved spec | — | ✅ |
| Refactoring within an existing public API | — | ✅ |
| Test scaffolding + boilerplate from a written spec | — | ✅ |
| Docstring / notebook-prose drafts (human edits before commit) | — | ✅ |
| Routine doc maintenance (link fixes, freshness sweeps) | — | ✅ |
| Lint / format auto-fixes | — | ✅ |

**Operating rule.** If any AI-generated change would alter a human-decided
column above (e.g. add a new public SDK method, change a test assertion, weaken
a quality gate, choose between two architectures), the human must sign off
explicitly *before* the code lands — typically by approving the PRD/PLAN edit
first, *then* letting the AI execute against it.

This contract is also evidenced in `docs/shared/PROMPTS.md` (the literal prompts used)
and in the per-section commit messages that name the § of the submission
guidelines being addressed.

## Hard Constraints (Apply to ALL Files)

### 1. File Size Limit — 150 Lines Maximum

Every Python file (`.py`) must not exceed 150 lines of code. If a file
approaches 150 lines, split into separate modules.

### 2. Test-Driven Development (TDD)

Write tests BEFORE implementation. RED → GREEN → REFACTOR. All new code must
achieve **≥ 85 %** test coverage (statement + branch).

Run: `uv run pytest tests/ --cov=src --cov-report=term-missing`

### 3. Object-Oriented Programming (OOP)

Use inheritance where it pays — `BaseAgent → REINFORCEAgent / A2CAgent`. No
code duplication; shared logic in base classes. **`WorkoutSDK` is the single
business-logic entry point** for all UIs (CLI, GUI, notebook). UIs never
import services / model / env directly.

### 4. No Hardcoded Values

ALL **algorithm-relevant** parameters, rewards, and thresholds live in
`config/config.yaml` and are accessed via the config loader. This covers: RL
hyperparameters (lr, γ, λ, entropy_coef, batch_size, episodes), reward weights
(λ₁, λ₂, w_p, w_v, overload_threshold, overload_exponent), state-design
constants, action-masking thresholds, seeds, and the colour palette.

Local UI-styling literals (button pixel dimensions, dashboard line offsets,
matplotlib `alpha` / `fontsize` / `dpi` values) stay in their rendering
modules. The test for "should this be in config" is: *"would I expect a
grader, contributor, or future-me to ever want to change this without
editing source?"* If yes → config. If no → keep it local.

### 5. No Code Duplication (DRY)

Extract common logic into shared methods/classes. If a pattern appears twice,
create a utility or base class method.

### 6. Linting — Zero Ruff Violations

Run: `uv run ruff check src/ tests/ main.py scripts/`. Must produce zero errors
before every commit.

### 7. Package Manager — UV Only

Use `uv` exclusively. No pip, no conda.
Run app: `uv run main.py`
Install deps: `uv sync --dev`

## Algorithm Requirements (brief §7)

### LSTM Transition Model (Phase 3, brief §7.3)

- Inputs: `(s_t, a_t)` with optional hidden `h_t`; output `ŝ_{t+1}`.
- Trained supervised on rolling windows over the synthetic trainee.
- Frozen during RL phases (`requires_grad=False` on every LSTM param).
- Loss curves emitted to `results/lstm_loss.png`.

### REINFORCE Agent (Phase 5, brief §7.4)

- Softmax policy over **7 discrete actions**
  `{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`.
- Update: `θ ← θ + α Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b)` (brief eq. 4/16).
- Reward-to-go `G_t = Σ_{t'≥t} γ^{t'−t} r_{t'}` (brief eq. 7).
- Running-mean baseline `b` (control variate, unbiased — see THEORY §3.3).
- Implemented as weighted cross-entropy:
  `(F.cross_entropy(logits, action, reduction='none') * G_t.detach()).sum()`
  (brief §2.6 / §2.7, lecture ~01:01:30).

### A2C Agent (Phase 6, brief §7.5)

- Shared trunk, actor head (softmax over 7 actions) + critic head (scalar V).
- Advantage: `A_t = δ_t = r_t + γ·V_ψ(s_{t+1}) − V_ψ(s_t)` (brief eq. 9/17).
- Actor update: `θ ← θ + α·∇_θ log π_θ(a_t|s_t) · δ_t` (brief eq. 10).
- Critic loss: `L(ψ) = ½·δ_t²` (brief eq. 12).
- Entropy bonus on actor for exploration.

### Reward (brief eq. 15, restored per must-fix)

`r_t = gain_t − λ_1·overload_penalty_t − λ_2·imbalance_penalty_t`

with `λ_1 = 2.0`, `λ_2 = 1.0`, both in `config.yaml`. Full derivation in
`docs/THEORY.md §6` and `docs/adr/ADR-003-reward-weighting.md`.

## Comparison Requirements (brief §7.6 / §7.7)

Generate one head-to-head comparison:
- REINFORCE vs A2C — paired seed schedule, **mean ± 1σ over ≥ 5 seeds** on a
  held-out start-state distribution. Save as `results/comparison.png`.
- Discuss in `notebooks/analysis.ipynb` next to the equation derivations.

## Honest Reporting Rules (must-fix #5 — A1 over-confidence penalty)

- No "achieves", "solves", "demonstrates", "earns the same ~XX" phrasings.
- Every numeric claim names: seed, episode count, mean ± std.
- README and PRD §11 carry an "Honest Limitations" section listing:
  (a) plan-content data ≠ real workout outcomes,
  (b) single synthetic trainee → no population generalisation,
  (c) LSTM may memorise periodisation rather than learn dynamics,
  (d) reward is hand-designed; alignment with real-world goals is not validated.

## Version Control

- New repository for A3 (NOT same as A1/A2). Branch: `main`.
- Initial version: `1.0.0` on the assignment-3 deliverable tag.
- Commit-message convention: `<phase>: <§-ref> <imperative summary>`.

## Config Structure

All config in `config/config.yaml`:
- `environment` — episode length, state dim, action count
- `rewards` — `lambda_1, lambda_2, w_progress, w_variety, overload_threshold, overload_exponent`
- `data_quality` — `seconds_per_rep` (the time-encoded-reps assumption from PRD §1.5.0)
- `lstm` — `hidden, num_layers, dropout, lr, epochs, window`
- `reinforce` — `lr, gamma, episodes, baseline_alpha`
- `a2c` — `actor_lr, critic_lr, gamma, entropy_coef, episodes`
- `seed` — single global seed (consumed by `src/utils/seeding.py::set_global_seed`)
- `action_masking` — thresholds for soreness-based masking
- `paths` — data, results, checkpoints
- `version` — semantic version string

## Pre-Submission Review Methodology

Before submission (or whenever asked "is this ready?"), walk the 9-phase
iterative pre-submission audit. The methodology and per-phase self-critique
prompts live under `instructions/review_methodology/` (gitignored, local
only — do NOT reference in shipped docs).

The methodology is reusable across assignments. Update its phase files only
when the *process* improves, not when the assignment content changes.
