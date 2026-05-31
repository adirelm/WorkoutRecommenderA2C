# ADR-001 — Hybrid Architecture: Deeply-Layered SDK + One Consumer Notebook

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-002 (state representation), ADR-003 (reward weighting), ADR-004 (action masking)

## Context

Assignment 3 (Workout Recommender, A2C) is graded by a rubric that explicitly
weighs *principle understanding* above absolute training quality. The lecturer
asks for an explicit REINFORCE-vs-A2C comparison, a justified state / action
design, a frozen LSTM world-model stage followed by RL on top, and a written
discussion that answers the §7.6 questions. A1 was scored on a similar rubric
and was praised for engineering hygiene (planning, docs, config / security,
testing, UI / UX, version management) but penalised for shallow research /
analysis depth, missing cost awareness, weak extensibility narrative, and the
absence of automated quality gates — plus a tone penalty for over-confident
self-assessment.

Three candidate architectures were considered:

- **A — Deeply-layered SDK.** `src/{data,env,model,services,sdk,cli,gui}/` with
  `TrainingSDK` as the single public entry point, tests mirroring the tree, all
  CLAUDE.md hard constraints (150-LOC ceiling, ≥85% coverage, OOP base /
  subclass with no duplication, externalised config, zero Ruff, `uv`-only)
  enforced in CI. This is the structure A2 used and the structure the
  CLAUDE.md §3 SDK rule explicitly mandates.
- **B — Monolithic.** A single `sdk.py` that owns data, env, models, training,
  and analysis. Fastest to write, fewest files, but collapses the layer
  boundaries that make the codebase legible to a grader and creates a god
  module that the 150-LOC rule then forces to be split anyway.
- **C — Notebook-first.** One Jupyter notebook per phase (data, world model,
  REINFORCE, A2C, analysis). Easiest to read linearly, but breaks the OOP,
  coverage, Ruff, and 150-LOC gates, and re-introduces the over-confident-narrative
  failure mode the prior assignment was penalised for.

## Decision

We adopt a **hybrid**: Architecture A verbatim for all runtime code, plus
exactly **one** consumer notebook at `notebooks/analysis.ipynb` that imports
the SDK and renders the §7.7 deliverables (LSTM loss curves, REINFORCE reward
graph, A2C training graph, head-to-head comparison, and the §7.6 discussion).

Concrete shape:

- `src/data/`, `src/env/`, `src/world_model/`, `src/agents/`, `src/training/`,
  `src/services/`, `src/sdk/`, `src/cli/`, `src/gui/`, `src/utils/`. Each
  module ≤ 150 lines of code per CLAUDE.md §1.
- `WorkoutRecommenderSDK` is the single public entry point. Every UI surface
  (terminal CLI, Tkinter GUI, the analysis notebook) calls only the SDK and
  never reaches into the engine layers.
- `tests/` mirrors `src/` one-for-one; coverage gate at 85 % is enforced by
  CI, and `.coveragerc` omits the notebook so the gate stays honest.
- The notebook is a *consumer* of the SDK, not a parallel implementation. It
  calls `sdk.load_data()` → `sdk.pretrain_world_model()` →
  `sdk.train_reinforce()` → `sdk.train_a2c()` → `sdk.compare()` and renders
  the figures the SDK saves to `results/comparison/`. The SDK writes the
  PNGs; the notebook displays the same PNGs (single source of truth). LaTeX
  cells derive the REINFORCE gradient `∇J(θ) = E[∇log π(a|s)·G_t]` and the
  A2C advantage `A(s, a) = r + γV(s′) − V(s)` next to the code that
  implements them.
- The action space is the 7-action discrete set
  `{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`
  fixed by ADR-002 / ADR-004 — every layer of the SDK is parameterised on this
  cardinality through `config.action.count = 7`.
- Branch: `assignment-3` off `main`. Version: 1.2.0. Repository:
  `WorkoutRecommenderA2C` (created at first push).

## Rationale

The hybrid keeps the engineering spine that the rubric expects (CLAUDE.md §3
SDK as single entry point; §1.4 architect / implementer contract with a clear
public API), and adds the narrative artefact that converts the work from a
runnable artefact into one that explains itself next to the equations.
Architecture A alone repeats the A2 pattern without adding the
pedagogical surface the §7.6 / §7.7 questions explicitly ask for. Architecture
C alone breaks four CLAUDE.md hard constraints simultaneously (CL1, CL2, CL3,
CL5) and would re-create the tone problem A1 was penalised for, because a
notebook-first layout invites a polished-final-only submission with no
test-evidence trail. Architecture B alone collapses the layer boundaries that
made A2 legible and would produce a god module the 150-LOC rule then
fragments without a coherent layering story.

By keeping the notebook a *strict consumer* of the SDK, we get the linear
reading path (README → PRD → PLAN → notebook → `src/sdk/` → engine layers
beneath) without duplicating logic and without weakening any test gate. The
notebook is regenerated from a papermill template in CI so its diffs stay
clean and the reproducibility story (one `uv run main.py` plus one notebook
re-execute) is enforced by the same pipeline that runs the tests.

## Consequences

**Positive.**

- All seven CLAUDE.md hard constraints (CL1–CL7) remain satisfiable; the
  notebook is excluded from coverage but contains no business logic.
- The SDK facade gives a grader a single drill-down path and gives a future
  contributor a single extension point (add a new on-policy algorithm →
  subclass `src.services.base_trainer.BaseTrainer`, register it in
  `WorkoutSDK._TRAINER_REGISTRY`, and the generic `sdk.train(algo, ...)`
  dispatcher routes to it with zero edits to the facade body). Worked
  example: `class PPOTrainer(BaseTrainer): ...` + `_TRAINER_REGISTRY["ppo"] = PPOTrainer`
  is the *entire* diff needed to expose a PPO algorithm through the SDK.
- The §7.6 discussion lives next to the figures it discusses, so the
  understanding-vs-result tension the rubric flags is answered structurally
  rather than rhetorically.
- The terminal CLI and Tkinter GUI carry forward from A2 unchanged in shape;
  graders who prefer running over reading get `uv run main.py`, graders who
  prefer reading get the notebook, graders who want to audit get
  `src/sdk/`.

**Negative.**

- Iteration speed during exploration is slower than in a notebook-first
  layout. Hyper-parameter sweeps and reward-shaping experiments go through
  the SDK + config round-trip rather than being typed into a cell. We accept
  this cost because the rubric grades the final artefact, not the
  exploration velocity, and the SDK round-trip is what makes the work
  reproducible — itself a graded item.
- File count is higher than Architecture B (planned ≈ 30 source files versus
  B's ≈ 9). The SDK facade collapses that navigation cost for any reader who
  starts at `src/sdk/`.
- A papermill-driven notebook regeneration step adds CI complexity. Mitigated
  by reusing the script harness pattern from A2 (`scripts/generate_results.py`
  plus `scripts/check_file_sizes.py`).

## Follow-ups

- ADR-002 fixes the state representation that the SDK exposes through
  `sdk.get_state()`.
- ADR-003 fixes the reward weights the SDK uses inside `sdk.train_*()`.
- ADR-004 fixes the action-mask service the SDK injects into both agents.
