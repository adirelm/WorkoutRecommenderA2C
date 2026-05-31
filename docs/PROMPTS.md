# PROMPTS.md — Architect ↔ Implementer Trail

Per CLAUDE.md §1.4, the architect (solo developer in this role) decides
scope/architecture/acceptance criteria; the AI (Claude) implements
against approved specs. This file is the auditable trail of that
boundary: every substantive decision, every prompt the architect
issued, every implementation pass — and the human review notes that
follow.

## §1. How to read this file

This document is the **evidence trail** for the Human↔AI Responsibility
Contract defined in `CLAUDE.md §1.4`. It is not a tutorial and it is
not a marketing artefact — it is the audit log a grader (or future-me)
can use to verify that:

1. Every column marked **Human-decided** in the §1.4 contract table
   was, in fact, decided by the architect *before* the AI generated
   the code/test/doc that depends on it.
2. Every column marked **AI-delegated** is traceable back to a written
   spec the architect approved (PRD / PLAN / ADR / TRACE row).
3. The AI did not silently widen scope, weaken a quality gate, or
   change a test's acceptance criteria without an explicit
   architect-signed approval recorded here.

**Conventions used below.**

- *Architect intent* — the goal the human held in mind *before*
  prompting (the "why"). This is not the prompt verbatim; it is the
  decision being made.
- *AI workflow used* — the named multi-agent workflow ID (recorded in
  §5) the prompt fanned out to. Token counts and agent counts are
  cited so cost is auditable.
- *Human review* — what the architect accepted, rejected, or
  amended after seeing the AI output. **Acceptance is never
  implicit.** If a row has no review note, the artefact is not yet
  approved.
- *"Approved"* means: the architect read the diff (or the agent
  summary, when the diff is too large), confirmed it matches the
  intent in the PRD/PLAN row it implements, and committed it under
  their own authorship — not the AI's.

Compliance with §1.4 is provable iff every commit on `main` is either
(a) referenced from a Pass entry below, or (b) a trivial
chore/lint/format pass which CLAUDE.md §1.4 explicitly delegates to
the AI without per-commit review. Anything else is a contract
violation and must be flagged in the per-assignment
`final_review_progress.md`.

## §2. Phase 0 — Planning & Scaffolding

### Pass 1 — Brief ingestion + lecture transcription

- **Architect intent**: read the full 33-page Assignment 3 brief end
  to end, transcribe the 1h47m Hebrew lecture, and capture every §7
  requirement (state, action, reward, A2C-specific deliverables) so
  the planning pass downstream has a complete source-of-truth corpus.
- **AI workflows used**: `mlx-whisper` with the Hebrew `large-v3`
  model. The initial run hallucination-looped on the token
  `קודם כל` ("first of all") starting around the 25-minute mark;
  the transcript stabilised into the same phrase repeating for
  ~80 minutes. Restarted with `condition_on_previous_text=False`
  plus a domain-specific initial prompt mentioning RL terminology
  (state / action / reward / policy / value / advantage / A2C) →
  1285 clean segments, no loops, full coverage.
- **Human review**: rejected the looped first pass; confirmed the
  salvageable first-25-minute portion of that transcript was *also*
  discarded (in favour of a clean re-run end-to-end), because
  splicing two transcripts together would have invalidated the
  segment-level timestamps the trace matrix downstream needs to cite.

### Pass 2 — Comprehensive planning workflow

- **Architect intent**: convert the corpus from Pass 1 into a full
  planning stack — requirements → architecture options →
  state/action/reward design → PRD/PLAN/TODO drafts → strict-grader
  critique — *before* a single line of source code is written. This
  is the §1.4-mandated "architect decides architecture, AI
  implements against approved spec" step.
- **AI workflow**: `a3-comprehensive-planning` (17 sub-agents,
  658k tokens, ~38m wall clock).
- **Outputs**: 5 brief-ingest summaries (one per major brief
  section); 53-row TRACE matrix mapping brief requirements →
  planned artefacts; 3 architecture candidates (A = monolithic
  runtime, B = library + thin CLI, C = notebook-first); 3 state
  representations (minimal / moderate / rich); 3 reward
  weightings; PRD + PLAN + TODO drafts; an 8-lens strict-grader
  critique that returned **YELLOW** with **5 must-fix items**.
- **Human decisions captured** (these are the non-delegable §1.4
  rows the architect signed off here):
  - Action space = **7** (Rest / Push / Pull / Legs / FullBody /
    Conditioning / Mobility). The workflow's recommendation
    *replaced* the architect's original 6-action set (which had
    HIIT instead of Conditioning + Mobility split). Accepted
    because the lecture explicitly framed mobility as a separate
    recovery modality, not a subset of conditioning.
  - All **5 must-fix** items applied before any code lands.
  - **Self-grade 93** only on the moodle cover sheet — internal
    docs target a higher bar but the claim of 93 is the only
    grade-shaped number that appears anywhere submitted.
  - Repo name: **WorkoutRecommenderA2C** (rejected the
    workflow-suggested `a3-rl-workout` as too cryptic).
  - Lecturer (rmisegal) invited read-only.

### Pass 3 — Must-fix application + planning artefact write

- **Architect intent**: apply the 5 must-fix items from Pass 2 to the
  PRD/PLAN/TODO drafts and write the full planning stack to disk —
  i.e. move from "ideas in the workflow" to "artefacts in the repo".
- **AI workflow**: `a3-finalize-planning-docs` (6 parallel agents,
  335k tokens).
- **Outputs**: **15 files** written to disk —
  PRD (530 L) · PLAN (291 L) · TODO (123 L) · THEORY (310 L) ·
  TRACE (138 L initial, 53 rows) · 4 ADRs (action-space,
  architecture, state-rep, reward-weights) · `src/seeding.py` ·
  `tests/test_reproducibility.py` · README skeleton.

### Pass 4 — Repo publication

- **Architect intent**: `git init` the A3 project, push to a public
  GitHub repo, and invite the lecturer with read-only access — the
  earliest possible audit anchor.
- **AI workflow**: `a3-init-repo-and-publish` (5 sequential phases:
  git init → first commit → repo create → push → invite).
- **Outputs**: commit `08e78ab` on `main`; repo
  https://github.com/adirelm/workoutRecommenderA2C ; rmisegal
  invited under invitation **#320698055** (read-only, pending
  acceptance at time of writing).

### Pass 5 — Phase 0 scaffolding

- **Architect intent**: stand up the standard quality scaffolding
  *before* feature code begins — `pyproject.toml`, ruff, pytest +
  coverage, pre-commit, GitHub Actions CI, the 150-LOC-per-file
  guard, `config/config.yaml` skeleton, and a `main.py` stub.
- **AI workflow**: `a3-phase0-scaffolding` (6 parallel agents +
  verify + commit, 154k tokens). One scripting bug surfaced: the
  verify agent was missing its output-schema declaration, which
  caused a *false-positive* abort signal; recovered by issuing the
  commit directly. Subsequent CI run then failed because
  `uv sync --dev` was reading `[project.optional-dependencies]`
  instead of the PEP-735 `[dependency-groups]` table that newer
  uv expects. Fixed in commit `3130417`. CI is now green.
- **Human review**: accepted the scaffold *and* the post-commit fix.
  The false-positive abort was logged as a workflow-quality bug to
  fix in the harness, not the assignment.

### Pass 6 — Phase 0 validation

- **Architect intent**: before declaring Phase 0 done, run an
  independent set of validators that don't trust the planning
  workflow's own self-grade — five parallel passes covering theory
  accuracy, trace completeness, submission-form compliance, gate
  enforcement, and a strict-grader re-pass.
- **Outputs / verdicts**:
  - **v1 — theory-accuracy**: 🟢 GREEN (12/12 checks)
  - **v2 — trace-completeness**: 🔴 RED (TRACE evidence-light,
    18 gaps — rows pointed at planned files that don't exist yet
    and didn't cite the lecture timestamp they came from)
  - **v3 — submission-compliance**: 🟡 YELLOW (1 expected
    blocker: the PDF deliverable, deferred to Phase 12)
  - **v4 — gate-enforcement**: 🟡 MIXED (11 paths where a quality
    gate could be weakened without tripping CI — all 11 logged as
    P1 fixes for this Pass 7)
  - **v5 — grader-repass**: 🟡 YELLOW (3 lenses improved over
    Pass 2's critique, 1 regressed)
- A bonus 6th workflow (`deep-quality`) misfired by importing
  acceptance criteria from Assignment 2 and flagging A3 for
  violations that don't apply; output discarded.

### Pass 7 — Phase 0 fix execution (this commit)

- **Architect intent**: address the **4 P0 + 3 P1** findings from
  Pass 6 before opening Phase 1, so Phase 1 starts from a clean
  validation baseline rather than dragging Phase-0 debt forward.
- **AI workflow**: `a3-phase0-fix-execution` (7 parallel agents).
- **Outputs**: TRACE matrix extended by **+18 rows** (now 71
  total, every row evidenced by a lecture timestamp or a brief
  §-number); **2 §1.4 enforcement tests** added to the test
  suite; this `PROMPTS.md` written; `THEORY.md §3.2` prose
  correction (advantage-vs-TD-error framing was ambiguous in the
  draft); secret-scan pattern set extended to catch the moodle
  group-code shape; `LICENSE` added.

## §3. Phase 1 — Data ingest

### Pass 8 — Phase 1: Data ingest (commits `238ba9a`, `2476893`, `f7ba06e`)

- **Architect intent**: stand up the data layer per PRD §1.5.0 — a
  `KaggleClient` (download + cache), a `Preprocessor` (schema
  validation + outlier clip + NaN policy), a `DailyAggregator`
  (per-user-per-day collapse), and a `ProgramFilter` that selects
  **PHUL** as the primary recurring program. TDD throughout: tests
  precede implementation, and the data-quality contract from PRD
  §1.5.0 is encoded as assertions, not as prose.
- **AI workflow**: `a3-phase1-data-ingest` (`wk6wqdd4o`).
- **Outputs**: 32 unit tests (all green); PHUL canonicalised as the
  primary program; data-quality contract enforced on every load.
- **Human review**: accepted. The PHUL choice was the architect's
  call (PRD §1.5.0); the AI was not permitted to substitute a
  different program at runtime even when row counts favoured one.

### Pass 9 — Phase 1 validation + fix (commit `2476893` + gate-fix `f7ba06e`)

- **Architect intent**: 5-workflow validation pass over the Phase 1
  deliverables (v1p1 through v5p1: schema, coverage, contract,
  determinism, grader re-pass). Treat the gate as binding even when
  the verdict is YELLOW rather than RED.
- **Outcome**: **YELLOW** verdict — three findings worth fixing
  before declaring Phase 1 done.
- **Key fixes applied**:
  - `kaggle_client` error handling: a `KaggleApiError` was being
    silently caught and re-raised as a generic `RuntimeError`,
    erasing the original cause. Now the chain is preserved.
  - Canonical-sort key order in `DailyAggregator`: sort was by
    `(date, user)` not `(user, date)`, which made downstream
    train/val splits non-reproducible across pandas versions.
  - Branch coverage on the `Preprocessor` NaN-policy switch — one
    branch (`policy = "drop"` with all-NaN column) was unreached.

## §4. Phases 2-7 — Env, World Model, RL, SDK, Notebook

### Pass 10 — Phase 2: Env layer (commits `eeafb0d`, `4ee28f2`)

- **Architect intent**: build the MDP layer — `WorkoutEnv` (Gym-ish
  API), `Reward` (the λ₁/λ₂ scalar from the decision log), an
  `ActionMask` (legality of consecutive same-action picks, per
  THEORY §3.4), the 12-dim `State` per ADR-003, and a
  `SyntheticTrainee` stand-in for the human-in-the-loop signal
  during world-model pretraining.
- **AI workflow**: built under the standard Phase-2 implementation
  pattern (test-first, agent per module).
- **Validation**: 5-workflow validation → **RED** because
  `reward.py` coverage was at 83% (the imbalance-only edge case
  where λ₁ = 0 was untested). Fix landed → coverage **100%**.
- **Human review**: accepted after the coverage fix. The RED→GREEN
  transition is the audit trail.

### Pass 11 — Phase 3: LSTM world model (commits `ce9747f`, `03fb41d`, `8a8ce5c`, `a646ba5`, `e7a8927`, `ba8f23a`)

- **Architect intent**: pretrain an LSTM world model on
  `SyntheticTrainee` trajectories so the RL agent in Phases 4-5 can
  bootstrap on a learned dynamics model rather than only on live
  rollouts. The world model is **frozen** during RL training (PRD
  §1.5.3) — this is non-negotiable; the AI must not unfreeze it to
  "improve" training-loss numbers.
- **Validation**: 5-workflow validation → **YELLOW**. Two
  follow-on items applied: THEORY **§7.3** added (world-model
  freezing rationale + bias-variance argument), and **ADR-005**
  authored to record terminal-condition choices (episode caps +
  fatigue overflow + invalid-mask exhaustion).
- **Human review**: accepted with the THEORY/ADR additions. The
  6-commit chain reflects iterative test/fix passes, not a clean
  single landing — recorded honestly here rather than collapsed.

### Pass 12 — Phase 4: REINFORCE baseline (commits `96eec6a`, `db0b8c6`, `3496179`)

- **Architect intent**: REINFORCE first — *before* A2C — so the
  comparison in Phase 5 is against a real baseline rather than a
  hand-wavy "untrained policy". `PolicyNet` is a deliberately small
  1-layer FC (hidden = 128) so the actor-vs-critic comparison in
  Phase 5 is about the algorithm, not network capacity.
  `REINFORCETrainer` uses a `RunningMeanBaseline` for variance
  reduction (THEORY §4.2).
- **Validation**: 5-workflow validation → **YELLOW**. Two
  fixes applied: strict gradient-sign tests (the trainer's
  `loss.backward()` was correct, but the test only checked
  *magnitude*, not *sign* — strengthened); CE-equivalence
  docstring on `PolicyNet.log_prob_action` explaining why the
  log-prob computation is mathematically equivalent to
  cross-entropy on the chosen action (THEORY §4.1 cross-reference).
- **Human review**: accepted post-fixes.

### Pass 13 — Phase 5: A2C (commits `9c191e0`, `f1fc34b`, `7088406`)

- **Architect intent**: the headline algorithm. `ActorCriticNet`
  (shared trunk, two heads), `A2CTrainer` (advantage from
  TD-target, separate optimizers for actor and critic per ADR), and
  a `comparator` module that runs REINFORCE vs A2C on identical
  seeds for the Phase-7 notebook.
- **Recovery event**: commit `9c191e0` was a **partial commit** —
  8 of 11 intended files were missing from the staged set (the
  agent that ran `git add` filtered by a pattern that didn't match
  three of the new modules). Detected by the post-commit gate
  sweep. Fixed via a dedicated recovery workflow that re-added the
  missing files in `f1fc34b`.
- **Validation**: consolidated 1-workflow validation (5 axes run
  in parallel: API, math, determinism, coverage, grader) →
  **HEDGE** verdict.
- **Fixes applied** (these are architect-signed, recorded in the
  decision log below):
  - **Seed-reuse bug**: `A2CTrainer.run_episode` was re-seeding
    the env on every episode, collapsing the rollout distribution.
    Now the env is seeded **once in `__init__`**.
  - **Grad-clip scope**: clipping was applied to all parameters
    in one shot, which fought the two-optimizer setup. Now clip
    is per-step + per-net (actor params clipped inside the actor
    step, critic params inside the critic step).
  - **Entropy surrogate**: bonus was computed as
    `-log_prob.mean()` (a sampling surrogate) instead of the
    closed-form `dist.entropy()`. Replaced with the closed form;
    the surrogate is biased for finite batches.

### Pass 14 — Phase 6: SDK + CLI (commits `07635cf`, `d4b38bc`)

- **Architect intent**: stand up the public surface — `WorkoutSDK`
  (the only entry point business logic should be reachable
  through, per CLAUDE.md §3) + a `CLIMenu` + a `main.py`
  entrypoint. The constraint that **no business logic may live in
  the CLI or in the notebook** is architect-signed; the AI is not
  allowed to inline an SDK call's body into a menu handler to
  "shave a layer".
- **Validation**: consolidated validation → **FAIL** with 3 hard
  failures and 3 P0s:
  - **CLI verb 6 `TypeError`** — `recommend()` was being called
    without the optional `state` argument, but the SDK signature
    made it required. Fixed: defaults to `State.initial()` when
    no state is provided.
  - **Tuple-unpacking bug** in the train-A2C handler — the SDK
    returns a 3-tuple but the menu unpacked it as a 2-tuple,
    silently dropping the metrics dict. Fixed.
  - **Private API leak** — the menu was importing
    `_internal_helper` from `sdk.py`. Promoted to public or
    inlined into the menu, depending on which side owns the
    behaviour.
- All 6 P0 findings fixed before declaring Phase 6 done.

### Pass 15 — Phase 7: Analysis notebook (commit `2e441e2`)

- **Architect intent**: write `notebooks/analysis.ipynb` per PRD
  §4 N7 — the notebook is a **consumer of `WorkoutSDK` only**, no
  business logic redefined inside it. Contents: LaTeX derivations
  cross-referenced to THEORY §3-§5, four plots
  (reward-curve / advantage-distribution / policy-entropy /
  REINFORCE-vs-A2C variance), and a 5-question discussion section
  per THEORY §7.6.
- **Human review**: accepted. Specifically verified the
  Action-Masking citation (lecture timestamp 01:14:33) is in the
  notebook's reference list, not just in THEORY — the brief
  requires the citation to be reachable from the analysis artefact.

### Pass 16 — Phase 8: Submission prep (this commit)

- **Architect intent**: pre-submission sweep — README expanded
  with the run/quickstart story, `COST_ANALYSIS.md` authored to
  document the workflow-token cost per phase, this `PROMPTS.md`
  extended to cover Phases 1-7, and a final pre-submission gate
  sweep (lint + coverage + 150-LOC guard + secret scan + TRACE
  freshness). Tagging **v1.0.0** at the end of this commit.
- **Human review**: this entry itself is the architect's review
  note — accepting the v1.0.0 surface as the submission cut.

## §4. Decision log

A flat chronological list of architect-approved decisions. Every
row here is a §1.4 *Human-decided* column entry — they should never
be implicitly changed by an AI pass.

- 2026-05-30 — Self-grade target: **93** (locked, not 100 — per
  §1.4 + standing-rule §5 "pedantry on 100").
- 2026-05-30 — Group code: **adrl-001** (8 chars, semester-long,
  per standing rule R3).
- 2026-05-30 — Deadline binding: **2026-06-10** (per lecturer
  text, not the moodle form's 2026-06-03 — the human-authoritative
  deadline wins when the platform disagrees with the human).
- 2026-05-31 — Action space: **7** (Rest / Push / Pull / Legs /
  FullBody / Conditioning / Mobility — workflow recommendation
  accepted over original 6 with HIIT).
- 2026-05-31 — **Hybrid architecture**: Architecture A runtime
  (monolithic) + ONE analysis notebook (best of A and C; B
  rejected as over-engineered for a single-developer assignment).
- 2026-05-31 — State representation: **C2_moderate_12d** (12-dim,
  moderate richness — minimal would underfit, rich would inflate
  episodes-to-converge past the brief's compute envelope).
- 2026-05-31 — Reward weights: **λ_1 = 2.0, λ_2 = 1.0** (overload
  penalised more heavily than imbalance; imbalance still active
  but secondary).
- 2026-05-31 — Repo: **adirelm/WorkoutRecommenderA2C** public.
- 2026-05-31 — rmisegal invited read-only (invitation
  **#320698055**).
- 2026-05-31 — `A2CTrainer` must seed the env **once in
  `__init__`**, not on every episode (P0 from v1p5 — re-seeding
  per episode collapses the rollout distribution).
- 2026-05-31 — Grad-clip scope is **per-step + per-net**: actor
  parameters clipped inside the actor optimizer step, critic
  parameters inside the critic step. Clipping the union in one
  call fights the two-optimizer design.
- 2026-05-31 — Entropy bonus uses the closed-form
  `dist.entropy()`, **not** the `-log_prob.mean()` sampling
  surrogate (the surrogate is biased for finite batches).
- 2026-05-31 — CLI verb 6 `recommend()` defaults to
  `State.initial()` when no state argument is provided — fixes
  the `TypeError` from Pass 14 without weakening the SDK
  signature for programmatic callers.
- 2026-05-31 — Notebook is **consumer-only**: no `WorkoutSDK`
  methods may be redefined or shadowed inside `analysis.ipynb`
  (per PRD §4 N7). Any "convenience helper" the AI wants to
  inline into a cell must be promoted to the SDK first.

## §5. Workflow registry

Every multi-agent workflow run is logged here so its cost is
auditable and so a future reviewer can re-open the exact run that
produced a given artefact.

| Workflow ID | Phase | Agents | Tokens | Outcome |
|---|---|---|---|---|
| `a3-comprehensive-planning` (wmyibh2pl) | 0 plan | 17 | 658k | YELLOW; 5 must-fix |
| `a3-finalize-planning-docs` (w058pq917) | 0 plan | 6 | 335k | 15 files written |
| `a3-init-repo-and-publish` (waftqx6d9) | 0 publish | 5 | 112k | repo live, invite sent |
| `a3-phase0-scaffolding` (wjzk25zze) | 0 build | 7 | 154k | green (after fix commit) |
| `v1-theory-accuracy` (wb7pvn6wj→w301ws66p) | 0 validate | 14 | 449k | 🟢 GREEN (12/12) |
| `v2-trace-completeness` (wnzyk1ygj→w0aceheae) | 0 validate | 6 | 268k | 🔴 RED (18 gaps) |
| `v3-submission-compliance` (w2v028r83) | 0 validate | 6 | 138k | 🟡 YELLOW (1 expected blocker) |
| `v4-gate-enforcement` (wu3c1jaer) | 0 validate | 6 | 136k | 🟡 MIXED (11 upgrades) |
| `v5-grader-repass` (wso3q6f8c) | 0 validate | 9 | 230k | 🟡 YELLOW (3 lenses ↑, 1 ↓) |
| `a3-phase0-fix-execution` (wso3q6f8c→) | 0 polish | 9 | ~210k | landed (commit on `main`) |
| `a3-phase1-data-ingest` (wk6wqdd4o) | 1 build | 8 | ~290k | green; 32 unit tests |
| `v1p1-schema` / `v2p1-coverage` / `v3p1-contract` / `v4p1-determinism` / `v5p1-grader` | 1 validate | 5×6 | ~310k | 🟡 YELLOW; 3 fixes applied |
| `a3-phase2-env` | 2 build | 7 | ~245k | green post-coverage-fix |
| `v1p2…v5p2` (env validation) | 2 validate | 5×5 | ~260k | 🔴 RED → 🟢 GREEN after reward.py fix |
| `a3-phase3-world-model` | 3 build | 8 | ~360k | YELLOW; THEORY §7.3 + ADR-005 added |
| `v1p3…v5p3` (world-model validation) | 3 validate | 5×5 | ~270k | 🟡 YELLOW; follow-ons landed |
| `a3-phase4-reinforce` | 4 build | 6 | ~205k | YELLOW; gradient-sign tests + CE docstring |
| `v1p4…v5p4` (REINFORCE validation) | 4 validate | 5×4 | ~230k | 🟡 YELLOW; fixes applied |
| `a3-phase5-a2c` | 5 build | 8 | ~340k | partial commit recovered in `f1fc34b` |
| `a3-phase5-a2c-recovery` | 5 recover | 3 | ~85k | re-added 3 missing files |
| `v1p5-consolidated` (5 axes parallel) | 5 validate | 9 | ~310k | HEDGE; 3 P0 fixes (seed/grad-clip/entropy) |
| `a3-phase6-sdk-cli` | 6 build | 6 | ~195k | FAIL; 6 P0s fixed before sign-off |
| `v1p6-consolidated` | 6 validate | 9 | ~250k | FAIL → GREEN after P0 sweep |
| `a3-phase7-notebook` | 7 build | 4 | ~175k | green; notebook is SDK-consumer only |
| `a3-phase8-submission-prep` (THIS) | 8 polish | 5 | (in flight) | (this commit) |
