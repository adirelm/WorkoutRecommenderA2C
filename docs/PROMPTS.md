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

## §3. Phase 1 — Data ingest (in progress)

Placeholder. To be filled when Phase 1 begins. The architect-decided
column for Phase 1 will include: dataset schema, train/val split
ratio, feature normalisation policy, and the acceptance test that
defines "ingest is done". The AI will be allowed to implement the
loader, the normaliser, and the tests against those criteria — not
to negotiate them.

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
| `a3-phase0-fix-execution` (THIS) | 0 polish | 9 | (in flight) | (this commit) |
