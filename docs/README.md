# A3 Documentation Index

This directory holds the architect-grade documentation for WorkoutRecommenderA2C (Bar-Ilan Vibe-Coding & RL Workshop, Assignment 3). Read in this order for the cleanest narrative:

## 1 — Frame the problem
- [PRD.md](PRD.md) — Product requirements: scope, MDP definition, state/action/reward, F1-F17 acceptance criteria.
- [STATE_DESIGN.md](STATE_DESIGN.md) — 12-channel C2_moderate_12d state vector rationale.
- [ACTION_DESIGN.md](ACTION_DESIGN.md) — 7-action discrete set + masking rules.

## 2 — Architect the solution
- [PLAN.md](PLAN.md) — Architecture, C4-Container mermaid diagram, 10-phase build sequence, ADR pointers.
- [adr/](adr/) — 5 Architecture Decision Records (hybrid SDK + analysis-notebook, state, reward, action masking, terminal conditions).

## 3 — Theory↔Code contract
- [THEORY.md](THEORY.md) — Brief's 12 equations transcribed verbatim in LaTeX, each cross-linked to the src/ file that implements it.
- [TRACE.md](TRACE.md) — 71-row traceability matrix (brief §, transcript, standing rules, CLAUDE.md constraints → deliverable + test).

## 4 — Build trail
- [TODO.md](TODO.md) — 53 phase-tagged tasks with definition-of-done per row.
- [PROMPTS.md](PROMPTS.md) — Architect↔implementer evidence trail (Phase 0 → Phase 8) — the §1.4 contract in audit form.

## 5 — Cost + experiments
- [COST_ANALYSIS.md](COST_ANALYSIS.md) — Token-spend table across all phases + per-deliverable amortisation.
- [EXPERIMENTS.md](EXPERIMENTS.md) — Hypothesis-setup-result-verdict log for §7.6 discussion + DA4 comparison.

## 6 — Submission
- [adrl-001-ex03.pdf](../adrl-001-ex03.pdf) — moodle cover sheet (gitignored — contains PII).
- [../README.md](../README.md) — top-level README with Quick Start.
- [../notebooks/analysis.ipynb](../notebooks/analysis.ipynb) — §7.7 chart deliverables + LaTeX + §7.6 discussion.
