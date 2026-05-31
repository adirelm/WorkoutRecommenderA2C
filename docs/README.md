# A3 Documentation Index

This directory holds the architect-grade documentation for WorkoutRecommenderA2C (Bar-Ilan Vibe-Coding & RL Workshop, Assignment 3). Read in this order for the cleanest narrative:

## 1 — Frame the problem
- [PRD.md](PRD.md) — Master product requirements: scope, MDP definition, state/action/reward, F1-F17 acceptance criteria.
- [prd/PRD-LSTM.md](prd/PRD-LSTM.md) — Per-algorithm PRD for the LSTM world model (V3 §2; F4–F6; derived from PRD.md §3.2).
- [prd/PRD-REINFORCE.md](prd/PRD-REINFORCE.md) — Per-algorithm PRD for REINFORCE (V3 §2; F7–F9; derived from PRD.md §3.3).
- [prd/PRD-A2C.md](prd/PRD-A2C.md) — Per-algorithm PRD for synchronous A2C (V3 §2; F10–F12; derived from PRD.md §3.4).
- [STATE_DESIGN.md](STATE_DESIGN.md) — 12-channel C2_moderate_12d state vector rationale.
- [ACTION_DESIGN.md](ACTION_DESIGN.md) — 7-action discrete set + masking rules.

## 2 — Architect the solution
- [PLAN.md](PLAN.md) — Architecture, C4-Container mermaid diagram, 10-phase build sequence, ADR pointers.
- [adr/](adr/) — 8 Architecture Decision Records (hybrid SDK + analysis-notebook, state, reward, action masking, terminal conditions, Streamlit GUI, ADR-007 documenting the NA path for the V3 §5 ApiGatekeeper requirement, and ADR-008 recording the LSTM world-model simplifications vs Ha & Schmidhuber 2018).
- [adr/ADR-007-no-api-gatekeeper.md](adr/ADR-007-no-api-gatekeeper.md) — V3 §5 NA justification: A3 has no rate-limited inference endpoint and only one one-shot Kaggle CLI invocation.
- [adr/ADR-008-lstm-world-model-simplifications.md](adr/ADR-008-lstm-world-model-simplifications.md) — three intentional departures from Ha & Schmidhuber 2018 (deterministic MSE head, rolling 7-step hidden, learned action embedding).

## 3 — Theory↔Code contract
- [THEORY.md](THEORY.md) — Brief's 12 equations transcribed verbatim in LaTeX, each cross-linked to the src/ file that implements it.
- [TRACE.md](TRACE.md) — 71-row traceability matrix (brief §, transcript, standing rules, CLAUDE.md constraints → deliverable + test).

## 4 — Build trail
- [TODO.md](TODO.md) — 53 phase-tagged tasks with definition-of-done per row.
- [PROMPTS.md](shared/PROMPTS.md) — Architect↔implementer evidence trail (Phase 0 → Phase 8) — the §1.4 contract in audit form.

## 5 — Cost + experiments
- [COST_ANALYSIS.md](COST_ANALYSIS.md) — Token-spend table across all phases + per-deliverable amortisation.
- [EXPERIMENTS.md](EXPERIMENTS.md) — Hypothesis-setup-result-verdict log for §7.6 discussion + DA4 comparison.

## 6 — Quality + Submission
- [QUALITY.md](QUALITY.md) — ISO/IEC 25010:2011 product-quality characteristics mapped to repo evidence (V3 §13).
- [SUBMISSION.md](SUBMISSION.md) — V3 §17.1–§17.6 grader-facing evidence table (bound by TRACE row R2).
- [adrl-001-ex03.pdf](../adrl-001-ex03.pdf) — moodle cover sheet (gitignored — contains PII).
- [../README.md](../README.md) — top-level README with Quick Start.
- [../notebooks/analysis.ipynb](../notebooks/analysis.ipynb) — §7.7 chart deliverables + LaTeX + §7.6 discussion.

## GUI
- [adr/ADR-006-streamlit-gui-framework.md](adr/ADR-006-streamlit-gui-framework.md) — Architecture Decision Record for the Streamlit GUI layer.
- [../src/gui/pages/README.md](../src/gui/pages/README.md) — page-by-page walkthrough of the multi-page Streamlit app.
- [UX.md](UX.md) — Nielsen's 10 usability heuristics + 5 quality criteria mapped to GUI pages and `assets/gui_*.png` screenshots (V3 §10).
- [assets/](assets/) — `gui_*.png` screenshots captured from the running app for the submission write-up.
