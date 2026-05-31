# CHANGELOG

All notable changes to this project. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.1.0] — 2026-05-31
### Added
- Streamlit GUI surface (10 pages) — see ADR-006 + PRD §1.8 + Phase 9 in PROMPTS.md
- Bar-Ilan blue theme + live training charts + interactive Recommend page (12 sliders) + Action Masking demo
- CLI verb 7 launch-gui
- scripts/capture_gui_screenshots.py + docs/assets/gui_*.png screenshots
- streamlit + plotly dependencies (+ streamlit-extras + playwright dev)

### Changed
- docs/PRD.md §1.3 — removed "no GUI required" line; added §1.8 GUI Surface
- docs/PLAN.md §3 — added src/gui/ layer responsibility row
- docs/COST_ANALYSIS.md — Phase 9 cost rows
- tests/test_commits_reference_sections.py — relaxed allowed-prefix regex to include PII scrub + Revert + Merge

## [1.0.0] — 2026-05-31
### Added
- Phase 0-8 build: planning artefacts, data ingest (KaggleClient + Preprocessor + Aggregator + ProgramFilter), env layer (State + SyntheticTrainee + Reward + ActionMask + WorkoutEnv), LSTM world model + trainer + adapter + trajectory builder, REINFORCE (PolicyNet + trainer + baseline + weighted-CE), A2C (ActorCriticNet + trainer + comparator), WorkoutSDK + CLI menu, analysis notebook (4 §7.7 charts + LaTeX + §7.6 discussion + Action Masking).
- 271/271 tests passing, 98% coverage on src/, ruff zero violations, all .py ≤150 LOC.
- 5-workflow validation cycle per phase with adversarial test-quality probes + theory-code fidelity audits.
- adrl-001-ex03.pdf cover sheet (gitignored — PII; upload to moodle manually).
- v1.0.0 tag.
