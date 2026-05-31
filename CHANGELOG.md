# CHANGELOG

All notable changes to this project. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.1.1] — 2026-05-31
### Added
- `src/__version__.py` + version bump to 1.1.1 across pyproject/config/init (303c0e8)
- `src/env/workout_env_helpers.py` — extracted helpers so `workout_env.py` stays ≤150 raw LOC (9b7fb2d)
- `BaseTrainer` base class + trainer registry under `src/training/` for pure-registry dispatch (5540b3d, bcfedb7)
- ADR-007 documenting the NoApiGatekeeper / NA path for V3 §5 ApiGatekeeper requirement (8f08d88)
- `docs/QUALITY.md` mapping ISO/IEC 25010 characteristics to in-repo evidence (V3 §13) (616d73f)
- `docs/UX.md` mapping Nielsen heuristics + quality criteria for the Streamlit GUI (V3 §10) (569726b)
- λ_1 × λ_2 sensitivity-sweep heatmap (V3 §9.1) (dae40c8)
- Per-model token/cost table + batch-processing notes in `docs/COST_ANALYSIS.md` (V3 §11) (3993fc2)
- `tests/unit/` + `tests/integration/` split with shared `tests/conftest.py` fixtures (7a8e62d)
- Contributing / Usage / Examples sections in `README.md` + fixed broken docs/README.md ADR-006 link (608520d)
- Parallel-processing scope note + filled-in `<X ms/step>` placeholder in `docs/QUALITY.md` §17.6 (8dd9b5d)

### Fixed
- CI yaml redundant smoke step removed — main pytest run already covers `tests/test_gui_*.py` (fc8ef54)
- GUI screenshots regenerated (10 pages) — previous capture caught a stale broken state (9925aa0)
- `BaseTrainer` refactor exposing public `sdk.ensure_env()` (5540b3d)
- Per-model cost analysis populated in `docs/COST_ANALYSIS.md` (3993fc2)
- ISO 25010 quality-characteristic mapping documented (616d73f)
- UX / Nielsen heuristic mapping documented (569726b)
- λ_1 × λ_2 sensitivity sweep added to satisfy V3 §9.1 (dae40c8)
- Pure-registry dispatch in `WorkoutSDK.train()` — dropped if/elif fallback (V3 §12 open-closed) (bcfedb7)
- Broken slow e2e test — now passes proper (s, a, s') triples into `build_windows` (469e94f)
- PII scrub — stripped leaked Google Drive absolute paths from notebook cell outputs (05d9193)
- ADR-007 added so V3 §5 ApiGatekeeper requirement has a documented NA path (8f08d88)
- Orphan `results/figures/live_training_overhead.png` reference removed from `EXPERIMENTS.md` (82204c2)
- pip-install hint dropped from `kaggle_client.py`; moved `PROMPTS.md` under `docs/shared/` (ec3e19a)
- ruff/format clean-up: reformatted `__version__.py` + `workout_env_helpers.py` + synced `uv.lock` to 1.1.1 (2ce16de)

### Changed
- SDK encapsulation — private `_ensure_env` promoted to public `ensure_env()` for the new trainer registry (5540b3d)

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
