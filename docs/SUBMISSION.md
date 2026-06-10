# SUBMISSION — A3 final-checklist evidence (V3 §17)

This file is the single grader-facing summary mapping each V3 §17.x checkbox
to concrete evidence on disk. TRACE.md row R2 binds to this doc.

## §17.1 — Structure + docs
- README.md with §-aware section headers
- docs/PRD.md + docs/PLAN.md + docs/TODO.md
- docs/THEORY.md (brief's 12 equations transcribed verbatim + cross-linked to src/)
- docs/shared/PROMPTS.md (architect-AI trail, 551 lines, 17 passes + decision log)
- 8 ADRs in docs/adr/
- 5 mermaid diagrams in docs/diagrams/

## §17.2 — Architecture + code
- src/sdk/sdk.py: single entry point + _TRAINER_REGISTRY
- src/services/base_trainer.py: BaseTrainer ABC
- src/model/masking.py: shared apply_mask helper (DRY)
- All .py ≤150 LOC (enforced by scripts/check_file_sizes.py)

## §17.3 — Tests + quality
- 440 tests collected / ≥91% coverage (last verified 2026-06-10, post Phase-15)
- pyproject.toml fail_under = 85
- TDD discipline: tests written before/with code
- Ruff clean (0 violations)
- CI on every push (.github/workflows/ci.yml)

## §17.4 — Config + security
- config/config.yaml + src/utils/config_loader.py (cached YAML loader)
- src/__version__.py = "1.2.0" synced with pyproject + config
- .env in .gitignore + .env-example committed
- uv-only (no pip/python -m anywhere in tracked code)
- ADR-007: no ApiGatekeeper needed (NA path documented)

## §17.5 — Research + viz
- 4 result charts: lstm_loss, reinforce_rewards, a2c_training, comparison —
  all regenerated Phase 11+ over the **frozen LSTM world model trained on the
  real Kaggle PHUL program** (audit F-1/F-2 closure)
- comparison.png is the Phase-13 headline: **30 seeds × 200 episodes with a
  masked-random baseline** (3 series; raw arrays in results/comparison_seeded.json)
- 1 sensitivity chart: lambda_sensitivity (5×5×3-seed, over the LSTM env —
  λ₂ active, λ₁ inert; see EXPERIMENTS E9)
- 1 hyperparameter chart: reinforce_lr_sweep (1e-3 > default 3e-4)
- results/data_quality_report.txt — §7.2.3 audit trail (312 rows, 26 time-encoded reclassified)
- 10 GUI screenshots in docs/assets/
- Cost analysis with per-model breakdown (docs/COST_ANALYSIS.md)

## §17.6 — Extension + standards
- ADR-001: BaseTrainer + registry extension point
- ADR-008: LSTM simplifications vs Ha & Schmidhuber 2018
- docs/QUALITY.md: 8 ISO/IEC 25010 characteristics mapped
- Git history: 110+ commits across 15 phases with §-ref subjects
- v1.0.0 + v1.1.0 + v1.1.1 + v1.2.0 release tags
- LICENSE (MIT) + lecturer (rmisegal) as read-only collaborator

## Honest limitations (V3 §1.4 transparency)
See docs/QUALITY.md "Honest limitations" section. Self-grade target: 93/100.
