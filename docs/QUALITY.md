# QUALITY — ISO/IEC 25010:2011 quality-characteristics mapping (V3 §13)

This document maps each of the 8 ISO/IEC 25010 product-quality characteristics
to concrete evidence in this repo. It exists because V3 §13 mandates standing
against ISO/IEC 25010 and the V3 deep audit flagged the omission.

## 1. Functional Suitability (completeness, correctness, appropriateness)
- Brief §7 deliverables all present: see docs/TRACE.md
- 402 unit + integration tests pass; ≥85% branch coverage enforced by pyproject.toml fail_under=85
- Acceptance criteria documented per feature in docs/PRD.md

## 2. Performance Efficiency (time-behavior, resource-utilization, capacity)
- LSTM forward pass benchmarked at ~0.072 ms/step on developer macOS arm64
  (single-trainee inference, eval mode, no_grad, 200-iter mean after 20-iter warmup;
  reproduce via `uv run --active python -c "..."` — see commit message for §17.6 fix)
- Streamlit GUI uses @st.cache_resource for the heavy SDK instance (src/gui/state.py)
- A2C training scales linearly in episodes; no quadratic blow-up
- Parallel processing / thread safety: intentionally out-of-scope (single-process
  training, no multiprocessing / no torch.distributed). See docs/PLAN.md §6 and
  ADR-007 for rationale (academic single-GPU/CPU project; complexity > value here).
- Limitation: no formal perf benchmark suite (single-process academic project)

## 3. Compatibility (co-existence, interoperability)
- Python ≥3.11 pinned in pyproject.toml
- macOS arm64 (developer env) and Linux x86_64 (GitHub Actions Ubuntu 22.04) both validated by CI
- No platform-specific code paths; pathlib.Path everywhere (V3 §14.3)

## 4. Usability (learnability, operability, accessibility, etc.)
- 10-page Streamlit GUI with branded hero + bilingual (en/he) UX on Discussion page
- Nielsen-heuristic mapping: see docs/UX.md
- Screenshots for every page: docs/assets/gui_*.png

## 5. Reliability (maturity, availability, fault-tolerance, recoverability)
- Deterministic seeding (src/utils/seeding.py) — every entry point seeds torch+numpy+python random
- Tests assert reproducibility (e.g. tests/test_seeding.py)
- Graceful Kaggle fallback to synthetic deterministic trainee when CLI unavailable

## 6. Security (confidentiality, integrity, authenticity, non-repudiation, accountability)
- No secrets in source (V3 §7.4)
- .env in .gitignore; .env-example committed with placeholders
- PII redaction policy documented in CLAUDE.md
- Kaggle credentials via ~/.kaggle/kaggle.json (not committed)

## 7. Maintainability (modularity, reusability, analysability, modifiability, testability)
- All .py files ≤150 LOC (enforced by scripts/check_file_sizes.py in CI)
- Docstrings on every module + class (V3 §3.3)
- BaseTrainer + registry pattern for adding new algorithms without SDK edit (ADR-001)
- DRY: REINFORCE and A2C share rollout via BaseTrainer

## 8. Portability (adaptability, installability, replaceability)
- uv-managed via pyproject.toml + uv.lock (V3 §8.4)
- One-command setup: `uv sync --dev`
- No system dependencies beyond the Python runtime + (optional) Kaggle CLI
