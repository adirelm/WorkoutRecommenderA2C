# WorkoutRecommenderA2C — Bar-Ilan Vibe Coding Workshop A3

*Status: planning phase (Phase 0 pending; see docs/TODO.md).*

## What this is
An attempt to reproduce the §7 pipeline of Assignment 3 (LSTM transition model + REINFORCE + A2C on a single synthetic workout trainee), with explicit limitations.

## Documents
- [PRD](docs/PRD.md) — product requirements
- [PLAN](docs/PLAN.md) — architecture + ADRs
- [TODO](docs/TODO.md) — phased task list
- [THEORY](docs/THEORY.md) — brief equations transcribed verbatim
- [TRACE](docs/TRACE.md) — requirement traceability matrix
- [ADR-001..004](docs/adr/) — architecture / state / reward / action-masking decisions

## Honest Limitations
1. The Kaggle dataset is plan-content, not real workout outcomes — the LSTM cannot learn true physiological response.
2. Single synthetic trainee → no claim about population generalisation.
3. LSTM fit on plan-derived sequences may memorise periodisation patterns rather than learn dynamics.
4. Reward is hand-designed; alignment with real strength-training goals is not validated.
5. All numeric claims (convergence, comparison) cite seed + episode-count + mean ± std — never bare adjectives.

## Dataset

This project uses the **600K+ Fitness Exercise & Workout Program Dataset** (Adnane Louardi) from Kaggle:
- URL: https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset
- License: ODbL 1.0 — non-commercial use only
- Source: Boostcamp.app
- Chosen program for the synthetic trainee: **PHUL** (12 weeks, 60 min/session, intermediate, strength+hypertrophy hybrid). Fallbacks: GZCLP, nSuns 5/3/1.
- The brief's `programs_detailed_boostcamp_kaggle.csv` is named `fitness_exercises.csv` in the actual download; both name conventions are handled in `src/data/kaggle_client.py`.
- See PRD §1.5.0 for the data quality contract (negative reps reclassified as seconds; rest days inserted).

## Reproducibility caveats
- CUDA non-determinism: scatter_add and index_add are non-deterministic on CUDA. We default to CPU; CUDA opt-in is documented.
- MPS LSTM kernels are not bit-deterministic on Apple Silicon.
- Multi-worker DataLoader shuffle order depends on worker scheduling — we run single-worker for the reported curves.
- Mixed-precision adds drift; we report fp32 numbers.
