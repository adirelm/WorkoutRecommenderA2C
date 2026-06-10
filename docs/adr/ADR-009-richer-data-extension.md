# ADR-009 — Richer-data extension path (real biometric outcomes)

**Status:** Proposed (extension design — brief §7.6 "extend the project with richer data")
**Date:** 2026-06-10

## Context

The pipeline trains on the *prescribed* content of a real Kaggle program
(Optimized PHUL): day-level action sequence + Σ(sets·reps) volumes. The dataset
carries **no biological outcomes** — the physiological response (fatigue,
soreness, readiness) is supplied by deterministic trainee rules (brief §7.6 Q4).
Consequently (a) the LSTM learns plan rhythm filtered through a simulated body,
and (b) the shaped reward is flat enough inside the action mask that
masked-random is near-optimal (EXPERIMENTS E4).

## Decision (proposed design)

Extend the state/reward with measured outcomes from wearables + logs:

1. **New inputs** (per day): morning resting HR + HRV, per-muscle-group DOMS
   self-report (0-10), session RPE, and rolling 1RM-proxy strength markers.
2. **State**: 12-d → 16-18-d; new channels normalised per ADR-002 conventions.
   `STATE_CHANNEL_NAMES` grows; LSTM input width follows automatically.
3. **Transition model**: retrain the LSTM on *(state, action) → measured next
   state* — replacing the SyntheticTrainee rules with data, which is exactly the
   World-Model promise (Ha & Schmidhuber 2018) the current design stubs.
4. **Reward**: refit λ₁/λ₂ (and the gain weights) by supervised preference or
   regression against measured recovery/progress, instead of hand-design —
   directly attacking the flat-reward ceiling E4 exposed.
5. **Validation**: sim-to-real check on ≥5 trainees × ≥8 weeks; per-trainee
   held-out weeks as the val split (chronological, as today).

## Consequences

- Closes the two honest limitations at once (plan-content-only data;
  random-beats-RL ceiling) — RL headroom becomes an empirical question.
- Multi-trainee data breaks the single-trajectory memoisation
  (`world_model_builder` cache keys gain a trainee id) and motivates the PRD F2
  parquet logbook that is currently deferred.
- Out of scope for this assignment (no such dataset in the brief); recorded so
  the extension is a design, not a slogan.

## Alternatives considered

- **Multi-program mixture** (train the LSTM on several Kaggle programs):
  improves plan-rhythm generality but still contains zero outcome data — does
  not address Q4. Kept as a cheaper orthogonal step (config
  `dataset.fallback_programs` already lists candidates).
