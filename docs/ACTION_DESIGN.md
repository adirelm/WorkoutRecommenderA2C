# Action Design — 7-action discrete set

This document is the canonical pointer for the workout-recommendation action space.

## Decision

7 discrete actions: `{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`.

Full per-action semantics: [PRD §1.6](PRD.md#16-action-design).

## Why 7 (not 4 or 6)?

4-action (Rest/Push/Pull/Legs) collapses too much variety. 6-action loses Mobility
as a distinct guardrail-safe option. 7 is the brief-grading sweet spot for a 28-day
episode (cited workflow recommendation).

## Why these specific actions?

- Rest: explicit recovery — no soreness/fatigue addition, decay-only.
- Push/Pull/Legs: standard PPL split aligning with the chosen PHUL program.
- FullBody: distributes load across all chains — useful for low-readiness days.
- Conditioning: HIIT-style cardio — separate signal from strength training.
- Mobility: ALWAYS-LEGAL safe option (ADR-004) — prevents empty-legal-action sets.

## Action Masking

See [ADR-004](adr/ADR-004-action-masking.md) for the soreness/streak-driven masking
rules that integrate expert knowledge into the policy distribution.

## Tests

- `tests/test_env_state.py::test_action_count_seven` (cardinality)
- `tests/test_action_mask.py` (all four masking rules verified)
- `tests/test_workout_env.py::test_action_space_is_seven` (smoke)
