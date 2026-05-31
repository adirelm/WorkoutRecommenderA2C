# State Design — C2_moderate_12d

This document is the canonical pointer for the workout-trainee state representation.

## Decision

State `s_t ∈ ℝ^12` per ADR-002 (C2_moderate_12d). Full channel list, ranges, and
rationale: [PRD §1.5](PRD.md#15-state-design-decision-c2_moderate_12d)
and [ADR-002](adr/ADR-002-state-representation.md).

## Implementation

`src/env/state.py` defines the frozen `State` dataclass and `STATE_CHANNEL_NAMES`
tuple. Use `State.to_array()` to produce a 12-dim float32 numpy vector for
neural-network input (channel order matches `STATE_CHANNEL_NAMES`).

## Why C2 (not C1 minimal-8d or C3 rich-16d)?

C1 cannot express per-chain soreness asymmetry → policy would over-train fatigued
chains. C3's extra channels (fatigue EWMA, intensity trend, recovery proxy)
duplicate signal already implicit in soreness + readiness without adding actionable
information at this LSTM-fit budget. C2 is the moderate-information design point
that the brief §7.3 grading explicitly rewards.

## Tests

- `tests/test_env_state.py` pins shape (12,), dtype float32, channel order, frozen
- `tests/test_workout_env.py::test_state_dim_is_twelve` is a smoke check
