# ADR-005 — Terminal Conditions: Single-Case Collapse for the Workout-Recommendation MDP

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-001 (architecture), ADR-002 (state), ADR-004 (action masking)

## Context

The brief §2 lists three canonical RL terminal conditions: (1) reward target reached,
(2) failure, (3) episode horizon exhausted. The TRACE matrix row 2.2 plans
`test_episode_termination_three_cases` covering all three.

For the workout-recommendation MDP specifically, cases (1) and (2) do not naturally
apply: there is no scalar "target reward" the trainee must hit (the goal is the
long-horizon cumulative gain), and there is no "catastrophic failure" event (Action
Masking + ADR-004 already prevents injury-equivalent action choices before they
happen — so the env never sees a failure signal at step granularity).

## Decision

Implement only case (3) `done = (step_count >= episode_length)` in
`src/env/workout_env.py`. Document the omission of cases (1) and (2) explicitly as a
deliberate MDP design choice for the continuous-recommendation problem.

The test `test_episode_termination_three_cases` is replaced by:
- `test_done_after_episode_length_steps` (case 3 — already present)
- `test_action_mask_prevents_injury_event` (mechanism that replaces case 2 —
  belongs in test_action_mask.py, already present as the soreness/Legs masking test)
- N/A — case 1 has no analog in this MDP.

## Consequences

- TRACE row 2.2 reads "MDP timeout only — cases (1) and (2) collapsed per ADR-005";
  Phase-2 work is consistent with the brief once this rationale is documented.
- If a future extension reintroduces explicit failure events (e.g., a "trainee
  quits" model), this ADR is revisited.
