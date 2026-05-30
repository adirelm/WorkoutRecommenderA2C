# ADR-002 — State Representation: 12-Dimensional Moderate State Vector

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-001 (architecture), ADR-003 (reward weighting), ADR-004 (action masking)

## Context

Brief §1.3 says the state is "a vector of numbers" (the CartPole example uses
four). Brief §7.3 (state representation for the workout recommender) tells us
the state must summarise the trainee's recent training history in a form the
LSTM world-model and the policy can both consume, and the lecturer noted
verbally that state design is graded — we must justify which dataset columns
become state features and which become actions.

Three candidates were considered:

- **C1 — minimal 5-d.** `[fatigue, soreness_total, readiness, vol_7d,
  day_in_cycle]`. Cheap and easy to debug but loses muscle-group resolution,
  so the reward's variety term (ADR-003) cannot be computed from the state
  alone.
- **C2 — moderate 12-d.** Per-muscle-group soreness, rolling volume, streak
  / recovery flags, balance ratio, adherence, weekly progress. Enough
  resolution for the reward and for action masking, still small enough for an
  LSTM and a 1-layer 128-unit policy net to train on a single synthetic
  trainee.
- **C3 — rich 30+-d.** Add HRV, sleep, RPE, soft-tissue scores, micro-cycle
  flags. Closer to what a real wearable stack would produce but unsupported
  by the Kaggle dataset (plan-content only) and dishonest given the §7.6 Q4
  ask.

## Decision

Adopt **C2 — moderate 12-dimensional state vector** with the following
features (order is the SDK's canonical order; tests pin both order and
shape):

1. `fatigue` ∈ [0, 1] — exponentially-decayed cumulative volume across all
   muscle groups; the brief's "global readiness signal" (§7.3.1).
2. `soreness_push` ∈ [0, 1] — chest / shoulders / triceps soreness state,
   updated by the world model from yesterday's Push or FullBody action.
3. `soreness_pull` ∈ [0, 1] — back / biceps soreness, same update rule for
   Pull or FullBody.
4. `soreness_legs` ∈ [0, 1] — quads / hamstrings / glutes soreness; gates
   the Legs-mask rule in ADR-004.
5. `soreness_core` ∈ [0, 1] — core soreness; updated by FullBody and
   Conditioning.
6. `readiness` ∈ [0, 1] — `1 − fatigue` smoothed across a 3-day window; the
   brief's "ready to train hard today" feature (§7.3.2).
7. `rolling_7d_volume` ∈ [0, ∞), normalised by trainee baseline — feeds the
   overload term in the reward (ADR-003) and gates the
   HIIT / Conditioning mask in ADR-004.
8. `streak_days_trained` ∈ ℤ≥0 — count of consecutive non-Rest days; used
   by the Rest-mask rule (ADR-004) and as an adherence signal.
9. `days_since_last_rest` ∈ ℤ≥0 — symmetric to streak; helps the policy
   distinguish "long streak with a rest yesterday" from "long streak with no
   rest". Per brief §7.3.3 (recovery context).
10. `muscle_balance_push_vs_pull` ∈ [−1, 1] — signed share difference of
    Push-derived volume vs Pull-derived volume over the last 14 days; feeds
    the imbalance term in the reward (ADR-003).
11. `adherence_signal` ∈ [0, 1] — rolling fraction of planned sessions
    completed in the last 7 days, as a proxy for trainee engagement
    (brief §7.3.4).
12. `weekly_progress` ∈ [−1, 1] — change in average per-session volume from
    week *t − 1* to week *t*, clipped; the brief's "are we progressing"
    feature (§7.3.5).

All twelve features are computed in `src/data/state_builder.py`, exposed
through `WorkoutRecommenderSDK.get_state()`, and consumed by both the LSTM
world-model and the policy / value heads.

## Rationale

C1 cannot support the reward's variety term — variety is `1 − JS(muscle_dist,
target)` and without per-group soreness in the state, the agent cannot learn
to anticipate the imbalance penalty from state alone, only from reward
feedback after it lands. C3 introduces features the Kaggle dataset does not
back (no HRV, no RPE, no sleep), which would force us to synthesise them and
then misrepresent the result; that runs straight into §7.6 Q4 (data
limitations) and the M5 honesty rule.

C2 is the smallest representation that lets every other ADR be evaluated
from the state alone:

- ADR-003 reward eq. 15 needs `rolling_7d_volume` (overload), per-muscle
  shares (imbalance), and weekly progress (gain). C2 carries all three.
- ADR-004 action-masking rules need `soreness_legs` (Legs mask),
  `rolling_7d_volume` (HIIT mask), and `streak_days_trained` (Rest mask).
  C2 carries all three.
- The LSTM world model in brief §7.4 has a hidden state of 64; a 12-d input
  keeps the LSTM small enough to train on a single synthetic trainee
  without obviously over-fitting and matches the "1-layer 128-neuron"
  guidance from the lecturer's transcript at 01:06:59.

## Consequences

**Positive.**

- Every reward term and every masking rule is computable from the state
  alone — no out-of-band signals leak into the policy.
- 12 features × 28-day episodes × a few thousand episodes is small enough
  that the entire training run fits on a laptop CPU within the rubric's
  realistic compute envelope.
- The state vector is the documented contract between data, environment,
  world-model, and agent layers — extending the state in a later
  assignment is a single-file change in `state_builder.py` plus a test
  shape update.

**Negative.**

- Per-muscle-group soreness (features 2–5) is synthesised from the action
  history by a hand-tuned decay rule; this is a modelling assumption, not a
  measurement, and is called out explicitly in `README` §X "Honest
  Limitations" and in the §7.6 discussion.
- `adherence_signal` is derived from plan-content data and so collapses to
  near-1 on a synthetic trainee. We log this and treat it as a placeholder
  that would become meaningful with real wearable data.
- 12 dimensions is enough that the policy's 128-unit hidden layer is the
  binding capacity constraint; if a later experiment needs richer state we
  must revisit the policy head size before adding features.

## Follow-ups

- `tests/test_state_builder.py` pins shape (12,), dtype `float32`, and the
  ordered list of feature names.
- `docs/STATE_DESIGN.md` mirrors this ADR in trainee-facing language for the
  notebook's discussion cells.
- ADR-003 and ADR-004 cite this state schema by feature index.
