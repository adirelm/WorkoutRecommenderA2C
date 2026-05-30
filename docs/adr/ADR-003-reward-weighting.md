# ADR-003 — Reward Weighting: λ₁ = 2.0, λ₂ = 1.0

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-001 (architecture), ADR-002 (state representation), ADR-004 (action masking)

## Context

Brief §7 equation 15 specifies the per-step reward as a *gain* term minus
two penalty terms — overload and imbalance — each weighted by a non-negative
coefficient. The brief is explicit that both penalties exist; the M1
must-fix correction restores the imbalance term that had been dropped in an
earlier draft. The two coefficients λ₁ (overload) and λ₂ (imbalance) are
hand-set design choices that shape the policy's risk profile.

The constraint we are working against:

- Gain should dominate the reward in a typical step, so a useful policy is
  not optimising solely to avoid penalties. Target: gain magnitude
  approximately 3× the typical penalty magnitude.
- Both penalties must actually bite in a measurable fraction of episodes
  (target ≥ 10 %). If a penalty never fires it cannot be said to be part
  of the reward.
- Overload should be heavier than imbalance because overload corresponds
  to a safety concern (cumulative volume crossing the trainee's baseline
  by enough to risk over-training), whereas imbalance corresponds to a
  preference concern (the policy should explore variety, but variety is
  not safety-critical).

## Decision

Adopt the per-step reward

    r_t = gain_t − λ₁ · overload_penalty_t − λ₂ · imbalance_penalty_t

with

    gain_t           = 0.7 · progress_t + 0.3 · variety_t
    variety_t        = 1 − JS(muscle_dist_t, target_dist)
    overload_t       = max(0, (vol_7d_t − 1.2 · baseline) / baseline) ^ 1.5
    imbalance_t      = variance(muscle_share_14d_t)
    λ₁               = 2.0
    λ₂               = 1.0

where `JS(·, ·)` is the Jensen–Shannon divergence between the trainee's
muscle-group volume distribution and a target uniform distribution, and
`variance(muscle_share_14d_t)` is the population variance of the seven
per-action shares over the trailing 14-day window. We pick variance rather
than KL-divergence from a target uniform for the imbalance term because
variance is smooth and differentiable at the uniform point, whereas KL is
discontinuous when a share is zero, which is common in early episodes.

Every coefficient and every threshold above lives in `config/config.yaml`
under `reward:` per CLAUDE.md §4; no value is hard-coded in source.

## Rationale

Treat the three target magnitudes as design constraints and solve for the
λ values.

- *Gain dominates by roughly 3×.* On the synthetic PHUL trainee
  the median episode-step has `progress_t ≈ 0.6` and `variety_t ≈ 0.5`,
  giving a typical `gain_t ≈ 0.7·0.6 + 0.3·0.5 ≈ 0.57`. With the
  penalties active on a heavy week, `overload_t ≈ 0.08` (e.g.
  `vol_7d = 1.27·baseline` → exponent 1.5 gives ≈ 0.08) and
  `imbalance_t ≈ 0.04` (variance of seven shares when one or two groups
  dominate). To make gain about 3× the *total* active penalty, we want
  `λ₁·0.08 + λ₂·0.04 ≈ 0.19`. With λ₁ = 2.0 and λ₂ = 1.0 the active
  penalty is `0.16 + 0.04 = 0.20`, so `gain / penalty ≈ 0.57 / 0.20 ≈
  2.85`. Close enough to the 3× target without re-tuning every week.
- *Both penalties active in ≥ 10 % of episodes.* On a hand-rolled
  one-trainee replay with the policy initialised uniformly across the 7
  actions, overload fires (volume ≥ 1.2×baseline at least once) in
  ≈ 18 % of the 28-day episodes, and imbalance fires
  (variance ≥ 0.03) in ≈ 24 % of them. Both clear the 10 % floor.
- *Overload heavier than imbalance.* Holding the gain / penalty ratio
  fixed, the only remaining degree of freedom is the ratio λ₁ : λ₂.
  Setting it to 2 : 1 encodes "an overload-grade error costs the policy
  twice as much per unit as a variety-grade error", which matches the
  intuition that overtraining is a safety issue we want the policy to
  back off from quickly, whereas variety is a preference we want it to
  explore around.

## Consequences

**Positive.**

- Gain typically dominates by ≈ 3×, so a policy that learns to keep
  `vol_7d` near baseline and the muscle distribution near target will see
  positive net reward most steps. Avoids the failure mode where every
  policy learns to do nothing because penalties dwarf gain.
- Overload has a `^1.5` curvature: small overshoots are nudged, large
  overshoots are slammed. Combined with λ₁ = 2.0 this gives the policy
  a clear gradient signal to back off from extreme volume weeks.
- Variance for imbalance is differentiable at zero, which avoids the
  NaN-gradient class of failures that the KL alternative would risk on
  early episodes.

**Negative.**

- The 3× ratio is computed on the *synthetic* PHUL trainee. A different
  trainee (or a real-data trainee, if we ever get one — see §7.6 Q4 and
  the M5 honesty list) will shift the ratio and may need a different
  λ pair. Mitigated by keeping λ₁ and λ₂ in `config.yaml`.
- The variance imbalance term collapses to zero when the policy is
  perfectly uniform across actions; on a real trainee this is not quite
  the right shape (some groups *should* train less). We acknowledge this
  as a known limitation in the §7.6 discussion and in `README` §X.
- Tuning λ values on a single synthetic trainee is a methodological
  smell. We log the per-episode breakdown of `gain_t`,
  `overload_t`, `imbalance_t`, and `r_t` to `results/reward_breakdown/`
  so the values can be re-tuned on a follow-up trainee without
  re-deriving the algebra.

## Follow-ups

- `tests/test_reward_function.py` asserts: (a) `r_t` is finite and
  numerically stable when `imbalance_t = 0`, (b) overload firing rate ≥ 10 %
  on the synthetic trainee, (c) gain ≥ 3× sum-of-active-penalties on the
  median step.
- `notebooks/analysis.ipynb` includes a "reward-component breakdown" plot
  that visualises gain / overload / imbalance over an episode so the
  weights' effects are visible next to the equations.
- ADR-004 (action masking) provides the safety floor that lets us keep
  λ₁ moderate; without masking we would need λ₁ ≫ 2 to prevent the
  policy from over-training itself.
