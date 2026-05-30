# ADR-004 — Action Masking via Logits → −∞ Before Softmax

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-001 (architecture), ADR-002 (state representation), ADR-003 (reward weighting)

## Context

Brief §7.6 asks (and grades) the *explanation* of how expert knowledge is
folded into the policy, not only the final masking code. The action space
fixed by ADR-002 / ADR-001 is the 7-action discrete set
`{0:Rest, 1:Push, 2:Pull, 3:Legs, 4:FullBody, 5:Conditioning, 6:Mobility}`.
Two implementation choices exist:

- **Post-hoc rejection.** Sample an action from the unmasked Categorical
  distribution; if the sample is illegal, resample. Simple to code, but it
  biases the policy gradient: the surrogate loss is computed under the
  unmasked distribution while the trajectory is generated under the
  rejection-thinned distribution, so the importance ratio is not 1 and
  ignoring it introduces unbiased-gradient violations.
- **Masked-logit softmax.** Set the logits of illegal actions to −∞
  *before* the softmax. The policy distribution itself becomes a
  Categorical only over legal actions, so the policy-gradient estimator
  `∇log π(a|s) · G_t` is computed on the same distribution that generated
  the trajectory. This preserves the unbiasedness result of the
  policy-gradient theorem [Williams 1992; Sutton et al. 2000].

## Decision

Implement masking as **logits → −∞ before softmax** inside an
`ActionMaskService` injected into both the REINFORCE and A2C agents. The
service exposes one method, `mask(logits, state) -> masked_logits`, which
adds a vector of zeros and `-inf` entries (per legal / illegal action) to
the raw logits before the agent constructs the Categorical distribution.

Four masking rules are applied, each derived from the 12-d state vector
fixed in ADR-002:

1. **Rest mask.** Action 0 (Rest) is masked if the last 3 days were all
   Rest. (Derived from `streak_days_trained == 0` for the last 3
   timesteps, which the SDK passes alongside the state.) Rationale: a
   policy that converges to "Rest forever" achieves a zero penalty and
   near-zero gain; we want to forbid that absorbing state.
2. **Legs mask.** Action 3 (Legs) is masked if `soreness_legs > 0.8`.
   Rationale: training already-very-sore legs is the prototypical
   safety-relevant decision the brief uses to motivate masking ("trainee
   did Legs yesterday with high soreness → mask Legs next day").
3. **HIIT / Conditioning mask.** Action 5 (Conditioning) is masked if
   `rolling_7d_volume > overload_threshold`
   (`overload_threshold = 1.2 · baseline`, the same threshold that
   triggers the overload penalty in ADR-003). Rationale: piling
   conditioning on top of an already-overloaded week is the worst case
   the overload penalty exists to discourage; masking turns the soft
   penalty into a hard constraint when volume is dangerous.
4. **Mobility never masked.** Action 6 (Mobility) is always legal — it
   is the safety-net fallback when every other action is masked.
   Rationale: ensures the legal-action set is never empty (the softmax
   would otherwise produce `NaN`s) and gives the agent a meaningful
   "active recovery" choice that does not require Rest.

Worked example. Suppose state at *t* shows `soreness_legs = 0.83`,
`rolling_7d_volume = 1.05·baseline`, `streak_days_trained = 4`. Rule 2
fires → mask action 3 (Legs). Rules 1, 3 do not fire. Legal actions:
`{0, 1, 2, 4, 5, 6}`. Raw logits `[0.1, 0.2, 0.1, 0.4, 0.0, 0.1, 0.1]`
become `[0.1, 0.2, 0.1, -inf, 0.0, 0.1, 0.1]`. After softmax the
probability mass on action 3 is exactly 0 — verified by
`test_masked_logit_positions_yield_zero_probability`.

Implementation note. `ActionMaskService` lives in
`src/services/action_mask_service.py` (≤ 150 LOC per CLAUDE.md §1) and
its single public method returns a `torch.Tensor` of the same shape as
the logits, so the agent can use either `F.softmax(masked_logits, dim=-1)`
or `Categorical(logits=masked_logits)` interchangeably. The same service
is used at training time and at inference time, so there is no
train / inference policy drift.

## Rationale

The policy-gradient theorem requires that the policy used to sample
actions be the same one whose log-probabilities are differentiated. Logit
masking preserves that invariant by construction. Rejection sampling
violates it: the realised sampling distribution is the unmasked
distribution truncated by the rejection rule, not the unmasked
distribution itself, so the score-function estimator
`∇log π(a|s) · G_t` is the gradient of the wrong distribution. Huang &
Ontañón [reference [8] in the brief bibliography, FLAIRS 2022, "A
closer look at invalid action masking in policy gradient algorithms"]
formalise this and show empirically that logit masking is the unbiased
choice, which is the citation we use in the notebook's §7.6 cell.

The four masking rules above are the smallest set that simultaneously
(a) prevents the two failure modes the reward function flags as
unsafe — over-training a sore muscle group, and piling conditioning on
top of overload — and (b) prevents the absorbing-rest failure mode and
the empty-legal-set failure mode. The Guardrails framing the lecturer
used ("humans in the loop next to optimisation") is exactly this: the
human encodes the rules that the optimiser is not allowed to violate,
and the optimiser is then free to explore the rest of the action space.

## Consequences

**Positive.**

- Policy-gradient updates remain unbiased; no importance-sampling
  correction needed.
- The safety-relevant failure modes (overloading sore legs,
  conditioning on top of overload, absorbing rest) are forbidden by
  construction, not discouraged probabilistically. This lets us keep
  the overload weight λ₁ moderate (= 2.0 in ADR-003) instead of
  cranking it up to "scare the policy off" — a setting that would hurt
  gain-vs-penalty balance.
- The masking rules are state-only and SDK-injected, so the same
  service is reused across REINFORCE and A2C; no agent-specific
  duplication (CLAUDE.md §5).

**Negative.**

- Masking shrinks the effective action space the policy can explore,
  which can slow the policy's discovery of legitimate but
  near-boundary strategies (e.g. high-volume weeks deliberately
  followed by long mobility blocks). We mitigate by keeping the
  thresholds in `config.yaml` so they can be relaxed if exploration
  stalls.
- The four rules are hand-designed and may not align with real
  trainee biology. This is explicitly listed in the README "Honest
  Limitations" section and in the §7.6 discussion: the rules encode
  what *we* believe is safe, not what is provably safe.
- Rule 4 (Mobility never masked) means that in the worst case
  (every other rule fires) the policy is forced into Mobility. This
  is the intended behaviour but it limits the agent's expressivity in
  pathological states.

## Follow-ups

- `tests/test_action_mask_service.py`:
  `test_masked_logit_positions_yield_zero_probability`,
  `test_mobility_always_legal`,
  `test_rule_1_fires_after_three_rest_days`,
  `test_rule_2_fires_when_soreness_legs_above_threshold`,
  `test_rule_3_fires_when_volume_above_overload_threshold`,
  `test_unbiased_gradient_under_masking` (numerical check that the
  score-function estimator matches the analytic gradient on a 2-step
  toy MDP).
- `notebooks/analysis.ipynb` §7.6.1: one paragraph on logit-vs-rejection
  unbiasedness with the Huang & Ontañón [8] citation, the worked
  example above, the safety-vs-exploration trade-off framed as
  Guardrails, and a code-outline cell showing the masked-softmax
  implementation.
- All four rule thresholds (`rest_lookback_days = 3`,
  `legs_soreness_threshold = 0.8`,
  `volume_overload_threshold = 1.2`) live in `config/config.yaml` under
  `action_mask:` per CLAUDE.md §4.
