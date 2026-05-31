# COST_ANALYSIS — AI Tooling Cost for A3

## §1 Scope

This document accounts for the AI tooling cost of A3 across the dev cycle.
It exists in direct response to A1 lecturer feedback flagging two recurring
gaps: (a) over-confidence in self-grading and (b) the absence of any honest
accounting for what the AI tooling actually *cost* to produce the artefact.
The numbers below are approximations rounded to the nearest dollar; they
are derived from per-workflow agent counts and average per-agent token
budgets, not from line-itemed billing exports. Treat them as
order-of-magnitude rather than receipts-grade.

## §2 Token-spend table

| Phase | Workflow ID | Agents | Tokens (input + output approx) | $ at standard rates |
|---|---|---|---|---|
| 0 plan | a3-comprehensive-planning (wmyibh2pl) | 17 | ~660k | ~$3.30 |
| 0 docs | a3-finalize-planning-docs (w058pq917) | 6 | ~335k | ~$1.68 |
| 0 publish | a3-init-repo-and-publish (waftqx6d9) | 5 | ~112k | ~$0.56 |
| 0 scaffold | a3-phase0-scaffolding (wjzk25zze) | 7 | ~154k | ~$0.77 |
| 0 validate (×5+1) | v1p0..v5p0 | ~30 | ~1.2M | ~$6.00 |
| 0 fix | a3-phase0-fix-execution (w6nwpgt9e) | 9 | ~184k | ~$0.92 |
| 1 impl | a3-phase1-data-ingest (wk6wqdd4o) | 8 | ~228k | ~$1.14 |
| 1 validate (×5) | v1p1..v5p1 | ~25 | ~800k | ~$4.00 |
| 1 fix | a3-phase1-fix-execution (wwf0t4q7l) | 7 | ~194k | ~$0.97 |
| 2 impl | a3-phase2-env-layer (w2cegj0u1) | 9 | ~312k | ~$1.56 |
| 2 validate (×5) | v1p2..v5p2 | ~25 | ~900k | ~$4.50 |
| 2 fix | a3-phase2-fix (wfldknrfr) | 7 | ~274k | ~$1.37 |
| 3 impl | a3-phase3-lstm-world-model (w7yrmqbp5) | 8 | ~315k | ~$1.58 |
| 3 validate (×5) | v1p3..v5p3 | ~30 | ~1.1M | ~$5.50 |
| 3 fix | a3-phase3-fix-execution (w5yva7ho1) | 7 | ~260k | ~$1.30 |
| 4 impl | a3-phase4-reinforce (w3turidy5) | 9 | ~313k | ~$1.57 |
| 4 validate (×5) | v1p4..v5p4 | ~30 | ~1.0M | ~$5.00 |
| 4 fix | a3-phase4-fix-execution (wrc03ip1t) | 6 | ~162k | ~$0.81 |
| 5 impl | a3-phase5-a2c + recovery (wfuhqps1i + w1oxotf6g) | 12 | ~427k | ~$2.14 |
| 5 validate (consolidated) | wx1gp782i | 6 | ~180k | ~$0.90 |
| 5 fix | a3-phase5-fix (wzm0alic6) | 4 | ~114k | ~$0.57 |
| 6 impl | a3-phase6-sdk-cli (wz88srac1) | 6 | ~196k | ~$0.98 |
| 6 validate | wtoim87nh | 6 | ~167k | ~$0.83 |
| 6 fix | wa2uidtif | 4 | ~139k | ~$0.70 |
| 7 notebook | whnnzmzpd | 3 | ~102k | ~$0.51 |
| 8 submission prep | (this) | 3 | ~50k | ~$0.25 |
| **TOTAL** | | **~270 agents** | **~9.5M tokens** | **~$48** |

Rates approximate: $0.005/k input, $0.015/k output, mixed → ~$0.005/k
effective; rounded. The split between input and output tokens is heavily
skewed toward input (≈85/15) because every fan-out agent re-reads the
brief, the relevant ADRs, and the section of source it's reviewing.

## §3 Per-deliverable cost amortisation

- Each `src/` file (~28 files × ~$1.50 amortised) ≈ $42 implementation
- Each `docs/` file (~10 files × ~$0.50) ≈ $5
- 4 §7.7 charts × ~$0.25 ≈ $1

These three lines sum to ≈ $48, which matches the row total above to within
rounding. They are *amortised* in the accounting sense — i.e. each file
absorbs a proportional share of validation and fix-execution cost on top
of its raw implementation cost, not just the tokens spent producing the
first draft. A naive "just count impl tokens" view would understate the
true cost by roughly 3× because validation + fix dominate the bill.

## §4 What we'd cut if budget were 50% lower

In retrospect, the highest-cost / lowest-marginal-value lines were:

1. **The 5-parallel-workflow validation set per phase** (×6 phases × ~$5 = $30).
   After Phase 2, validation findings converged to similar themes
   (test-quality, TRACE freshness, docstring honesty). Consolidating to a
   single workflow with 5 parallel axes (as in `v-p5-consolidated`,
   `v-p6-consolidated`) cuts each phase's validation cost by ~70% with
   minimal coverage loss. Had we adopted that pattern from Phase 0, the
   total would have dropped to roughly $30 instead of $48.
2. **Recovery workflows** (e.g., `a3-phase5-complete-missing-files` due
   to partial completion of the impl workflow) — ~$2 wasted that smarter
   workflow scripting (`verify-files-exist` before commit) would have
   caught for free. The fix is mechanical (one more shell check in the
   workflow YAML); the cost was avoidable.

## §5 What we'd spend MORE on if budget allowed

1. **Multi-seed convergence study** (50+ seeds × 500 episodes for both
   REINFORCE + A2C) to make the comparison statistically meaningful —
   Phase 7 notebook only does 3 seeds × 30 episodes for speed. A proper
   study would cost on the order of $15-25 in additional agent time
   plus non-trivial compute, but it would let us replace the current
   "REINFORCE wins on this seed set" framing with an actual confidence
   interval.
2. **Hyperparameter sweep on λ_1, λ_2** (overload + imbalance weights) —
   ADR-003 picks defaults without sensitivity analysis. A grid sweep
   would cost ~$10 of agent time and would let us defend the chosen
   weights instead of just declaring them.
3. **Real Kaggle download + full dataset preprocessing** (requires
   kaggle credentials — current build uses fixture CSVs). The fixtures
   are honest about being fixtures, but a real ingest would close the
   loop end-to-end. Estimated ~$3-5 of agent time to wire up plus
   credential management outside the AI cost envelope.

## §6 Comparison to no-AI baseline

A solo dev writing this from scratch (no LLM assistance) would likely
take 60-100 hours at typical RL-engineer rates ($75-150/hr) ⇒
$4,500-15,000 labour. AI tooling cost (~$48) is ~0.3% of the
labour-substitution alternative. Wall-clock dev time was ~6-8 hours of
human review/architect time + ~12-16 hours of parallel AI-agent
wall-clock.

This comparison is **not** a claim that "AI replaced 100 hours of
engineering" — it didn't. The human architect time (~6-8 h) and the
review/sign-off loop (per the §1.4 Human ↔ AI contract in `CLAUDE.md`)
are the load-bearing parts. The $48 number is the cost of the
*implementation* substrate, not the design or the acceptance criteria.
The honest framing: AI tooling at this price point makes the
architect-led model economically obvious, but the architect work itself
is what makes the artefact worth submitting.

## §7 Lessons for next assignment

1. **Use the consolidated single-workflow validation pattern from the
   start.** The five-workflow fan-out gave diminishing returns after
   Phase 2 and was the single biggest avoidable cost line.
2. **Add an explicit `verify_files_exist` step in every impl workflow
   before the commit step.** This would have caught the Phase 5
   partial-completion that triggered the recovery workflow.
3. **Cache the brief PDF read across phase agents** (currently re-read
   in every phase). The brief is ~12 pages of static content; pinning it
   once and referencing the cache key would save a non-trivial fraction
   of input tokens across the ~270-agent run.
4. **Prefer agent prompts with explicit `import pytest` reminders** to
   avoid recurring `NameError` in scaffolded test files — this was a
   small but repeated drag on fix-execution cost across Phases 1-3.
5. **Budget the cost envelope up front** (per CLAUDE.md §1.4, this is a
   non-delegable human decision). A1 had no envelope at all; A3
   retroactively reconstructed one. A4 should declare a target cost
   before Phase 0 kicks off, and treat overruns as a signal worth
   investigating rather than absorbing silently.
