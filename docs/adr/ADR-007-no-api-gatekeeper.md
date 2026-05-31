# ADR-007 — No central ApiGatekeeper (NA per V3 §5)

Status: Accepted
Date: 2026-05-31
Decider: solo developer (architect role per CLAUDE.md §1.4)

## Context
V3 software submission guidelines §5 mandate that all external API calls go
through a centralized ApiGatekeeper class with rate-limit config (rate_limits.json,
version=1.00+), FIFO queue with backpressure, retry logic, and per-call logging.

## Decision
A3 does NOT implement an ApiGatekeeper. The §5 mandate is treated as NA
(Not Applicable) for this assignment.

## Justification — Why §5 does not apply to A3
1. Single one-shot external call. The only external network surface is the
   Kaggle dataset download in src/data/kaggle_client.py:_invoke_kaggle_cli(),
   which is invoked at most once per fresh checkout to populate data/raw/. The
   downloaded parquet is then cached on disk; subsequent runs read locally.
2. No rate-limited inference endpoint. A3 does not call OpenAI / Anthropic /
   any LLM provider at runtime — all ML happens locally on torch + numpy.
3. No multi-tenant client surface. The project is a solo academic submission;
   there is no third-party caller that could overwhelm us with bursty traffic.
4. The Kaggle CLI binary itself enforces its own auth + rate-limit politeness
   (kaggle.com policy + ~/.kaggle/kaggle.json). Wrapping it in a second
   gatekeeper would be redundant ceremony, not safety.

## Consequences
- If A3 ever grows to add an LLM-backed recommender / a live inference
  service / scheduled re-downloads, this ADR MUST be re-opened and an
  ApiGatekeeper implemented (src/services/api_gatekeeper.py with execute() +
  get_queue_status() per V3 §5).
- Forward-compat hook: the SDK has a single entry point for all logic
  (sdk.SDK class), so an ApiGatekeeper can be wired in as a constructor
  dependency without touching the GUI/CLI layer.

## Alternatives considered
- (A) Implement full ApiGatekeeper now → rejected: zero traffic to gate,
  pure ceremony, would inflate src/ LOC for no validated need.
- (B) Use a token-bucket library (limits / aiolimiter) → rejected: same
  problem (one shot/checkout doesn't need a bucket).
- (C) Document NA in PLAN.md only → insufficient per V3 §5 NA path; an ADR
  is the canonical justification artifact.

## V3 §5 traceability
| §5 requirement | A3 status | Evidence |
|---|---|---|
| Centralized ApiGatekeeper class | NA | this ADR |
| Per-call rate-limit config (rate_limits.json) | NA | this ADR |
| FIFO queue + backpressure | NA | this ADR |
| All external calls logged | Partial | src/data/kaggle_client.py logs to stdout via standard logging |
