# PRD — REINFORCE Policy Gradient

**Anchor:** `#prd-reinforce`
**V3 §2 compliance:** This file satisfies V3 §2 (per-algorithm PRDs) by
extracting the REINFORCE-specific product requirements from the master PRD
into a focused, single-algorithm document. The consolidated PRD is the
master; this file is a derived view.
Reference: docs/PRD.md §3.3, §6, §7.1 (rows F7–F9) — master document.

---

## 1. Objective

Implement Williams 1992 REINFORCE with a softmax policy over the 7 discrete
actions and a learned-baseline variant (running mean of episodic returns)
to reduce gradient variance, then train it against the LSTM rollout so the
brief §7.4 + §7.7 deliverables can be generated.

Brief reference: §7.4 (REINFORCE).
Master reference: [docs/PRD.md §3.3](../PRD.md#33-reinforce).

## 2. Functional requirements (extracted)

- **F7.** Implement REINFORCE with a softmax policy over the 7 discrete
  actions and a learned-baseline variant (running mean of episodic returns)
  to reduce gradient variance.
- **F8.** Train against the LSTM rollout; log per-episode return.
- **F9.** Emit the REINFORCE reward curve required by brief §7.7.

Full F-row context: [docs/PRD.md §3.3](../PRD.md#33-reinforce).

## 3. Inputs

- Trained `WorldModelResult` (from the LSTM PRD — see
  [PRD-LSTM.md](PRD-LSTM.md)) providing `rollout(s0, policy, horizon)`.
- 7-action discrete set + action mask
  ([docs/ACTION_DESIGN.md](../ACTION_DESIGN.md)).
- Reward function from [docs/PRD.md §1.7](../PRD.md#17-reward-design).
- Hyperparameters from `config/config.yaml` (`reinforce.*`): learning rate,
  discount γ, episode count, baseline mode, entropy coefficient, seed
  schedule (paired with A2C, see F11).

## 4. Outputs

- Trained `REINFORCEPolicy` returned by
  `TrainingSDK.train_reinforce(world_model, ...)`.
- Per-episode return log (numpy array, length = number of training
  episodes).
- `results/reinforce_rewards.png` — reward curve over episodes (mean ± 1σ
  across seeds where applicable).
- Policy callable usable inside `world_model.rollout(...)` for the
  comparison plot (F13).

## 5. Acceptance criteria

| Req | Acceptance criterion                                                                   | Evidence pointer                                       |
|-----|----------------------------------------------------------------------------------------|--------------------------------------------------------|
| F7  | REINFORCE update equals `∇log π(a\|s) · (G_t − b)` symbolically (unit test)            | `tests/test_reinforce.py::test_grad_form`              |
| F8  | Mean episodic return over last 20 % of training > first 20 % (seed + std reported)     | `tests/test_reinforce.py::test_learning_progress`      |
| F9  | `results/reinforce_rewards.png` emitted                                                | `notebooks/analysis.ipynb` cell 5                      |

Master DoD table: [docs/PRD.md §7.1](../PRD.md#71-per-requirement-dod).

## 6. Theory pointer

Policy-gradient identity transcribed in LaTeX and cross-linked to source in
[docs/THEORY.md](../THEORY.md).

## 7. Out of scope (for this PRD)

- Actor-critic / advantage formulation — that is the A2C PRD
  ([PRD-A2C.md](PRD-A2C.md)).
- Continuous action space (master PRD §8).
- Online human-in-the-loop training.

---

**Master PRD:** [docs/PRD.md](../PRD.md) — authoritative source for scope,
non-functional requirements, public API, and project-level DoD.
