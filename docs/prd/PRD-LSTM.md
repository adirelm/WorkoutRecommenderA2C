# PRD — LSTM World Model

**Anchor:** `#prd-lstm`
**V3 §2 compliance:** This file satisfies V3 §2 (per-algorithm PRDs) by
extracting the LSTM-specific product requirements from the master PRD into
a focused, single-algorithm document. The consolidated PRD is the master;
this file is a derived view.
Reference: docs/PRD.md §3.2, §6, §7.1 (rows F4–F6) — master document.

---

## 1. Objective

Train a learned dynamics model `f_φ: (s_t, a_t) → s_{t+1}` over the 21-day
training window (chronological split, 21/7) of the synthetic trainee log,
so that downstream policy-gradient learners (REINFORCE, A2C) can train
against rollouts instead of the real environment.

Brief reference: §7.3 (world-model training with teacher forcing).
Master reference: [docs/PRD.md §3.2](../PRD.md#32-world-model).

## 2. Functional requirements (extracted)

- **F4.** Train an LSTM `f_φ: (s_t, a_t) → s_{t+1}` on the 21-day training
  window with teacher forcing.
- **F5.** Report training and validation MSE loss curves on the 12-d state.
- **F6.** Expose `world_model.rollout(s0, policy, horizon)` returning a
  trajectory of length `horizon` for downstream RL training.

Full F-row context: [docs/PRD.md §3.2](../PRD.md#32-world-model).

## 3. Inputs

- `LogbookHandle` from `TrainingSDK.prepare_data()` — 28-row parquet with
  the 12-d C2_moderate_12d state schema (see
  [docs/STATE_DESIGN.md](../STATE_DESIGN.md)).
- 21/7 chronological split (F3 — train on first 21 days, validate on last 7).
- Action embedding for the 7 discrete actions
  ([docs/ACTION_DESIGN.md](../ACTION_DESIGN.md)).
- Hyperparameters from `config/config.yaml` (`world_model.*`): hidden size,
  learning rate, epochs, teacher-forcing ratio, seed.

## 4. Outputs

- `WorldModelResult` returned by `TrainingSDK.train_world_model(logbook)`
  containing trained weights, train/val MSE per epoch, and the rollout
  callable.
- `results/lstm_losses.png` — train + validation MSE curve over epochs.
- `world_model.rollout(s0, policy, horizon)` → `np.ndarray` of shape
  `(horizon, 12)` with no NaNs (used by REINFORCE and A2C trainers).

## 5. Acceptance criteria

| Req | Acceptance criterion                                                          | Evidence pointer                                       |
|-----|-------------------------------------------------------------------------------|--------------------------------------------------------|
| F4  | LSTM train MSE decreases monotonically (5-epoch median) on the 21-day window  | `tests/test_world_model.py::test_train_loss_decreases` |
| F5  | Validation loss curve emitted and saved as `results/lstm_losses.png`          | `notebooks/analysis.ipynb` cell 3                      |
| F6  | `rollout(s0, π_uniform, 14)` returns a `(14, 12)` array with no NaNs          | `tests/test_world_model.py::test_rollout_shape`        |

Master DoD table: [docs/PRD.md §7.1](../PRD.md#71-per-requirement-dod).

## 6. Architectural simplifications

Three intentional departures from Ha & Schmidhuber 2018 are recorded in
[docs/adr/ADR-008-lstm-world-model-simplifications.md](../adr/ADR-008-lstm-world-model-simplifications.md):
deterministic MSE head, rolling 7-step hidden, learned action embedding.

## 7. Out of scope (for this PRD)

- Stochastic / mixture-density output head (see ADR-008).
- Multi-trainee population dynamics (master PRD §8).
- Real-time fine-tuning during policy training.

---

**Master PRD:** [docs/PRD.md](../PRD.md) — authoritative source for scope,
non-functional requirements, public API, and project-level DoD.
