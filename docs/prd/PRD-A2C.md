# PRD — A2C (Synchronous Advantage Actor-Critic)

**Anchor:** `#prd-a2c`
**V3 §2 compliance:** This file satisfies V3 §2 (per-algorithm PRDs) by
extracting the A2C-specific product requirements from the master PRD into
a focused, single-algorithm document. The consolidated PRD is the master;
this file is a derived view.
Reference: docs/PRD.md §3.4, §6, §7.1 (rows F10–F12) — master document.

---

## 1. Objective

Implement synchronous A2C — shared trunk, actor head (softmax) + critic
head (scalar V), one-step advantage
`A_t = r_t + γV_ψ(s_{t+1}) − V_ψ(s_t)`, entropy bonus on the actor loss,
critic minimises ½δ²_t MSE — and train it against the LSTM rollout with
the same seed schedule as REINFORCE so the comparison (F13) is paired.

Brief reference: §7.5 (A2C).
Master reference: [docs/PRD.md §3.4](../PRD.md#34-a2c).

## 2. Functional requirements (extracted)

- **F10.** Implement synchronous A2C: shared trunk, actor head (softmax) +
  critic head (scalar V), advantage `A_t = r_t + γV_ψ(s_{t+1}) − V_ψ(s_t)`,
  entropy bonus on the actor loss, critic minimises ½δ²_t MSE.
- **F11.** Train against the LSTM rollout with the same seed schedule as
  REINFORCE (so the comparison is paired).
- **F12.** Emit the A2C training graph required by brief §7.7.

Full F-row context: [docs/PRD.md §3.4](../PRD.md#34-a2c).

## 3. Inputs

- Trained `WorldModelResult` (from the LSTM PRD — see
  [PRD-LSTM.md](PRD-LSTM.md)) providing `rollout(s0, policy, horizon)`.
- 7-action discrete set + action mask
  ([docs/ACTION_DESIGN.md](../ACTION_DESIGN.md)).
- Reward function from [docs/PRD.md §1.7](../PRD.md#17-reward-design).
- Hyperparameters from `config/config.yaml` (`a2c.*`): actor LR, critic LR,
  discount γ, entropy coefficient, gradient-clip norm (0.5), episode count,
  seed schedule (paired with REINFORCE, see F11).

## 4. Outputs

- Trained `A2CPolicy` returned by `TrainingSDK.train_a2c(world_model, ...)`.
- Per-episode return log + per-episode critic-loss log.
- `results/a2c_training.png` — A2C training graph (actor return + critic
  loss over episodes).
- Policy callable usable inside `world_model.rollout(...)` for the
  comparison plot (F13) and for `TrainingSDK.recommend(s_t)` (F15).

## 5. Acceptance criteria

| Req | Acceptance criterion                                                          | Evidence pointer                            |
|-----|-------------------------------------------------------------------------------|---------------------------------------------|
| F10 | A2C advantage equals `r + γV(s') − V(s)` symbolically                         | `tests/test_a2c.py::test_advantage_form`    |
| F11 | A2C trained with the same seed schedule as REINFORCE                          | `tests/test_a2c.py::test_paired_seeds`      |
| F12 | `results/a2c_training.png` emitted                                            | `notebooks/analysis.ipynb` cell 7           |

Master DoD table: [docs/PRD.md §7.1](../PRD.md#71-per-requirement-dod).

## 6. Theory pointer

Advantage-update identity transcribed in LaTeX and cross-linked to source
in [docs/THEORY.md](../THEORY.md).

## 7. Out of scope (for this PRD)

- Asynchronous A3C / multi-worker variants — A3 is synchronous-only.
- PPO / GAE / TRPO extensions (master PRD §8 / future work).
- Continuous action space (master PRD §8).

---

**Master PRD:** [docs/PRD.md](../PRD.md) — authoritative source for scope,
non-functional requirements, public API, and project-level DoD.
