# THEORY.md — Bridge from Brief Math to `src/`

**Course:** Bar-Ilan Vibe-Coding Workshop, L07 — REINFORCE / A2C / Workout Recommender
**Brief:** Dr. Yoram Segal, May 2026 (33-page PDF, Hebrew).
**Repository:** `WorkoutRecommenderA2C` (branch `assignment-3`).
**Scope of this file:** transcribe the brief's twelve load-bearing equations into LaTeX *verbatim*, and for each one nail down (a) where it lives in `src/`, (b) which test asserts it, and (c) which paper the brief cites.

This document is the contract between **the math the brief grades us on** and **the code we ship**. Every equation here is reachable from `tests/test_spec_eq*.py` and from a docstring in `src/`. If an equation drifts between this file and `src/`, the test breaks — that is the design.

A short reading guide: §1 frames the objective. §2 derives the vanilla REINFORCE update. §3 introduces baseline subtraction. §4 introduces reward-to-go. §5 introduces the Advantage / A2C critic. §7 specialises the reward to the workout-recommender domain. §X closes with the cross-entropy equivalence that lets us implement REINFORCE in one PyTorch line.

The action space throughout this document is **|A| = 7**:
`{0: Rest, 1: Push, 2: Pull, 3: Legs, 4: FullBody, 5: Conditioning, 6: Mobility}`.
This is *not* the brief's "4 to 8" hand-wave — it is a tuning decision (see `docs/adr/ADR-002-action-space.md`) and it must match the policy-head output width in `src/policy/policy_net.py`.

---

## §1.2. Objective function $J(\theta)$ — what we are maximising

The brief (p. 7, eq. 1) defines the policy-optimisation target as the expected discounted return over trajectories sampled from the current policy:

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^{t}\, r_t\right]
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory and $\gamma \in [0,1]$ is the discount rate (Discount Rate / מקדם היוון).

**How this maps to `src/`.** `src/training/objective.py` exposes `discounted_return(rewards, gamma)` returning $\sum_t \gamma^t r_t$ for a single rollout, and `expected_return(rollouts, gamma)` returning the Monte-Carlo mean across a batch of rollouts. The trainer never optimises $J$ directly — it optimises a sample-based surrogate (eq. 2 below). $J$ exists in code only as the *metric* we plot on the training curve.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §3.1 — "objective curve" (placeholder until the notebook is authored).

**Reference.** Sutton & Barto [1], chapter 13 (policy-gradient setup).

---

## §2.4. Vanilla REINFORCE update

Brief p. 10, eq. 2. The classical Williams 1992 update — one network, sample-based, model-free:

$$
\theta \;\leftarrow\; \theta \;+\; \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t
$$

where $G_t$ is the return-from-step-$t$ accumulator (the brief calls it `Return` at this point and only formalises it as Reward-to-Go later, in eq. 7).

**How this maps to `src/`.** `src/training/reinforce_trainer.py` implements one episode = one update. The forward pass produces logits $z_t$ from the policy net, `torch.distributions.Categorical(logits=z_t).sample()` emits $a_t$, the env returns $r_t$, and at episode end we compute $G_t$ per eq. 7. The loss is the cross-entropy surrogate (§X below); the update direction is mathematically eq. 2.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §4.2 — "REINFORCE training loop" (placeholder).

**Reference.** Williams [3], "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, 1992.

---

## §2.5. Log-derivative trick (eq. 3)

Brief p. 11, eq. 3. The identity that makes eq. 2 a *sample-based* estimator instead of an integral:

$$
\nabla_\theta \!\int p_\theta(\tau)\, R(\tau)\, d\tau \;=\; \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau \;=\; \mathbb{E}_{\tau \sim p_\theta}\!\left[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\right]
$$

The brief calls this **משפט גרדיאנט המדיניות** (Policy-Gradient Theorem). The key consequence is "model-free": the gradient estimator does not require the env transition $P(s_{t+1}\mid s_t, a_t)$.

**How this maps to `src/`.** This identity is *not* a line of code — it is the proof that justifies why `loss.backward()` on the cross-entropy surrogate is an unbiased estimate of $\nabla_\theta J$. It is referenced in the docstring of `src/training/reinforce_trainer.py::compute_policy_loss`.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §2.3 — "log-derivative numerical check" (placeholder).

**Reference.** Sutton, McAllester, Singh, Mansour [4], "Policy gradient methods for reinforcement learning with function approximation," NeurIPS 1999. Original derivation in Williams [3].

---

## §3.2. Baseline-subtracted update (eq. 4)

Brief p. 13, eq. 4. The variance-reduction trick — subtract a baseline $b$ that does not depend on $a_t$:

$$
\theta \;\leftarrow\; \theta \;+\; \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \bigl(G_t - b\bigr)
$$

Brief §3.2: trajectory return averages to 8. A trajectory returning 12 has advantage +4 (above baseline); a trajectory returning 5 has advantage −3 (below baseline). The policy is encouraged to repeat actions from the +4 trajectory and discouraged from those in the −3 trajectory.

**How this maps to `src/`.** `src/training/baseline.py` implements `running_mean_baseline(returns)` and `value_baseline(states, V_psi)`. In REINFORCE we use the running-mean variant. In A2C we use the value-function baseline $b(s) \approx V^\pi(s)$ — which makes $G_t - b$ collapse into the Advantage (eq. 8).

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §4.4 — "variance with vs without baseline" (placeholder).

**Reference.** Sutton & Barto [1] §13.4; Schulman et al. GAE [5] for the formal bias-variance analysis.

---

## §4.2. Reward-to-Go (eq. 7)

Brief p. 15, eq. 7. The Markov-respecting credit assignment:

$$
G_t \;=\; \sum_{t'=t}^{T} \gamma^{\,t' - t}\, r_{t'}
$$

Rather than crediting every step with the *total* trajectory return, each step is credited only with rewards that *follow* it. The brief grounds this in **causality**: an action taken at time $t$ cannot retroactively change rewards already collected at $t' < t$ — that contribution has expected value zero in the gradient anyway, but including it inflates variance.

**How this maps to `src/`.** `src/training/credit_assignment.py::reward_to_go(rewards, gamma)` returns $[G_0, G_1, \ldots, G_T]$ in O(T) via a reverse cumulative sum. Used by both REINFORCE and A2C as the target for the policy update; in A2C it is also the regression target for the critic when using Monte-Carlo returns instead of bootstrapped TD.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §4.3 — "G_t reverse-cumsum sanity check" (placeholder).

**Reference.** Sutton & Barto [1] chapter 13; Schulman et al. [5] (causality argument, eq. 17 in the GAE paper).

---

## §5.4. Advantage function (eq. 8)

Brief p. 18, eq. 8. The "pure" definition of advantage — how much better is action $a_t$ than the *average* action at state $s_t$:

$$
A^{\pi}(s_t, a_t) \;=\; Q^{\pi}(s_t, a_t) \;-\; V^{\pi}(s_t)
$$

The brief's intuition (p. 17, §5.3): you tell the critic "my state is worth 5", and you actually got 7 from your action — the advantage is +2, *local* to this state, not contaminated by the rest of the trajectory.

**How this maps to `src/`.** We never *materialise* eq. 8 directly — training two separate networks ($Q^\pi$ and $V^\pi$) would double the parameter count. Instead we use the Bellman identity $Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})]$ to collapse the advantage into the one-sample TD error (eq. 9 / eq. 17). The "pure" form lives in `docs/THEORY.md` only — for derivation purposes.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §5.1 — "Q vs V illustration" (placeholder).

**Reference.** Sutton & Barto [1] §6 (Bellman), Schulman et al. [5] (GAE generalises eq. 8).

---

## §5.4.1. TD error δ_t (eq. 9)

Brief p. 18, eq. 9. The sample-based, biased-but-bounded-variance estimator of Advantage that powers all modern Actor-Critic:

$$
\delta_t \;=\; r_t \;+\; \gamma\, V_\psi(s_{t+1}) \;-\; V_\psi(s_t)
$$

The brief's key remark (p. 18, end of §5.4): $\delta_t$ is **not** the Advantage itself — it is a one-sample, biased-but-low-variance *estimator* of $A^\pi(s_t, a_t)$. The bias goes to zero in expectation; the variance is bounded by a single transition's noise instead of the whole trajectory's.

**How this maps to `src/`.** `src/training/td_error.py::compute_td_error(rewards, values_t, values_tp1, gamma)` returns $\delta_t$ per step. The same function feeds the actor update (eq. 10) and the critic loss (eq. 12).

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §5.2 — "TD-error stability across episodes" (placeholder).

**Reference.** Schulman et al. [5] §3 (GAE($\lambda=0$) reduces to eq. 9); Mnih et al. A3C [6].

---

## §5.5. A2C actor update (eq. 10)

Brief p. 19, eq. 10. Identical to REINFORCE-with-baseline (eq. 4) except $G_t - b$ is replaced by the TD-error $\delta_t$:

$$
\theta \;\leftarrow\; \theta \;+\; \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \delta_t
$$

The brief's framing (p. 17, §5.1): the actor is the "football player on the pitch" and the critic is "the coach standing on the sideline saying *normally from this state you don't pass right*". The actor never sees the full trajectory return — only the critic's bite-sized $\delta_t$.

**How this maps to `src/`.** `src/agents/a2c_agent.py::actor_update(states, actions, td_errors)` calls `loss = -(log_probs * td_errors.detach()).sum()` then `loss.backward()`. **Critical detail**: `.detach()` on $\delta_t$ — gradients must flow into $\theta$ via $\log\pi$, never into $\psi$ via the critic. (This is the most common A2C bug; we test for it explicitly in `tests/test_a2c_detaches_critic.py`.)

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §5.3 — "A2C actor loss curve" (placeholder).

**Reference.** Mnih et al. [6], "Asynchronous methods for deep reinforcement learning," ICML 2016 (A3C paper, where synchronous A2C is the natural simplification).

---

## §5.5.1. Critic update (eq. 11) and critic loss (eq. 12)

Brief p. 19, eq. 11 (critic SGD step) and eq. 12 (squared-TD loss):

$$
\psi \;\leftarrow\; \psi \;+\; \beta\, \delta_t\, \nabla_\psi V_\psi(s_t)
$$

$$
L(\psi) \;=\; \tfrac{1}{2}\, \delta_t^{\,2} \;=\; \tfrac{1}{2}\, \bigl(r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)\bigr)^{2}
$$

The brief's framing: the critic is "wrong by $\delta_t$"; minimising the MSE drives $V_\psi(s_t) \to \mathbb{E}[r_t + \gamma V_\psi(s_{t+1})]$ — exactly the Bellman fixed-point.

**How this maps to `src/`.** `src/agents/a2c_agent.py::critic_update(states, td_errors)` calls `loss = 0.5 * (td_errors ** 2).mean()` then `loss.backward()`. The actor and critic share *no* parameters in our implementation — two separate `torch.optim.Adam` instances, one for $\theta$ and one for $\psi$.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §5.4 — "critic-loss curve" (placeholder).

**Reference.** Mnih et al. [6]; Sutton & Barto [1] §6 (TD-learning fixed-point).

---

## §7.3 LSTM World Model (eq. 14)

The workout-trainee state is partially observable: yesterday's fatigue and soreness depend on history $h_t = (s_0, a_0, \ldots, s_{t-1}, a_{t-1})$, not solely on $s_{t-1}$. This is the POMDP framing of brief §1.4. To recover a Markovian transition we learn a recurrent **world model** [7] that maps history to a sufficient statistic:

$$ \hat{s}_{t+1} = f_\phi(s_t, a_t, h_t) $$

where $f_\phi$ is an LSTM whose hidden state $h_t$ carries the relevant history. Once trained, $f_\phi$ serves as the (frozen) transition kernel that REINFORCE and A2C roll out against in Phases 4 and 5 — a learned simulator standing in for the unknown true MDP transition [2, 7].

**How this maps to `src/`.** `src/model/lstm_world.py::LSTMWorldModel` is $f_\phi$. Architecturally it concatenates the 12-dim state with an 8-dim learned action embedding (action embedding is preferred over one-hot per Ha & Schmidhuber 2018) into a 20-dim input, passes it through `nn.LSTM(hidden=64, num_layers=1)`, and projects the last hidden state through a `Linear(64 → 12)` head. `src/model/lstm_trainer.py::LSTMTrainer` supervises $f_\phi$ with MSE between predicted and observed $s_{t+1}$ on a sliding-window dataset built by `src/model/dataset.py::build_windows`. The trajectory is generated by `src/model/trajectory_builder.py::generate_trajectory` running WorkoutEnv (Phase 2 SyntheticTrainee) under a baseline policy.

**Design choice — $h_t$ recomputation.** The implementation recomputes $h_t$ from a fixed-length window of past $(s, a)$ pairs at each call rather than carrying the LSTM hidden state across calls. This trades exact equation literalism for stateless API parity with `SyntheticTrainee.next_state` (the Phase-2 transition we are replacing). The window length matches the brief §7.3.1 `window_len` config knob. Both regimes converge to the same fixed point as `window_len` grows.

**Honest limitation.** $f_\phi$ is trained on plan-content data from the Kaggle program filtered to a single synthetic trainee — it has not seen real outcome data, and its predictions can only be as faithful as the SyntheticTrainee state-evolution rules. Brief §7.6 Q4 makes this caveat explicit.

**References.** [2] Kaelbling, Littman & Cassandra (1998) POMDP framework. [7] Ha & Schmidhuber (2018) Recurrent world models.

---

## §7.4. Reward (eq. 15) — the domain-specific shaping

Brief p. 29, eq. 15. The reward we hand to the policy after each simulated training day:

$$
r_t \;=\; \text{gain}_t \;-\; \lambda_1 \cdot \text{overload\_penalty}_t \;-\; \lambda_2 \cdot \text{imbalance\_penalty}_t
$$

The brief leaves the three terms as English placeholders. Our concrete operationalisation (decided by the human architect, documented in `docs/adr/ADR-003-reward-shaping.md`):

- $\text{gain}_t = 0.7 \cdot \text{progress}_t + 0.3 \cdot \text{variety}_t$, where $\text{progress}_t$ is the normalised week-over-week increase in `total_volume` and $\text{variety}_t = 1 - \mathrm{JS}(\text{muscle\_dist}_{14d}, \text{target})$ (Jensen-Shannon divergence to a uniform target over the 7 muscle groups).
- $\text{overload\_penalty}_t = \max\!\left(0,\, \frac{\text{vol}_{7d} - 1.2 \cdot \text{baseline}}{\text{baseline}}\right)^{1.5}$ — a soft super-linear penalty above 120% of baseline. Exponent 1.5 (not 2) keeps gradients informative near the threshold.
- $\text{imbalance\_penalty}_t = \mathrm{Var}(\text{muscle\_share}_{14d})$ — variance across the 7 muscle-group shares over the last 14 days. We pick variance over KL because variance is smooth and differentiable everywhere; KL spikes when any share goes to zero, which destabilises the policy gradient.
- $\lambda_1 = 2.0$, $\lambda_2 = 1.0$ — overload weighted heavier than imbalance because overload maps to real-world injury risk, while imbalance only maps to suboptimality. Both live in `config/config.yaml` under `reward.lambda_1` and `reward.lambda_2`.

**How this maps to `src/`.** `src/env/reward.py::compute_reward(state, action, history)` returns the scalar $r_t$. Each term has its own pure function (`progress_term`, `variety_term`, `overload_term`, `imbalance_term`) so we can unit-test each in isolation. `tests/test_spec_eq15.py` asserts: (a) reward is purely a function of state+history, (b) $\lambda_1, \lambda_2$ are read from config (not hardcoded), (c) overload exponent is 1.5, (d) imbalance is variance (not KL).

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §7.4.2 — "reward components over a 28-day rollout" (placeholder).

**Reference.** Brief §7.4 (no external paper — this is domain-specific shaping); Sutton & Barto [1] §17 on reward shaping pitfalls.

---

## §7.4.1. REINFORCE update for the workout policy (eq. 16)

Brief p. 29, eq. 16. Same formula as eq. 2, restated in the §7 context to make the actor-only baseline explicit:

$$
\theta \;\leftarrow\; \theta \;+\; \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t
$$

In the workout-recommender pipeline: $s_t$ is the daily state vector (7-day rolling volume, muscle-distribution, week-index, day-in-cycle, soreness proxy); $a_t \in \{0, \ldots, 6\}$ is the next-day workout *type*; $G_t$ is the discounted sum of eq. 15 rewards from day $t$ to end-of-episode (28-day horizon).

**How this maps to `src/`.** `src/agents/reinforce_agent.py::train_episode(env)` runs one episode against the LSTM-simulated environment, collects $(s_t, a_t, r_t)$ tuples, computes $G_t$ via `reward_to_go`, and applies the cross-entropy surrogate loss (§X). The LSTM is loaded with `requires_grad=False` — the policy never updates the world model.

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §7.4.3 — "REINFORCE training curve over LSTM world" (placeholder).

**Reference.** Williams [3]; Sutton et al. [4].

---

## §7.5. A2C advantage in the workout pipeline (eq. 17)

Brief p. 30, eq. 17. Identical structure to eq. 9, restated as $A_t$ (not $\delta_t$) to emphasise it is *used* as the advantage estimate in the actor update:

$$
A_t \;=\; r_t \;+\; \gamma\, V_\psi(s_{t+1}) \;-\; V_\psi(s_t)
$$

**How this maps to `src/`.** `src/agents/a2c_agent.py::train_episode(env)` collects $(s_t, a_t, r_t, s_{t+1})$ tuples; calls $V_\psi$ on both $s_t$ and $s_{t+1}$ (the latter `.detach()`-ed so critic-bootstrap gradients do not leak into the actor); applies the actor update (eq. 10) using $A_t$ in place of $G_t$; and applies the critic update (eq. 11, MSE on $\delta_t$).

**Notebook cross-link.** `notebooks/analysis.ipynb` cell §7.5.2 — "A2C vs REINFORCE training curves, mean ± std over N seeds" (placeholder).

**Reference.** Mnih et al. [6]; Schulman et al. [5].

---

## §6.2 Glossary — symbol cheat-sheet

Reproduced verbatim from the brief's Table 1 (p. 22) for reference:

| Symbol | Definition | Practical role |
|---|---|---|
| $\pi_\theta(a \mid s)$ | Policy — vector of action probabilities | Actor head |
| $G_t = \sum_{k=t}^{T} \gamma^{\,k-t} r_k$ | Reward-to-Go | Per-step credit signal |
| $b(s) \approx V^\pi(s)$ | Baseline | Variance-reduction control variate |
| $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | Advantage | "How much better than average" |
| $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ | TD-error | One-sample, low-variance estimator of $A$ |

---

## §X. Closing note — REINFORCE ≡ weighted cross-entropy

The brief's §2.6 and §2.7 (pages 12) make the single most implementation-critical observation in the entire document, which the lecture spends ~5 minutes on:

> REINFORCE is mathematically equivalent to **weighted cross-entropy** between the policy's predicted action distribution and the one-hot encoding of the sampled action, where the weight is $G_t$.

Formally:

$$
L_{\text{REINFORCE}} \;=\; -\sum_{t} G_t \cdot \log \pi_\theta(a_t \mid s_t)
$$

In PyTorch this is one line:

```python
loss = (F.cross_entropy(logits, action, reduction='none') * G_t.detach()).sum()
loss.backward()
```

**Why this matters.**
1. We get GPU-optimised cross-entropy "for free" — no custom autograd, no manual log-softmax, no numerical stability issues.
2. The `.detach()` on $G_t$ is non-negotiable — $G_t$ is treated as a *constant* weight; gradients flow only through $\log\pi_\theta$.
3. The same implementation, with $G_t$ swapped for $\delta_t$, gives us the A2C actor loss. The two algorithms differ by **one line**:

```python
weight = G_t.detach()        # REINFORCE
weight = td_error.detach()   # A2C
loss = (F.cross_entropy(logits, action, reduction='none') * weight).sum()
```

This is why `src/training/loss.py` exposes a *single* function `weighted_ce_loss(logits, actions, weights)` shared by both agents (DRY — see CLAUDE.md §5). `tests/test_loss_equivalence.py` asserts numerical equality between the hand-rolled $-\log\pi \cdot G$ form and the `F.cross_entropy` form to 1e-6 tolerance.

**Reference.** Brief §2.6, §2.7 (the only place the equivalence is written out); Williams [3] for the original derivation; PyTorch docs for `F.cross_entropy`.

---

## Bibliography (brief §8, verbatim reference list)

The numbers in `[N]` throughout this file map to:

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
2. L. P. Kaelbling, M. L. Littman, A. R. Cassandra, "Planning and acting in partially observable stochastic domains," *Artificial Intelligence*, 1998.
3. R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, 1992.
4. R. S. Sutton, D. McAllester, S. Singh, Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," NeurIPS 1999.
5. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," ICLR 2016.
6. V. Mnih et al., "Asynchronous methods for deep reinforcement learning," ICML 2016.
7. D. Ha and J. Schmidhuber, "Recurrent world models facilitate policy evolution," NeurIPS 2018.
8. S. Huang and S. Ontañón, "A closer look at invalid action masking in policy gradient algorithms," FLAIRS 2022.

---

## Honest scope of this document

This file transcribes the math. It does **not** claim the implementation matches every nuance — that claim is the job of `tests/test_spec_eq*.py`. If a test for an equation here is missing or skipped, the equation is *aspirational*, not realised. The `final_review_progress.md` tracker carries the up-to-date list of which equations have green tests and which are still in red.

The reward shaping in §7.4 (eq. 15) is hand-designed. It may not align with real-world training goals — see `README.md §Honest Limitations` for a head-on answer to brief §7.6 question 4 ("what are the main limitations of using plan-content data instead of real workout outcomes?"). No claim of physiological realism is made anywhere in this repo.
