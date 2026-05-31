"""§7.7 Long-form markdown copy for every Streamlit page.

Keeping prose here (instead of inline in page modules) means each page
file stays under the CLAUDE.md §1 150-LOC budget *and* the wording can
be reviewed in one place. Every string cites the brief § it satisfies,
the governing ADR, and the relevant equation from ``docs/THEORY.md``.
"""

from __future__ import annotations

PAGE_HOME_INTRO = """
### Workout Recommender — REINFORCE vs A2C

**Brief §7.7 · ADR-001 (hybrid architecture) · §1.4 contract**

This app trains two policy-gradient agents on a synthetic-trainee
environment and lets you compare them side-by-side. The full pipeline
— `env → world-model → trainer → SDK → GUI` — is documented in
`docs/PRD.md` and `docs/THEORY.md`.

* Pick a page from the sidebar to inspect each phase.
* Every chart is reproducible from a single seed (default `42`).
* The §1.4 contract (human-architect / AI-implementer) is enforced
  per-file via `CLAUDE.md` and per-prompt via `docs/PROMPTS.md`.
"""

PAGE_DATA_INTRO = """
### Data & Environment

**Brief §7.3 · ADR-002 (12-channel state) · ADR-004 (action mask)**

The synthetic trainee produces a 12-channel daily state vector
(fatigue, per-muscle soreness, readiness, rolling volume, streak,
rest-gap, push/pull balance, adherence, weekly progress).
See `THEORY.md` §"State design" for the closed-form transition.

The action space is 7 discrete workouts. The mask service
(ADR-004) zeroes illegal picks *before* the softmax, following
Huang & Ontañón (2022) — gradients only flow through legal actions.
"""

PAGE_LSTM_INTRO = """
### LSTM World Model

**Brief §7.3 · ADR-001 · THEORY.md §"World-model rollout"**

A small LSTM is trained offline on synthetic-trainee rollouts and
then **frozen**. At inference time the env can swap the closed-form
trainee for the LSTM without changing its public API — this is the
hybrid-architecture promise from ADR-001.

The frozen weights live in `results/lstm_world_model/`. Training
metrics shown here come from the most recent run logged via
`set_last_world_model_history()`.
"""

PAGE_REINFORCE_INTRO = r"""
### REINFORCE (Monte-Carlo policy gradient)

**Brief §7.4 · THEORY.md §"REINFORCE update"**

Sutton & Barto §13.3. For each episode trajectory $\tau$ we maximise

$$ \nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,(G_t - b)\right] $$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return-to-go and
$b$ is a running-mean baseline (variance reduction, ADR-003).

The chart below updates **per episode** via an `st.empty()` placeholder
fed by `src/gui/callbacks.py` — the trainer itself is not modified.
"""

PAGE_A2C_INTRO = r"""
### A2C — Advantage Actor-Critic

**Brief §7.5 · THEORY.md §"A2C update"**

Mnih et al. (2016). Two heads share no weights here (ADR-001):
the actor minimises $-\log\pi_\theta(a\mid s)\,A(s,a)$ and the
critic regresses $V_\phi(s)$ to the bootstrapped return
$r + \gamma V_\phi(s')$. The advantage is

$$ A(s_t,a_t) = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t). $$

Entropy bonus ($\beta = 0.01$) discourages premature collapse,
gradient clipping ($\|g\| \le 0.5$) prevents critic blow-ups.
"""

PAGE_COMPARISON_INTRO = r"""
### Comparison — mean ± std over N seeds

**Brief §7.6 · ADR-003 · `src/services/comparator.py`**

Each algorithm is re-trained from scratch on `N` seeds and the
per-episode reward curves are averaged. Shaded bands are
$\pm \sigma$ across seeds — *not* confidence intervals. With
$N{=}3$ this is a sanity check, not a publication result; the
brief §7.6 only asks for a qualitative comparison.
"""

PAGE_RECOMMENDATION_INTRO = """
### Single-step recommendation

**Brief §7.5 · `WorkoutSDK.recommend(state)`**

Pick any state, then ask the most-recently trained policy what to
do next. The bar chart shows the masked-softmax distribution
(illegal actions are forced to 0 *before* normalisation); the
critic value (A2C only) is the expected return from this state.
For REINFORCE the value column shows `—` (no critic). The
`UNKNOWN_REWARD` sentinel from `src/sdk/types.py` makes that
explicit so `0.0` is never confused with "predicted zero reward".
"""

PAGE_DISCUSSION_RTL_HE = """
<div dir="rtl" lang="he" style="text-align:right;font-size:15px;line-height:1.7;">

### דיון — §1.4

**אדם כאדריכל · AI כמיישם.** המתודולוגיה של הסדנה דורשת
שכל החלטה ארכיטקטונית (PRD, ADRs, ספי קבלה, חתימה על
PR) תיוותר בידי האדם, וה-AI יבצע *רק* מול מפרט מאושר.
החוזה הזה מתועד ב-`CLAUDE.md` ומגובה בכל ה-prompts
ב-`docs/PROMPTS.md`.

**מה למדנו.** REINFORCE מתכנס לאט (שונות גבוהה ב-Monte-Carlo
returns). A2C מצמצם שונות עם ה-baseline שמספק ה-critic,
אבל דורש כיוונון של `entropy_coef` כדי לא לקרוס למדיניות
דטרמיניסטית מוקדם מדי. ראו `docs/THEORY.md` להוכחות
המתמטיות ו-`docs/EXPERIMENTS.md` לתוצאות המספריות.

</div>
"""  # noqa: RUF001 — intentional Hebrew characters in RTL discussion copy.
