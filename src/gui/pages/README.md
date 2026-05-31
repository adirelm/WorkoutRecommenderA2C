# `src/gui/pages/` — dev-side index

Streamlit multi-page app for WorkoutRecommenderA2C. Pages are discovered
by `src/gui/app.py` via `st.Page("pages/NN_name.py", ...)` and rendered
through `st.navigation(...)`.

## Numbering convention

`NN_name.py` — two-digit zero-padded prefix sets sidebar order. Numbers
are stable: renumbering breaks bookmarks and screenshots in `docs/`.
Range `01–10` is reserved for the brief-aligned pages below; future
exploratory pages start at `90_`.

## "Consumer of SDK" rule (CLAUDE.md §3)

Every page is a **pure consumer** of `src.sdk.sdk.TrainingSDK` (see
`docs/PRD.md` §1 "GUI as SDK consumer" and ADR-006). No page may
import from `src.model`, `src.env`, `src.data`, or `src.services`
directly. No page may instantiate a trainer, a torch module, or a
dataset. If a page needs a new capability, add it to the SDK first,
update `docs/PRD.md` §6, then consume it here.

This keeps the §5 hybrid architecture intact and the trace matrix
(`docs/TRACE.md`) honest: every brief-§ claim resolves through one
SDK method.

## Page → brief-§ binding

W2 shipped pages 01–04 + 09 (skeleton + theory). W3 added the remaining
brief-aligned pages: **05 A2C**, **06 Compare**, **07 Recommend**,
**08 Action Masking**, **10 Discussion** — completing the 01–10 range.

| Page                   | Icon | Title             | Brief §                            | SDK verb wrapped                                       | Notes                                                                                                                 |
| ---------------------- | ---- | ----------------- | ---------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `01_home.py`           | 🏠   | Home              | §7 (overview)                      | static (deliverable checklist; no SDK call)            | W2. Hero, §1.4 architect/AI contract pointer, deliverable checklist scanned from `results/` + `notebooks/`.           |
| `02_data.py`           | 📊   | Data              | §7.2                               | `sdk.prepare_data()` → `LogbookHandle`                 | W2. 28-day masked-uniform heatmap, per-day reward decomposition, §1.5.0 Data Quality Contract surfaced.               |
| `03_lstm.py`           | 🧠   | LSTM World Model  | §7.3                               | ADR-006 exception: `LSTMTrainer` direct (SDK is policy-only) | W2. Live per-epoch loss curve; only page allowed to bypass SDK (documented in ADR-006 + page docstring).         |
| `04_reinforce.py`      | 🎯   | REINFORCE         | §7.4 (eq. 4 + eq. 16)              | `train_reinforce_live()` (callback → `sdk.train_reinforce()`) | W2. Live reward curve + EMA baseline; `mean_last_10` KPI; sidebar exposes hidden width, lr, γ, baseline α.   |
| `05_a2c.py`            | ⚡   | A2C               | §7.5                               | `train_a2c_live()` (callback → `sdk.train_a2c()`)       | **W3.** Synchronous A2C; live actor loss / critic loss / entropy / advantage; writes `last_a2c_history` to state.     |
| `06_compare.py`        | ⚖️   | Compare           | §7.6 / DA4 / TR1                   | `sdk.compare()` → `ComparisonResult`                    | **W3.** REINFORCE vs A2C mean ± 1σ over N seeds × N episodes; seed-count and episodes-per-seed sliders.               |
| `07_recommend.py`      | 💡   | Recommend         | §7.7                               | `sdk.recommend()`                                       | **W3.** 12 channel sliders or `State.initial()`; renders chosen action, π(a\|s) bar, predicted s', expected r decomposition. |
| `08_action_masking.py` | 🛡️   | Action Masking    | §7.6.1 + ADR-004                   | `ActionMaskService` direct (pure inspection, no policy state) | **W3.** Toggle each of 4 mask rules; side-by-side unmasked vs masked softmax; per-rule "why zeroed" annotation.  |
| `09_theory.py`         | 📐   | Theory            | §7.6 / §7.6.1 (12 load-bearing eqs) | static (renders `docs/THEORY.md` via KaTeX)             | W2. Parses `$$..$$` blocks → alternating `st.markdown` + `st.latex`; "maps to `src/`" callouts as `st.info` blocks.   |
| `10_discussion.py`     | 💬   | Discussion        | §7.6 + §11 (honest limitations)    | static (reads `results/` artefacts; no SDK call)        | **W3.** Honest-limitations narrative + ablation summary; cites the four §11 risks and what each chart actually shows. |

## Hard ceilings

- Each page file ≤ 150 LOC (CLAUDE.md §1).
- Live updates use `src/gui/callbacks.py` + `st.empty()` — never patch
  trainers for the GUI (PRD §1 "Live updates").
- Session state keys namespaced `gui.<page>.<key>` via `src/gui/state.py`.
- Tested headlessly with `streamlit.testing.v1.AppTest` at ≥ 85 % cov.
