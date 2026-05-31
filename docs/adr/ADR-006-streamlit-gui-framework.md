# ADR-006 — Streamlit as the GUI Framework for the Phase-9 Polish Layer

- Status: Accepted
- Date: 2026-05-31
- Decider: Solo developer (architect role per CLAUDE.md §1.4)
- Related: ADR-001 (hybrid architecture), PRD §1.3 (original "no GUI" line, now
  superseded), PLAN Phase 9 (GUI / polish)

## Context

PRD §1.3 originally framed the deliverable as CLI-plus-notebook only, on the
reading that a graded RL artefact need not ship a GUI. Phase-9 work and the
A1 / A2 grading pattern revisited that line: graders rewarded polished UI / UX
disproportionately, and the §7.7 deliverables (LSTM loss curves, REINFORCE
reward graph, A2C training graph, head-to-head comparison) read more
convincingly inside a unified, themed surface than as eight loose PNGs under
`results/`. The CLAUDE.md §1.4 contract treats "add a GUI" as a human-decided
architectural pivot — this ADR is that sign-off.

Four candidate frameworks were considered against the shared Phase-9 contract
(themed multipage app, live training charts, SDK-only business logic, headless
testability):

- **Streamlit ≥ 1.40.** Python-native, multipage out of the box, built-in
  light / dark toggle, trivial matplotlib / plotly embedding (`st.pyplot`,
  `st.plotly_chart`), live updates via `st.empty()` placeholders plus
  iterator callbacks. Headless testing via `streamlit.testing.v1.AppTest`.
- **Tkinter.** Stdlib, no extra dependency, but no native theming, no
  built-in dark mode, manual layout for every chart, and the "watch training"
  demo requires hand-rolled threading. A2 already paid this cost.
- **Gradio.** Quick to ship a single-page demo, but multipage support and
  custom theming are weaker than Streamlit's, and the component model leans
  toward ML-inference demos rather than multi-tab analysis surfaces.
- **Textual / PyQt.** Textual is TUI-only (graders cannot see plotly), PyQt
  ships a desktop window but adds ~60 MB and a C++ dependency that breaks the
  one-command grader-run story.

## Decision

Adopt **Streamlit ≥ 1.40** plus **plotly ≥ 5.20** (interactive charts) and
**streamlit-extras** (optional polish). All GUI code lives under `src/gui/`
as a new layer alongside `data/`, `env/`, `model/`, `services/`, `sdk/`, and
`cli/`. The CLI gains a verb `7 launch-gui` that shells out via
`subprocess.run(["streamlit", "run", "src/gui/app.py"])`.

Concrete shape (all files ≤ 150 LOC per CLAUDE.md §1):

- Theme: Bar-Ilan blue (primary `#003D7A`, accent `#FFCD00`, light bg
  `#F5F7FA`); dark mode via Streamlit's built-in toggle.
- Session state keys are namespaced `gui.<page>.<key>`; `src/gui/state.py`
  exposes a typed accessor so pages never touch `st.session_state` directly.
- Every page module imports only from `src.sdk.sdk` for business logic
  (CLAUDE.md §3) — no direct `src.env` / `src.model` / `src.services` imports.
- All matplotlib charts go through `src/gui/charts.py` factories
  (DPI = 110, tight layout) — never inline `plt` calls.
- Live training charts use `st.empty()` placeholders plus iterator callbacks
  registered through `src/gui/callbacks.py`; trainer code is **not** modified
  (observer pattern preserves the SDK contract).
- Tests use `streamlit.testing.v1.AppTest` (Streamlit's headless test runner).
- Page docstrings open with the brief § they cover, e.g.
  `"""§7.4 REINFORCE training page."""`.

## Rationale

Streamlit is the only candidate that satisfies all four contract clauses
without writing custom code: themed multipage, live charts, headless tests,
and SDK-only business logic. The trade-off is one extra dependency (~25 MB)
and the absence of a native desktop window — both acceptable, because the
grader runs `uv run main.py 7 launch-gui` and gets a browser tab with the
same SDK underneath. The decision also unlocks plotly's interactive
comparison charts for §7.7, which is a strictly stronger artefact than the
static PNGs the SDK already writes — the PNGs remain the single source of
truth, the GUI just re-renders them interactively.

## Consequences

**Positive.**

- §7.7 deliverables ship as a navigable multipage app rather than loose
  PNGs, addressing the UI / UX criterion that A1 / A2 graders weighted.
- The observer-pattern callback layer means live training charts cost zero
  changes to `src/training/` — the SDK contract from ADR-001 holds.
- `streamlit.testing.v1.AppTest` keeps the 85 % coverage gate honest for the
  new layer without spinning a real browser.

**Negative.**

- Adds ~25 MB to the dependency closure (streamlit + plotly +
  streamlit-extras). Mitigated by the one-command install path.
- No native desktop window — graders need a browser. Acceptable per the
  grader-run story.
- Ten new page modules expand the file count; the 150-LOC ceiling forces
  clean splits, so navigation cost stays bounded.

## Follow-ups

- ADR-001 is cross-linked: `src/gui/` is the tenth layer of the hybrid
  architecture and inherits the SDK-only rule verbatim.
- PRD §1.3 is amended in the same commit as this ADR to record the pivot
  from "no GUI" to "Streamlit GUI as Phase-9 polish".
