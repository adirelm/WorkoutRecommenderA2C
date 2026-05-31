# UX — Nielsen heuristics & quality criteria mapping (V3 §10)

This document closes the V3 §10 warning by making the GUI's UX claims
**falsifiable**: each Nielsen heuristic and each ISO-style quality
criterion is bound to a specific page, a screenshot filename, and the
concrete artefact in the running Streamlit app that supplies the
evidence. Every screenshot referenced below lives in `docs/assets/` and
was regenerated in commit `9925aa0` so the visuals match the current
codebase, not a stale capture.

The 10 GUI pages under review (see `src/gui/pages/` and the table in
[../README.md](../README.md#pages)):

| Page | Screenshot |
|---|---|
| Home | [`docs/assets/gui_home.png`](assets/gui_home.png) |
| Data | [`docs/assets/gui_data.png`](assets/gui_data.png) |
| LSTM World Model | [`docs/assets/gui_lstm.png`](assets/gui_lstm.png) |
| REINFORCE | [`docs/assets/gui_reinforce.png`](assets/gui_reinforce.png) |
| A2C | [`docs/assets/gui_a2c.png`](assets/gui_a2c.png) |
| Compare | [`docs/assets/gui_compare.png`](assets/gui_compare.png) |
| Recommend | [`docs/assets/gui_recommend.png`](assets/gui_recommend.png) |
| Action Masking | [`docs/assets/gui_action_masking.png`](assets/gui_action_masking.png) |
| Theory | [`docs/assets/gui_theory.png`](assets/gui_theory.png) |
| Discussion | [`docs/assets/gui_discussion.png`](assets/gui_discussion.png) |

---

## 5 Quality Criteria

- **Learnability**: 10-page sidebar navigation labeled in plain English; theory page (gui_theory.png) exposes the underlying RL equations alongside the controls. First-time user reaches a training run within 2 clicks.
- **Efficiency**: hyperparameter sliders pre-set to sensible defaults; one-click "Train" button on each algorithm page (gui_reinforce.png, gui_a2c.png).
- **Memorability**: consistent layout across all 10 pages (hero + sidebar params + main canvas); same Bar-Ilan blue (#003D7A) primary; brand wordmark always top-left.
- **Error Prevention**: action_masking page (gui_action_masking.png) visualizes which actions are forbidden BEFORE training, preventing silent illegal-action errors.
- **Satisfaction**: bilingual UX on discussion page (gui_discussion.png) — Hebrew + English side by side honors the course context.

---

## Nielsen's 10 Heuristics (mapped to GUI)

1. **Visibility of system status**: Streamlit progress bars during training (every page with a Train button).
2. **Match with real-world**: terminology mirrors the brief (REINFORCE, A2C, LSTM, action-mask, advantage) — see docs/THEORY.md.
3. **User control and freedom**: every page has a "Reset" affordance via Streamlit's stateless reload.
4. **Consistency and standards**: shared sidebar params; shared chart styling via src/gui/charts.py.
5. **Error prevention**: action_masking page; numerical sliders bounded to physically meaningful ranges.
6. **Recognition over recall**: page names + emoji icons (🧠 LSTM, 🎯 REINFORCE, ⚡ A2C) in sidebar.
7. **Flexibility and efficiency**: keyboard navigation works (Streamlit native); CLI menu (main.py) for power users.
8. **Aesthetic and minimalist design**: gradient hero, no clutter — see gui_home.png.
9. **Help users recognize/diagnose/recover from errors**: Streamlit's automatic exception rendering on each page.
10. **Help and documentation**: every page has a "What this page does" expander pointing to the brief §-id.

---

## How to re-verify

1. Launch the GUI: `uv run streamlit run src/gui/app.py` (or `uv run main.py` → verb 7).
2. Walk the 10 pages in sidebar order and confirm each heuristic above is
   visible in the live app.
3. Compare the live app against `docs/assets/gui_*.png` — they should
   match the current commit. If they drift, re-run the screenshot
   capture step before claiming this doc is fresh.

The Nielsen heuristics list is the canonical 1994 set
(<https://www.nngroup.com/articles/ten-usability-heuristics/>);
the five quality criteria are the standard
Learnability / Efficiency / Memorability / Errors / Satisfaction set
attributed to Nielsen's *Usability Engineering* (1993).
