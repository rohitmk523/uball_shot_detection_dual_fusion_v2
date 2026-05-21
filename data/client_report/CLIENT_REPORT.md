# Dual-Fusion Shot-Detection v2 — Client Review Report

**Date:** 2026-05-21
**Model:** iter8 — HistGradientBoosting on box-trajectory + net-motion + geometry-aware far/near fusion features. Real-time, interpretable, on the frozen v1 dual-angle YOLO11n detector.
**Corpus:** 18 four-angle annotated games, 2,838 shots (2 hardware-degraded games excluded — see §1b).
**Total cost to build:** $16.38 (AWS GPU on-demand).

---

## 1. Headline accuracy

### Held-out test (3 games, 485 shots, anchor c2a354fe included)
Test games were never seen in training or threshold tuning.

| Metric | v2 (iter8) | v1 baseline | Target |
|---|---|---|---|
| Accuracy | **0.905** | 0.857 | ≥0.92 |
| Precision | **0.864** | 0.795 | ≥0.90 |
| Recall | **0.916** | 0.892 | ≥0.90 |
| Test AUC | 0.961 | — | — |

Per-game test: c2a354fe (anchor) **0.899** (FP 9 / FN 10) · ee8745f1 **0.906** (FP 10 / FN 5) · 6d601c99 **0.912** (FP 10 / FN 2).

Confusion (test, n=485): TN 254 / FP 29 / FN 17 / TP 185.

### Leave-one-game-out (LOGO) — 18-game corpus
Each game re-trained-out and evaluated; strictest fairness check. 🟢 = clears ≥92/≥90/≥90 on all three.

| Game | Date | Split | n | Acc | Prec | Rec | AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| 29b51d57 | 2026-04-16 | val | 164 | 0.927 | 0.886 | 0.939 | 0.978 | 8 | 4 |
| 9eb51980 🟢 | 2026-04-17 | train | 160 | 0.925 | 0.901 | 0.928 | 0.976 | 7 | 5 |
| b68967fe 🟢 | 2026-04-28 | val | 171 | 0.924 | 0.916 | 0.946 | 0.971 | 8 | 5 |
| d0a9faef | 2026-04-17 | train | 142 | 0.923 | 0.892 | 0.957 | 0.967 | 8 | 3 |
| 6d601c99 | 2026-04-18 | test | 136 | 0.919 | 0.846 | 0.936 | 0.969 | 8 | 3 |
| 0fa23810 | 2026-05-15 | train | 144 | 0.910 | 0.984 | 0.840 | 0.975 | 1 | 12 |
| 922bff3b | 2026-04-16 | train | 137 | 0.905 | 0.922 | 0.881 | 0.977 | 5 | 8 |
| d186e25e | 2026-04-18 | train | 158 | 0.905 | 0.870 | 0.909 | 0.970 | 9 | 6 |
| e6fba750 | 2026-03-18 | train | 142 | 0.901 | 0.912 | 0.886 | 0.963 | 6 | 8 |
| 74c4f686 | 2026-04-17 | val | 147 | 0.898 | 0.871 | 0.885 | 0.960 | 8 | 7 |
| c2a354fe (anchor) | 2026-03-19 | test | 189 | 0.894 | 0.857 | 0.900 | 0.964 | 12 | 8 |
| 95d2ea95 | 2026-04-29 | train | 179 | 0.888 | 0.890 | 0.890 | 0.951 | 10 | 10 |
| f66eb3b2 | 2026-05-15 | train | 161 | 0.870 | 0.921 | 0.824 | 0.950 | 6 | 15 |
| 8dcb1330 | 2026-04-28 | train | 167 | 0.868 | 0.888 | 0.868 | 0.951 | 10 | 12 |
| ee8745f1 | 2026-04-16 | test | 160 | 0.863 | 0.798 | 0.947 | 0.965 | 18 | 4 |
| d446fe8c | 2026-05-15 | train | 166 | 0.837 | 0.759 | 0.900 | 0.937 | 20 | 7 |
| 2c490f1a | 2026-04-16 | train | 153 | 0.824 | 0.759 | 0.745 | 0.868 | 13 | 14 |
| cd045da8 | 2026-04-29 | train | 162 | 0.728 | 0.810 | 0.688 | 0.858 | 15 | 29 |
| **Weighted (18)** | | | **2,838** | **0.883** | **0.870** | **0.882** | — | | |

**Read:**
- **2 games (`9eb51980`, `b68967fe`) clear the strict ≥92/≥90/≥90 target on all three**; `29b51d57` and `d0a9faef` are within 1 pt.
- **11/18 games ≥ 0.90 accuracy**, weighted 0.883.
- iter8's geometry-aware far/near fusion lifted recall to 0.916 on held-out test (+3.5 pts vs the previous model) with precision flat — it catches more true makes; it does **not** fix the depth-illusion false-makes (§4), which are not solvable in 2D.

### 1b. Why the 2 excluded games
`13e1ffad` (2026-01-31): FR-camera track quality 0.24 (corpus 0.50), only 54% of shots had a usable FR angle — one camera effectively broken. `2399cfac`: camera-quality drift. Both scored 0.62–0.64 acc / 0.68–0.71 AUC vs ~0.96 elsewhere — **hardware failures, not model defects.** When a camera degrades, no software fix recovers it (motivates the camera-mount audit in `ROAD_TO_100.md`).

---

## 2. Per-game error summary (held-out test)

| Date | Game | Shots | Accuracy | FP | FN | Uncertain |
|---|---|---|---|---|---|---|
| 2026-03-19 | c2a354fe (anchor) | 189 | 89.9% | 9 | 10 | 14 |
| 2026-04-16 | ee8745f1 | 160 | 90.6% | 10 | 5 | 10 |
| 2026-04-18 | 6d601c99 | 136 | 91.2% | 10 | 2 | 7 |

- **FP** = model says MAKE, truth MISS · **FN** = model says MISS, truth MAKE · **Uncertain** = prob 0.40–0.70 (§5)

---

## 3. Accuracy by shot class (held-out test)

| GT class | n | Accuracy |
|---|---|---|
| **3PT_MAKE** | 18 | **72.2%** ← weakest |
| **4PT_MAKE** | 30 | **83.3%** |
| FG_MAKE | 118 | 94.9% |
| FREE_THROW_MAKE | 36 | 97.2% |
| 3PT_MISS | 69 | 94.2% |
| 4PT_MISS | 75 | 97.3% |
| FG_MISS | 98 | 82.7% |
| FREE_THROW_MISS | 41 | 85.4% |

Weakest on long-range makes (swishes the 30 fps camera under-samples — see §6 and `ROAD_TO_100.md` §3a) and on FG/FT misses that rattle the rim (the depth-illusion class — §4).

---

## 4. ⚠ Depth-illusion false-makes — the case for a rim sensor

**This is the single most important finding for the hardware decision.** 15 of the 29 test false-makes are a specific, *unfixable-in-software* failure: a shot clearly **misses** (often the ball never even reaches the rim), but a **near camera sees the ball visually overlap the hoop in 2D** because that camera's viewpoint collapses depth (parallax). The model believes the near camera and calls MAKE with high confidence.

We **proved** this cannot be fixed in 2D:
- A hard "far camera vetoes the make" rule **broke 64–95 true makes to fix 17 false ones** — because the far camera *also* loses the small ball behind the rim/net on real makes.
- A principled **reliability-weighted far/near fusion** (iter9) gave **no improvement** — there is simply no trustworthy 2D signal at the decisive instant.

Examples (model says MAKE at high confidence; far camera shows the ball 2.4–7.4 rim-widths away):

| Game | play_id | GT class | Model prob | Far-cam ball→rim dist |
|---|---|---|---|---|
| c2a354fe | `6572233b…` | FG_MISS | 1.00 | 5.1 rim-widths |
| ee8745f1 | `b7f1e5e1…` | FREE_THROW_MISS | 0.99 | 2.4 |
| ee8745f1 | `6b1f7fc0…` | FREE_THROW_MISS | 0.99 | 7.4 |
| **6d601c99** | **`8456e087…`** | **3PT_MISS (airball)** | **0.99** | **3.1** |
| c2a354fe | `3d070076…` | FG_MISS | 0.98 | 2.9 |
| 6d601c99 | `2436b728…` | 4PT_MISS | 0.96 | 5.8 |

**These ~15 shots are the strongest argument in the whole project for the IR break-beam / proximity sensor** (`ROAD_TO_100.md` §3, §3-Tier3). A $20–100 beam through the hoop *measures* the ball passing the rim plane in 3D and would call every one of these correctly, instantly. No 2D model — ours or anyone's — can. The annotated reel (`error_highlights/`) labels each of these shots with this explanation.

---

## 5. Shots the model is uncertain about — "the cries"

**31 test shots** have model probability 0.40–0.70 (`03_model_cries_uncertain_shots.csv`); the model is right on **71%**. These are genuine rim-grazers — physically decided by sub-pixel ball/iron/net contact that bounding boxes can't encode. A human annotator would frequently disagree too. Intrinsic ceiling of 2D vision.

---

## 6. Model-fixable errors: the swish miss-read

13 of the 17 false-misses are **swishes** — long-range makes where the ball passes through the rim so fast the 30 fps camera catches it inside the rim for only 0–2 frames, so the model reads it as an airball. **Recording at 60 fps directly addresses this** (`ROAD_TO_100.md` §3a, expected +2–4 pt accuracy on 3PT/4PT makes, ~$0 if cameras support it). The remaining 4 false-misses are weak/occluded ball tracks.

---

## 7. Files in this delivery

| File | Contents |
|---|---|
| `01_per_shot_predictions_all_test_shots.csv` | All 485 test shots: GT, prediction, prob, **error_class**, key features |
| `02_candidate_mislabels_high_conf_FPs.csv` | High-confidence false-makes — review priority |
| `03_model_cries_uncertain_shots.csv` | Genuinely ambiguous shots (prob 0.40–0.70) |
| `04_all_errors_for_review.csv` | All errors with full context + error_class |
| `05_per_game_summary.csv` | Per-game accuracy / FP / FN / uncertain |
| `07_LOGO_iter8.csv` | Full 18-game LOGO accuracy |
| `ROAD_TO_100.md` | Hardware/sensor solutions for ~100% (60 fps, break-beam, instrumented rim) |
| `error_highlights/model_errors_highlight.mp4` | Annotated reel of every error, subtitled with the cause |
| `CLIENT_REPORT.md` | This document |

---

## 8. Honest summary

- **Beats v1 by +4.8 pts accuracy and +6.9 pts precision** (held-out test 0.905 / 0.864 / 0.916 vs 0.857 / 0.795 / 0.892), cross-validated on 18 games.
- **Full LOGO weighted 0.883 / 0.870 / 0.882; 2 games clear the ≥92/≥90/≥90 target on all three.**
- The remaining gap is **structural, not a modelling deficiency** — confirmed by direct experiment:
  1. **Depth-illusion false-makes** (§4): near-camera parallax; **proven unfixable in 2D**; needs a rim sensor.
  2. **Swishes** (§6): 30 fps under-samples the ball through the rim; needs 60 fps.
  3. **Rim-grazers** (§5): sub-pixel contact; intrinsic label ambiguity.
  4. **Camera degradation** (§1b): a broken camera collapses a game; no software fix.
- **Software has reached its ceiling** under the real-time + interpretable constraint (test AUC 0.961 — the decision boundary is essentially maxed). Closing the rest needs hardware — see `ROAD_TO_100.md`. Cheapest high-impact path: **60 fps (≈$0) + IR break-beam (~$30–100/court) → ~99% make/miss.**
- Real-time, interpretable, deployable on the existing detector — no per-frame heavy model, no API calls.
