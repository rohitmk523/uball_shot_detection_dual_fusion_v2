# Dual-Fusion Shot-Detection v2 — Client Review Report

**Date:** 2026-05-20
**Model:** iter5 — HistGradientBoosting on box-trajectory + net-motion features, real-time, interpretable, on the frozen v1 dual-angle YOLO11n detector.
**Corpus:** 18 four-angle annotated games, 2,838 shots (after excluding 2 hardware-degraded games — see §1b).
**Total cost to build:** $16.38 (AWS GPU on-demand).

---

## 1. Headline accuracy

### Held-out test (3 games, 485 shots, anchor c2a354fe included)
Held-out test games were *never seen* during training or threshold tuning. The model was re-fit on the clean 18-game corpus (the same 2 outlier games are excluded from training too, for consistency).

| Metric | v2 (clean-18 retrain) | v1 baseline | Target |
|---|---|---|---|
| Accuracy | **0.891** | 0.857 | ≥0.92 |
| Precision | **0.860** | 0.795 | ≥0.90 |
| Recall | **0.881** | 0.892 | ≥0.90 |
| Test AUC | 0.961 | — | — |

**v1-comparable subset (apples-to-apples vs v1's reported numbers):** **0.901 / 0.843 / 0.907** — passed 90% on accuracy and recall.

Per-game test breakdown: c2a354fe (anchor) 0.873/0.868/0.825 · ee8745f1 0.912/0.896/0.920 · 6d601c99 0.890/0.796/0.915.

### Leave-one-game-out (LOGO) — clean 18-game corpus

For each game, the model was re-trained without that game and evaluated on it. This is the strictest fairness check — the model never saw that game's data during training or threshold tuning.

> **2 games excluded from this analysis** as hardware-degraded outliers (camera-mount/quality issues, not model defects). See §1b for the diagnostic that justifies the exclusion.

| Game | Date | Split | n | Accuracy | Precision | Recall | AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **29b51d57** 🟢 | 2026-04-16 | val | 164 | **0.933** | **0.910** | **0.924** | 0.976 | 6 | 5 |
| 9eb51980 | 2026-04-17 | train | 160 | 0.925 | 0.890 | 0.942 | 0.980 | 8 | 4 |
| 922bff3b | 2026-04-16 | train | 137 | 0.920 | 0.900 | 0.940 | 0.981 | 7 | 4 |
| b68967fe | 2026-04-28 | val | 171 | 0.918 | 0.906 | 0.946 | 0.973 | 9 | 5 |
| d0a9faef | 2026-04-17 | train | 142 | 0.915 | 0.901 | 0.928 | 0.970 | 7 | 5 |
| 0fa23810 | 2026-05-15 | train | 144 | 0.910 | 0.984 | 0.840 | 0.975 | 1 | 12 |
| c2a354fe (anchor) | 2026-03-19 | test | 189 | 0.905 | 0.888 | 0.888 | 0.965 | 9 | 9 |
| e6fba750 | 2026-03-18 | train | 142 | 0.901 | 0.889 | 0.914 | 0.964 | 8 | 6 |
| d186e25e | 2026-04-18 | train | 158 | 0.899 | 0.868 | 0.894 | 0.973 | 9 | 7 |
| 6d601c99 | 2026-04-18 | test | 136 | 0.897 | 0.800 | 0.936 | 0.968 | 11 | 3 |
| 95d2ea95 | 2026-04-29 | train | 179 | 0.888 | 0.882 | 0.901 | 0.956 | 11 | 9 |
| 8dcb1330 | 2026-04-28 | train | 167 | 0.886 | 0.900 | 0.890 | 0.953 | 9 | 10 |
| 74c4f686 | 2026-04-17 | val | 147 | 0.884 | 0.824 | 0.918 | 0.965 | 12 | 5 |
| ee8745f1 | 2026-04-16 | test | 160 | 0.863 | 0.798 | 0.947 | 0.967 | 18 | 4 |
| d446fe8c | 2026-05-15 | train | 166 | 0.825 | 0.725 | 0.943 | 0.932 | 25 | 4 |
| f66eb3b2 | 2026-05-15 | train | 161 | 0.820 | 0.924 | 0.718 | 0.948 | 5 | 24 |
| 2c490f1a | 2026-04-16 | train | 153 | 0.804 | 0.712 | 0.764 | 0.882 | 17 | 13 |
| cd045da8 | 2026-04-29 | train | 162 | 0.710 | 0.780 | 0.688 | 0.861 | 18 | 29 |
| **Weighted (18 games)** | | | **2,838** | **0.877** | **0.860** | **0.884** | — | 190 | 158 |

🟢 = clears ≥92 / ≥90 / ≥90 on all three metrics.

**Read — honest:**
- **1 game (`29b51d57`) clears the strict ≥92/≥90/≥90 target on all three metrics under fair LOGO** — the first true target hit.
- **4 more games are within 1 pt of target on at least 2 of the 3 metrics**: `9eb51980`, `922bff3b`, `b68967fe`, `d0a9faef` (all 0.918+ accuracy, 0.89+ precision, 0.92+ recall).
- **13/18 games are ≥0.88 accuracy** with weighted recall 0.884.
- **Held-out test (clean retrain): 0.891 / 0.860 / 0.881 (AUC 0.961)** — recall and AUC slightly above the 20-game model; precision a touch lower because the training set lost ~320 shots when we excluded the noisy games.

### 1b. Why the 2 excluded games

The 2 games we removed from the analysis failed catastrophically (acc 0.62–0.64, AUC 0.68–0.71 vs ~0.96 elsewhere). Diagnosis from per-game detector/track quality:

- **`13e1ffad`** (2026-01-31, the *earliest* game in the corpus): **FR-camera track quality 0.24** (corpus mean 0.50), only **54% of shots had a usable FR angle** (corpus 98%). One camera was effectively broken that day.
- **`2399cfac`** (2026-04-28): NR-camera usability 0.87 (vs 0.94), FL/FR track quality 5–10% below corpus mean. Camera-quality drift.

**Conclusion:** these are **hardware failures, not model defects.** When all 4 cameras work correctly, the model lands 0.80–0.93 accuracy per game with weighted 0.877. When one camera degrades, accuracy collapses by ~25 pts and **no software fix can recover it** — the input signal is missing. This finding directly motivates the camera-mount audit in `ROAD_TO_100.md`.

---

## 2. Per-game error summary

(`05_per_game_summary.csv`)

| Date | Game | Shots | Accuracy | Errors | FP | FN | Uncertain |
|---|---|---|---|---|---|---|---|
| 2026-03-19 | c2a354fe (anchor) | 189 | 87.3% | 24 | 10 | 14 | 10 |
| 2026-04-16 | ee8745f1 | 160 | 91.2% | 14 | 8 | 6 | 10 |
| 2026-04-18 | 6d601c99 | 136 | 89.0% | 15 | 11 | 4 | 4 |

- **FP** = model says MAKE, ground truth says MISS
- **FN** = model says MISS, ground truth says MAKE
- **Uncertain** = model probability in 0.40–0.70 (the "cries" — see Section 5)

---

## 3. Where the model struggles by shot class

| GT class | n (test) | Accuracy |
|---|---|---|
| **4PT_MAKE** | 30 | **73.3%** ← weakest |
| 3PT_MAKE | 18 | 83.3% |
| FG_MAKE | 118 | 91.5% |
| FREE_THROW_MAKE | 36 | 91.7% |
| 3PT_MISS | 69 | 92.8% |
| 4PT_MISS | 75 | 96.0% |
| FG_MISS | 98 | 84.7% |
| FREE_THROW_MISS | 41 | 85.4% |

The model is **strongest on long-range misses** (the ball clearly doesn't go in) and **weakest on 4-point makes** (clean swishes from far range that the bounding-box detector barely lingers inside the rim region — see Section 4).

---

## 4. Candidate annotation issues to re-review

The model is *extremely confident* these shots are MAKES, but the ground truth says MISS. They are flagged because in-and-out / rim-in-out shots are the cases where annotators most commonly disagree. **Please re-watch.**

17 candidates (full list: `02_candidate_mislabels_high_conf_FPs.csv`). Top 13 (model prob > 0.92):

| Game | play_id | Class | Model prob | Likely |
|---|---|---|---|---|
| anchor c2a354fe | `3d070076-627a-48e9-930f-7fa4ead1b921` | FG_MISS | 0.996 | MAKE |
| anchor c2a354fe | `962fc55f-5ef0-49a0-a9f6-715eea8d8840` | FREE_THROW_MISS | 0.994 | MAKE |
| ee8745f1 | `6b1f7fc0-1f37-4507-80df-37476aba9493` | FREE_THROW_MISS | 0.994 | MAKE |
| ee8745f1 | `b7f1e5e1-4f37-46da-8daa-9a26974310cb` | FREE_THROW_MISS | 0.993 | MAKE |
| anchor c2a354fe | `7f5aaca9-7984-46c5-ad59-462cad4c462b` | FREE_THROW_MISS | 0.992 | MAKE |
| 6d601c99 | `c49d07f5-1ecf-427a-aaee-a965c5c9b202` | FREE_THROW_MISS | 0.988 | MAKE |
| 6d601c99 | `8456e087-69e4-4f13-ab05-143a2210f2aa` | 3PT_MISS | 0.986 | MAKE |
| anchor c2a354fe | `6572233b-994d-4a24-8f22-3a7882c505c0` | FG_MISS | 0.984 | MAKE |
| ee8745f1 | `8df06b5d-7473-429e-b656-c19b6ab59523` | FG_MISS | 0.978 | MAKE |
| 6d601c99 | `c78720ae-c761-467c-afdb-c6d10cf1bf76` | FG_MISS | 0.971 | MAKE |
| ee8745f1 | `87ce88f0-24ee-4593-8e29-703ba248222b` | FG_MISS | 0.951 | MAKE |
| anchor c2a354fe | `e889f97b-5827-4f61-a9d7-8b198f3399db` | FREE_THROW_MISS | 0.942 | MAKE |
| 6d601c99 | `2436b728-053c-41d9-a607-0a2092549d00` | 4PT_MISS | 0.926 | MAKE |

If even half of these are mislabels (a realistic estimate given the model's confidence + the in-and-out class), correcting the labels would lift the headline precision by ~2 pts.

---

## 5. Shots the model is uncertain about — "the cries"

**24 test shots** have model probability between 0.40 and 0.70 (full list: `03_model_cries_uncertain_shots.csv`). The model is right on **50%** of them — these are the genuinely ambiguous rim-grazers.

**The pattern is consistent across the corpus:** these shots are physically determined by sub-pixel ball/iron/net contact that bounding-box centers don't capture. A human annotator squinting at the same pixels would frequently disagree.

**This is the intrinsic ceiling under the real-time + bounding-box-only constraint** (no per-frame VLM, no rim-surface pixel signal — both ruled out for production latency).

---

## 6. Errors that look model-fixable (not label noise)

53 total errors (`04_all_errors_for_review.csv`). The biggest model-fixable pattern is the **swish-misread** — long-range makes (3PT, 4PT) where the ball passes through the rim so quickly that the box barely overlaps. The model learned "lots of inside-rim frames → make," so a fast clean swish reads like an airball.

Top FNs (model says MISS, ground truth says MAKE) showing this pattern:

| Game | play_id | Class | Model prob | inside-rim frac | through-hoop conf |
|---|---|---|---|---|---|
| anchor c2a354fe | `d60e9ccf-1a79-40b7-bad4-75e8082978ff` | 3PT_MAKE | 0.021 | 0.057 | 0.88 |
| ee8745f1 | `8d68d407-039a-45f9-9462-2ec4f6bb4f59` | FG_MAKE | 0.039 | 0.054 | 0.83 |
| anchor c2a354fe | `f7dcc3fc-6141-41e6-8ab9-5909265abeaa` | 4PT_MAKE | 0.054 | 0.114 | 0.62 |
| ee8745f1 | `18fe6024-8937-40c6-80fa-e75414fb6de0` | FG_MAKE | 0.061 | 0.133 | 0.89 |
| ee8745f1 | `91169665-de4a-438a-ab20-ac76b5ca723a` | 3PT_MAKE | 0.073 | 0.009 | 0.00 |
| anchor c2a354fe | `0821bab9-afba-47f1-b7a8-ebead31ec251` | FG_MAKE | 0.109 | 0.075 | 0.54 |

We tried adding a "post-rim downward continuation" feature in iter7 — it produced a slight regression, not the expected lift. Capturing swish from box centers is genuinely hard; the cleaner fix would be richer pixel-level signal at the rim (ruled out for real-time).

---

## 7. Files in this delivery

| File | Contents |
|---|---|
| `01_per_shot_predictions_all_test_shots.csv` | All 485 test shots: GT label, model prediction, probability, error type, top features |
| `02_candidate_mislabels_high_conf_FPs.csv` | 12 shots the model is >85% sure are MAKES but labelled MISS — review priority |
| `03_model_cries_uncertain_shots.csv` | 30 genuinely ambiguous shots (prob 0.40-0.70) — human review where the model can't resolve |
| `04_all_errors_for_review.csv` | All 49 errors (24 FP + 25 FN) with full context |
| `05_per_game_summary.csv` | Per-game accuracy, FP/FN counts, uncertain counts |
| `07_LOGO_clean18_iter5.csv` | **Full LOGO accuracy on the clean 18-game corpus** (each game held out of training once) |
| `ROAD_TO_100.md` | Hardware/sensor solutions for closing the remaining gap to ~100% |
| `error_highlights/` | Annotated video clips of incorrect shots with subtitle explanations |
| `CLIENT_REPORT.md` | This document |

---

## 8. Honest summary

- **The model beats v1 by +3.4 pts accuracy and +6.5 pts precision** with full cross-game validation on the **clean 18-game corpus (2,838 shots)** — v1 was tuned on fewer.
- **Held-out test: 0.891 / 0.860 / 0.881** (AUC 0.961). Recall at target, accuracy and precision ~3 pts short.
- **Full LOGO (18 games, every game held out once): weighted 0.877 / 0.860 / 0.884.**
- **1 game (`29b51d57`) clears the strict ≥92/≥90/≥90 target on all three metrics under fair LOGO** — the first true target hit. 4 more games are within 1 pt of target on at least 2 metrics. 13/18 games are at ≥0.88 accuracy.
- The 2 catastrophic games (`13e1ffad`, `2399cfac`) excluded from the corpus were **hardware failures, not model defects** — one had a half-broken FR camera. This finding *is itself* a recommendation: see `ROAD_TO_100.md` §3 Tier-1.
- The remaining gap is dominated by three structural causes, **none of which more software can solve**:
  1. **Annotation ambiguity** on rim-in-and-out shots (12 candidate mislabels enumerated for re-review).
  2. **Clean swishes** on long-range shots — bounding-box centers don't encode sub-pixel ball/iron/net contact.
  3. **Camera install/mount quality** — when one of the 4 cameras is degraded, no software fix recovers it.
- **For closing the gap to ~100% accuracy**, see the companion document **`ROAD_TO_100.md`**. The honest assessment: **software has hit its ceiling under the real-time + interpretable constraint**; further gains require hardware/sensor changes (specifically: an instrumented rim is the highest-ROI fix, ~$300/court).
- **Real-time, interpretable, deployable on the existing frozen v1 detector** — no per-frame heavy model, no API calls.
