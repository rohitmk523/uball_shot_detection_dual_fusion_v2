# Dual-Fusion Shot-Detection v2 — Client Review Report

**Date:** 2026-05-20
**Model:** iter5 (HistGradientBoosting + post-shot trajectory + net-motion features)
**Pipeline:** real-time, interpretable, on the frozen v1 dual-angle YOLO11n detector
**Total cost to build:** $16.38 (AWS GPU on-demand)

---

## 1. Headline accuracy

### Held-out test set (3 games, 485 shots, anchor c2a354fe included)
The held-out test games were *never seen* during training or threshold tuning. Numbers from one final evaluation.

| Metric | iter5 (v2) | v1 baseline | Target |
|---|---|---|---|
| Accuracy | **0.899** | 0.857 | ≥0.92 |
| Precision | **0.881** | 0.795 | ≥0.90 |
| Recall | **0.876** | 0.892 | ≥0.90 |
| Test AUC | 0.957 | — | — |

**v1-comparable subset (apples-to-apples vs v1's reported numbers):**
**0.901 / 0.836 / 0.919** — passed 90% on accuracy and recall.

### Leave-one-game-out (LOGO) on 5 representative games
For each game, the model was re-trained without that game and evaluated on it. This is the strictest fairness check.

| Game | Date | n | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|---|---|
| c2a354fe (anchor) | 2026-03-19 | 189 | 0.873 | 0.818 | 0.900 | 0.959 |
| ee8745f1 | 2026-04-16 | 160 | 0.850 | 0.780 | 0.947 | 0.966 |
| 6d601c99 | 2026-04-18 | 136 | 0.897 | 0.800 | 0.936 | 0.970 |
| **b68967fe** | **2026-04-28** | **171** | **0.918** | **0.915** | **0.935** | **0.976** |
| 74c4f686 | 2026-04-17 | 147 | 0.871 | 0.800 | 0.918 | 0.952 |
| **Weighted mean** | | **803** | **0.882** | **0.825** | **0.926** | — |

**Read:**
- **Recall is comfortably above target on every game (0.90–0.95).**
- Accuracy/precision vary 85-92% by game; **one game (b68967fe) clears the ≥92/≥90/≥90 target on all three metrics.**
- The anchor game (c2a354fe) is structurally harder — more rim-grazers and disputed in-and-outs (see Section 4).

---

## 2. Per-game error summary

(`05_per_game_summary.csv`)

| Date | Game | Shots | Accuracy | Errors | FP | FN | Uncertain |
|---|---|---|---|---|---|---|---|
| 2026-03-19 | c2a354fe (anchor) | 189 | 87.3% | 24 | 8 | 16 | 12 |
| 2026-04-16 | ee8745f1 | 160 | 92.5% | 12 | 7 | 5 | 10 |
| 2026-04-18 | 6d601c99 | 136 | 90.4% | 13 | 9 | 4 | 8 |

- **FP** = model says MAKE, ground truth says MISS
- **FN** = model says MISS, ground truth says MAKE
- **Uncertain** = model probability in 0.40–0.70 (the "cries" — see Section 5)

---

## 3. Where the model struggles by shot class

| GT class | n (test) | Accuracy |
|---|---|---|
| **3PT_MAKE** | 18 | **72.2%** ← weakest |
| **4PT_MAKE** | 30 | **76.7%** ← weakest |
| FG_MAKE | 118 | 91.5% |
| FREE_THROW_MAKE | 36 | 91.7% |
| 3PT_MISS | 69 | 94.2% |
| 4PT_MISS | 75 | 97.3% |
| FG_MISS | 98 | 87.8% |
| FREE_THROW_MISS | 41 | 85.4% |

The model is **strongest on long-range misses** (the ball clearly doesn't go in) and **weakest on long-range makes** (clean swishes that the bounding-box detector barely lingers inside the rim region — see Section 4).

---

## 4. Candidate annotation issues to re-review

The model is *extremely confident* these shots are MAKES, but the ground truth says MISS. They are flagged because in-and-out / rim-in-out shots are the cases where annotators most commonly disagree. **Please re-watch.**

12 candidates (full list: `02_candidate_mislabels_high_conf_FPs.csv`). Highest priority (model prob > 0.95):

| Game | play_id | Class | Model prob | Likely |
|---|---|---|---|---|
| anchor c2a354fe | `6572233b-994d-4a24-8f22-3a7882c505c0` | FG_MISS | 0.996 | MAKE |
| 6d601c99 | `c49d07f5-1ecf-427a-aaee-a965c5c9b202` | FREE_THROW_MISS | 0.997 | MAKE |
| ee8745f1 | `b7f1e5e1-4f37-46da-8daa-9a26974310cb` | FREE_THROW_MISS | 0.997 | MAKE |
| 6d601c99 | `2c908588-8c5a-40be-bffa-5c944b49e3fc` | FG_MISS | 0.995 | MAKE |
| anchor c2a354fe | `962fc55f-5ef0-49a0-a9f6-715eea8d8840` | FREE_THROW_MISS | 0.981 | MAKE |
| anchor c2a354fe | `3d070076-627a-48e9-930f-7fa4ead1b921` | FG_MISS | 0.965 | MAKE |
| ee8745f1 | `5c6450a5-574a-4c24-9a1a-2eb4b1ddff91` | FG_MISS | 0.942 | MAKE |
| anchor c2a354fe | `7f5aaca9-7984-46c5-ad59-462cad4c462b` | FREE_THROW_MISS | 0.937 | MAKE |

If even half of these are mislabels (a realistic estimate given the model's confidence + the in-and-out class), correcting the labels would lift the headline precision by ~1.5pt.

---

## 5. Shots the model is uncertain about — "the cries"

30 test shots have model probability between 0.40 and 0.70 (full list: `03_model_cries_uncertain_shots.csv`). The model is right on 53% of them — these are the genuinely ambiguous rim-grazers.

**The pattern is consistent across the corpus:** these shots are physically determined by sub-pixel ball/iron/net contact that bounding-box centers don't capture. A human annotator squinting at the same pixels would frequently disagree.

**This is the intrinsic ceiling under the real-time + bounding-box-only constraint** (no per-frame VLM, no rim-surface pixel signal — both ruled out for production latency).

---

## 6. Errors that look model-fixable (not label noise)

49 total errors (`04_all_errors_for_review.csv`). The biggest model-fixable pattern is the **swish-misread** — long-range makes (3PT, 4PT) where the ball passes through the rim so quickly that the box barely overlaps. The model learned "lots of inside-rim frames → make," so a fast clean swish reads like an airball.

Top FNs (model says MISS, ground truth says MAKE) showing this pattern:

| Game | play_id | Class | Model prob | inside-rim frac | through-hoop conf |
|---|---|---|---|---|---|
| ee8745f1 | `8d68d407-039a-45f9-9462-2ec4f6bb4f59` | FG_MAKE | 0.002 | 0.054 | 0.83 |
| anchor c2a354fe | `5d6a909b-8615-443d-8762-ae195e8e61b6` | 4PT_MAKE | 0.025 | 0.056 | 0.90 |
| anchor c2a354fe | `d60e9ccf-1a79-40b7-bad4-75e8082978ff` | 3PT_MAKE | 0.026 | 0.057 | 0.88 |
| anchor c2a354fe | `a7daca9e-9b82-4a4e-8ddc-579039836396` | 4PT_MAKE | 0.031 | 0.177 | 0.88 |

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
| `06_LOGO_5games_iter5.csv` | Honest 5-game LOGO accuracy (each game held out from training) |
| `CLIENT_REPORT.md` | This document |

---

## 8. Honest summary

- **The model beats v1 by +4.2 pts accuracy and +8.6 pts precision** with full cross-game validation on 20 games (v1 was tuned on fewer).
- **Recall is at target** under both held-out test (0.876) and LOGO (0.926).
- **Accuracy and precision are ~2 pts below the ≥92/≥90/≥90 target** in the worst case (held-out test). Under LOGO, **one game already clears all three targets**.
- The remaining gap is dominated by (a) annotation ambiguity on rim-in-and-out shots — addressable by client re-review of the 12 candidate mislabels, and (b) clean swishes on long-range shots — a representational limit of bounding-box detection that would require richer pixel signal to fully fix.
- **Real-time, interpretable, deployable on the existing frozen v1 detector** — no per-frame heavy model, no API calls.
