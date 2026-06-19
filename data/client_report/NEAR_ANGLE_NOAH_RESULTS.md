# Near-Angle Focus: Noah-Style Depth Cue — Results

**Date:** 2026-05-25
**Scope:** Per request — focus ONLY on the near angle (NL/NR), 5 fresh games,
out-of-sample. Apply Noah's depth idea to the near cameras, compare new vs old,
and decide whether the near detector needs more training. **No GPU was spent
(all CPU); detection analysis showed retraining isn't warranted.**

## 1. Does the near detector need more training? **No.**
Probe over all 854 fresh shots, each on its play's near camera (LEFT→NL, RIGHT→NR):

| | near | far (far_v16) |
|---|---|---|
| rim detected | **98%** | 100% |
| ball detected (window avg) | 52% | 62% |
| **ball seen AT the rim** | **100% of shots** | 99% |

The near camera detects the ball at the rim in **every single shot (0/854 blind)**.
The lower window-average ball rate is just frames where the ball is mid-court /
out of frame — irrelevant. **Near's weakness is the oblique camera ANGLE
(geometry), not detection.** A `near_v16` retrain would not help → GPU not needed.

## 2. Noah-style depth cue on the near angle
Noah resolves depth with a **known-ball-size scale**. We replicated the monocular
version on the existing near detections:

> at the frame where the ball is horizontally inside the rim **and** at rim level,
> measure **ball_width / rim_width**. Makes vs misses separate: median **0.355 vs
> 0.286** (single-feature AUC ≈ 0.60).

(Note: the naive version — min ratio over the whole approach — has *no* signal;
the discriminative measurement must be taken **at the rim crossing**.)

## 3. New vs old — NEAR-ONLY model (out-of-sample, 5 fresh games)

| Near-only model | Acc | Prec | Recall | AUC | FP | FN |
|---|---|---|---|---|---|---|
| OLD (existing near feats) | 0.876 | 0.829 | 0.893 | 0.956 | 67 | 39 |
| **NEW (+ Noah depth cue)** | **0.902** | 0.855 | 0.926 | **0.969** | 57 | 27 |

**The depth cue genuinely improves the near angle: +2.6 pts accuracy, AUC +0.013,
and BOTH false positives (−10) and false negatives (−12) fall.** Real gain, not a
threshold trade.

## 4. …but it does NOT improve the full fusion

| Full far+near fusion (fresh) | Acc | Prec | Recall | AUC | FP | FN |
|---|---|---|---|---|---|---|
| Production (210 feats) | **0.951** | 0.899 | 0.997 | 0.993 | 41 | 1 |
| + Noah depth cue | 0.948 | 0.892 | 1.000 | 0.994 | 44 | 0 |

Adding the near depth cue to the production model is **flat-to-slightly-worse
(0.951 → 0.948)**. Reason: the **far angle (far_v16) already supplies the depth /
through-rim information**, so the near cue is **redundant in fusion** — it only
helps when the far view is absent (near-alone).

## 4b. Can the near cue *rescue* the fusion's errors? (gated fusion)
Tried 4 ways to inject the near depth cue into the fusion, all tuned on
train/val, evaluated on the 5 fresh games (here the base threshold is
accuracy-optimized, so base = 0.958; production's recall-tuned base is 0.951):

| Strategy | Acc | Prec | Rec | FP | FN |
|---|---|---|---|---|---|
| base fusion (210) | **0.958** | 0.934 | 0.970 | 25 | 11 |
| blend w·fus+(1−w)·near | 0.945 | 0.893 | 0.989 | 43 | 4 |
| gate (near when fusion uncertain) | 0.956 | 0.924 | 0.975 | 29 | 9 |
| flip make→miss if near says "in-front" | 0.936 | 0.940 | 0.907 | 21 | 34 |

**No strategy beats the fusion alone.** Blending, gating, and flipping are all
flat-to-worse. The near cue is genuinely **redundant with the far angle** inside
the fusion — far already resolves the depth those shots need.

(Side note: the fusion can read **0.958** with an accuracy-tuned threshold vs the
shipped **0.951** — but that trades recall down, FN 1→11; we deliberately tune
for near-perfect recall, so this isn't a real accuracy unlock, just a knob.)

## 5. Takeaways
1. **Near detector: leave it.** It already sees the ball+rim at the rim 100% of
   the time; retraining won't move accuracy. (No GPU spend — confirmed by data.)
2. **Noah's depth idea works monocularly — but only meaningfully where there's no
   far camera.** It lifts near-alone 87.6%→90.2%; in the full system the far angle
   already covers it.
3. **The production ceiling stays ~0.951.** The residual depth-illusion errors are
   *not* fixable by better near detection or a monocular near cue inside the
   fusion. The only lever left is **hardware: an overhead rim-axis camera**
   (see [`NOAH_HARDWARE_BLUEPRINT.md`](./NOAH_HARDWARE_BLUEPRINT.md)) — which is
   exactly why Noah mounts above the rim instead of using a side view.

## Reproduce
```bash
python3 pipeline/near_detect_quality.py     # detection probe (Q1)
python3 pipeline/near_noah_compare.py        # near-only OLD vs NEW (Q3)
# fusion check is the inline snippet in the session notes (Q4)
```
Artifacts: `data/near_detect_quality_fresh.parquet`,
`pipeline/near_noah_compare.py` (nd_* depth features).
