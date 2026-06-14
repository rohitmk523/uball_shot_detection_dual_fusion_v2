# Detector v2 (motion-blur augmentation) — NEGATIVE RESULT (2026-06-14)

Hypothesis: ball-at-rim recall (0.825) is capped by 30fps motion blur; fine-tune
v1 on motion-blur-augmented ball frames to lift it. Built whole-frame motion-blur
copies of all 2359 ball-containing train frames (6745 total), warm-started from
v1 best.pt, 45 epochs @1280 on AWS GPU.

## Result: blur aug did NOT lift ball-at-rim recall
| metric | v1 | v2 (blur-aug) |
|---|---|---|
| HOOP recall / prec | 1.000 / 0.998 | 1.000 / 1.000 |
| BALL recall | 0.786 | 0.776 |
| BALL precision | 0.885 | **0.939** |
| **BALL @ RIM MOMENT** | **0.825** | **0.799** (worse) |
val mAP50 ~0.925 (≈ v1 0.919). Augmentation confirmed ran (6745 train imgs).

## Why it failed + what it means
- Whole-frame blur traded RECALL for PRECISION (fewer false balls, but no better
  at detecting the blurred ball at the rim). Likely because whole-frame blur is
  unrealistic (real ball blur is localized to the fast ball, not the scene) and
  pushed the operating point toward precision.
- Combined with the serving sweep (conf/imgsz/TTA gave only +1-3pt rim recall at
  a precision cost), this is strong evidence that **ball-at-rim recall ~0.80-0.82
  is a 30fps SENSOR LIMIT, not a model-tuning gap.** Motion blur at 30fps is the
  fundamental enemy (as NEAR_ANGLE_PLAN predicted), and it resists software fixes.

## Strategic implication (important)
This EMPIRICALLY VALIDATES the 120fps-camera investment: the measured bottleneck
(spotter recall, capped by ball-at-rim detection) is exactly the blur problem
that 120fps solves (4x temporal samples = catch the fast ball passage). We have
now tried and falsified the 30fps software fixes for it.

## Disposition
- KEEP v1 as the recall-optimal detector (near_det_v1_best.pt).
- v2 is precision-optimal (ball prec 0.939, hoop 1.0/1.0) — could reduce false
  spotter EVENTS; worth testing if false-event rate becomes the priority.
- Remaining 30fps SOFTWARE lever for spotter recall: net-motion trigger (detect
  net deformation in the rim region) to catch shots where the ball was never
  detected — decouples spotting from ball detection. Uncertain payoff.
- make/miss on found shots is already ~0.93 (near the 0.925 classifier ceiling).
