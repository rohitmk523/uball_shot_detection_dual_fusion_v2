# 01 — Architecture

## Principle
Keep detection (YOLOv11n, ball+rim) **as-is** — it works. Replace the two hand-coded made/miss rules **and** the hand-coded fusion resolver with **one interpretable model** that consumes **trajectory features from both angles** and is **fit on annotator ground truth**, validated cross-game.

## Pipeline

```
 per side (A=FR+NR, B=FL+NL):
   video → YOLOv11n (ball,rim per frame, both angles)         [REUSE v1 detectors / weights]
        → per-shot window detection (clip the ball-near-rim event)   [REUSE detector event segmentation]
        → TRAJECTORY EXTRACTION  : per-frame (t, ball_xy, conf, rim_bbox) for near AND far   [NEW, v2]
        → FEATURE VECTOR         : geometry features per angle + cross-angle/fusion features [NEW, v2]
        → MADE/MISS MODEL        : interpretable classifier, GT-fit                          [NEW, v2]
        → calibrated confidence  : Platt/isotonic, so the score is a real probability        [NEW, v2]
```

The model **subsumes** both per-angle made/miss logic and `resolve_disagreement`: one decision over a feature vector that includes both angles (and degrades gracefully when an angle is missing/occluded).

## Model choice (interpretable, defensible)
- Start: **gradient-boosted trees (XGBoost/LightGBM, shallow)** or **logistic regression** on engineered features. Both are auditable (feature importances / coefficients) — required for a client-facing accuracy claim.
- **No deep black box.** If a per-shot decision is questioned, we must be able to say *which feature* drove it.
- Output a **calibrated probability**; the made/miss threshold is then chosen on a held-out set to hit the precision/recall target — not hand-picked.

## Graceful degradation
Feature vector must encode angle availability (`near_present`, `far_present`, `n_tracked_points`, `occlusion_ratio`). Single-angle shots are valid inputs; the model learns to lean on whichever angle is reliable per shot type (this is the learned replacement for `resolve_disagreement`).

## What is reused vs new
| Component | v1 source | v2 |
|---|---|---|
| YOLO ball/rim detector + weights | `s3://uball-cv-models/yolov11/v2-prod-far/{near,far}/best.pt` | **reuse unchanged** |
| Shot-event windowing | v1 detectors | reuse (wrap to emit raw track) |
| Per-frame ball/rim track export | — | **new** |
| Trajectory features (both angles) | — | **new** (`03_FEATURES.md`) |
| Made/miss decision | hand rules + `resolve_disagreement` | **new** GT-fit model |
| Confidence | hand-coded multipliers | **new** calibrated probability |

## Non-goals
- Not retraining YOLO here (detection is solved enough; undetected-shot retrain is a separate, later track — see `06`).
- Not changing the temporal matching/offset logic (reuse), only the decision on top.
