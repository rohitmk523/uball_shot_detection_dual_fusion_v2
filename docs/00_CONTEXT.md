# 00 — Context: what v1 proved, and the honest ceiling

## The pipeline (unchanged parts)
Two cameras per hoop: **near** (close, overlap-based made/miss) and **far** (down-court, line-crossing geometry). Per side they are temporally matched and **fused**. Side A = FR+NR (right hoop), Side B = FL+NL (left hoop). Detector = YOLOv11n, 2 classes (ball, rim). Made/miss = downstream geometry, **not** the detector.

## Validated numbers (game `c2a354fe`, vs human annotator `plays` GT = 189 shots: 80 made, 109 missed)

| | Detection | Made/miss acc | "Made" precision | "Make" recall |
|---|---|---|---|---|
| V1 far (old) | 68.3% | 86.8% | 84.7% | 86.2% |
| **v16 far (retrained)** | **92.6%** | 85.7% | 79.5% | 89.2% |
| Near alone (v1 near) | 89.9% | 86.5% | 86.3% | 82.9% |
| Far alone (v16) | 92.1% | 81.0% | 75.6% | 80.8% |
| **Fused (near+v16 far)** | **92.6%** | **85.7%** | 79.5% | **89.2%** |

Controlled: V1→v16 changed **only** the far model (same near, same fusion) → the +24 pt detection gain is purely the far retrain.

## What this tells us (the basis for v2)

1. **Detection is largely solved** by the far retrain. Remaining detection gap ≈ 14/189, much of it irreducible near-angle occlusion.
2. **Made/miss is a LOGIC problem.** Near's 23 misjudged shots had detection confidence 0.83 (vs 0.88 correct) and 100% ball-in-rim overlap — the model saw them; the rule failed.
3. **Error is bidirectional.** Branch `fix/shot-geometry-AB` (widen far bottom-gate + loosen near swish gate) was run on AWS and scored vs `plays`: FN unchanged, **precision −6.5 pp** (FP 17→24). One-sided threshold tuning of a flawed rule cannot win — documented negative result.
4. **Complementary angles.** Far = best coverage/detection; near = best made/miss precision; far sees the through-plane instant near is most occluded for. Fusion already nets the best of both — so the target is the **fused** decision, not near-alone.

## The honest ceiling (must be stated to stakeholders)

100% is not a credible target:
- **Annotator label noise**: humans disagree on borderline rim in-and-outs; GT itself has a few-percent ambiguity. You cannot exceed label consistency.
- **Occlusion physics**: from the near angle the decisive instant (ball through hoop plane) is frequently blocked by shooter/defenders/net; from far the ball can be tiny/occluded. Some shots are not determinable from the available pixels at all.
- Realistic objective: **maximize the fused confusion matrix on held-out games** and converge to the practical ceiling (target band in `05_VALIDATION.md`), with transparent reporting when iterations hit diminishing returns. "Drive to the achievable maximum," not "100%."

## Legacy logic to replace (reference, in the v1 fusion repo)
- Near made/miss: `Uball_dual_angle_fusion/Uball_near_angle_shot_detection/shot_detection.py` (decision ~`:690–817`, rim-bounce ~`:538–573`, downward-consistency fallback `:526`).
- Far made/miss: `Uball_dual_angle_fusion/Uball_far_angle_shot_detection/simple_line_intersection_test.py` (decision ~`:356–386`, bottom-cross `:157`, bounced_back_out `:331–348`).
- Fusion resolver: `Uball_dual_angle_fusion/dual_angle_fusion.py` (`resolve_disagreement` ~`:684–777`, weights `:508–513`). Implicated in ~9/17 false positives.

v2 replaces the made/miss + resolver with a learned, interpretable decision over dual-angle trajectory features. The detectors (YOLO) are reused as-is.
