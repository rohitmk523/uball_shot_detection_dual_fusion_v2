# Fusion re-test with the NEW near CNN (2026-06-15)

Joined the far/fusion model's out-of-fold predictions with the NEW near rim-crop
CNN's LOGO predictions, per shot (1357 common shots, 9 games, labels agree 100%).
Honest: simple blends are param-free; learned stack uses leave-one-game-out.

## Result: fusing the new near model BEATS far-alone (both error types drop)
| model | acc | AUC | FP | FN |
|---|---|---|---|---|
| FAR / current fusion | 0.9609 | 0.9870 | 34 | 19 |
| NEW near CNN (alone) | 0.9492 | 0.9698 | 34 | 35 |
| **mean blend** | **0.9727** | **0.9930** | **24** | **13** |
| max-confidence | 0.9727 | — | 24 | 13 |
| **LOGO logistic stack (honest)** | **0.9698** | **0.9931** | 23 | 18 |
| geometric-mean (precision-lean) | 0.9624 | 0.9917 | **13** | 38 |

Far 0.961 -> fused ~0.97 (+1pt), AUC 0.987 -> 0.993, **FP 34->24 and FN 19->13
BOTH fall** = complementary errors, not a threshold trade. The honest LOGO stack
(no tuning) still gives 0.970.

## Why this REVERSES the old "near is redundant in fusion" finding
- OLD near = weak geometric features (0.876) -> redundant with far (far already had
  the depth/trajectory info those features encoded).
- NEW near = a strong, INDEPENDENT learned signal (0.949, rim-crop image CNN, a
  different modality than far's trajectory geometry). Two strong independent models
  make DIFFERENT mistakes, so fusing them cancels errors (FP and FN both drop).
- The geometric-mean rule cuts FP to 13 (precision 0.978) if precision is the
  priority; mean-blend is the best balanced.

## Implication
- The new near model RE-OPENS fusion as the max-accuracy path on the existing
  4-camera setup: ~0.97 (vs old 0.951), honestly ~0.970 via the LOGO stack.
- This is the "best accuracy on current hardware" lane; the standalone-near +
  overhead-120fps camera remains the "low-cost / near-perfect" lane. Both valid.
- Next: re-fit the full fusion with the near CNN as a feature (not just blend),
  add far-angle ARC detection (the far side view is ideal for entry-angle), and
  the Noah-style rim map (needs depth/L-R -> approximate from triangulation now,
  accurate from the overhead camera later).
