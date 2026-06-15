# Can we port Noah's single-camera method to our near angle? VERIFIED NO (2026-06-15)

Question: Noah recovers 3D make/miss from ONE camera via calibration + ball-size
depth + parabola fit + rim-plane crossing (patent US 12,288,344). Does it port to
our OBLIQUE 30fps near camera? Tested by 4 independent implementations (mine +
3 adversarial re-implementations via workflow) on the same 72 shots (2 games),
all leave-one-game-out honest.

## VERDICT: robust NO — it's a camera-placement + frame-rate play, not an algorithm
Noah's actual features (L-R offset, ball-size depth) cap at **0.555-0.583 honest
LOGO AUC** — statistically indistinguishable from the through-passage baseline
(0.544), far below the ~0.75 useful bar and the **0.93 CNN**. All 4 implementations
converged. The failure is GEOMETRY/OPTICS, not a fitting bug.

## Three verified reasons
1. **Oblique depth ambiguity (the core).** Ball-size-at-crossing depth feature is
   SIGN-INCONSISTENT across games (AUC 0.375 game1 vs 0.798 game2) and INVERTS
   under LOGO (in-sample 0.595/0.670 -> 0.180/0.190). An oblique side view cannot
   cleanly encode front-rim vs back-rim depth -> no portable threshold exists.
2. **The extrapolation advantage never engages.** Verified extrap distance ~0 for
   40/41 fired shots -> crossings only land when a detection already sits at the
   rim plane -> degenerates to the through-passage check (0.544). Noah's key trick
   (extrapolate WITHOUT a crossing detection) doesn't help on our data.
3. **Bbox homography is degenerate.** A symmetric rim bbox is underdetermined for
   perspective; recovered crossings land at radial ~17.5in (make) vs ~19.2in
   (miss), both ~2x the 9in rim radius; inside-rim flag is noise (AUC 0.515).
   Proper calibration needs the rim ELLIPSE (5-DOF fit) or true extrinsics.

## What IS solvable (but doesn't rescue it)
- Fire-rate: robust RANSAC/descent fitting lifts makes-crossing-fire from 35% ->
  61.5% -> 80.8%. But firing more produces no separation.
- One strong feature appeared: min_dist_rimcenter (2D image proximity) AUC 0.813
  — but it's NOT a parabola feature, and it DEGRADES to 0.467 (below chance) when
  detections are decimated to ~15fps. It is frame-rate-HUNGRY: more frames help,
  fewer hurt — which reinforces the 120fps thesis.

## Implication (answers Rohit's question)
- On the CURRENT oblique near rig: keep the learned CNN (0.93). Do NOT port the
  geometric/parabola method — verified dead end at oblique 30fps.
- The Noah advantage is unlocked ONLY by HARDWARE: (a) straight-down / over-rim
  placement so the rim circle + front/back depth become observable (not degenerate),
  (b) 120fps + fast shutter so the descending arc is dense and the crossing is
  cleanly sampled, (c) calibrate via the rim ELLIPSE (5-DOF), not the bbox.
- This is consistent with everything measured this session: the 30fps software
  ceiling is real; the 120fps + overhead-calibrated near camera is the path to
  Noah-class make/miss + arc/depth/L-R metrics.

Scripts: parabola_test.py (+ workflow verify-parabola-port). Patent: US 12,288,344.
