# Does the OLD fusion NR logic help our new system? (2026-06-14)

Methodology rule satisfied (v0 built + measured), so re-read the old near-angle
fusion logic: near_fusion_gated.py, near_noah_compare.py, near_detect_quality.py,
NEAR_ANGLE_NOAH_RESULTS.md.

## 1. Noah depth cue (ball_w/rim_w at rim crossing) — TESTED, does NOT help
Old fusion: lifted near-ALONE 0.876->0.902, but was redundant inside the full
fusion (far angle already gave depth). Our new rim-crop CNN already does 0.93 on
found shots -- we've ALREADY surpassed the old near-alone 0.902.

Re-derived the cue on OUR detector boxes (stride-1 caches): it DOES separate
make/miss, even cleaner than old fusion:
  MAKE ratio median 0.414 vs MISS 0.214, single-feature AUC 0.640 (old: 0.60).

But as a post-hoc GATE on the CNN (flip CNN-make->miss if ratio<0.28, threshold
from held-out tuning games): on e74164e6 it made things WORSE 0.731->0.692,
caught 0 of the false-makes (incl. the #04 long-range rim-out), broke 1 true make.
=> AUC 0.64 is too weak to hard-gate a 0.93 CNN; it breaks as many as it fixes.
This MIRRORS the old fusion's own conclusion: gate/blend/flip all flat-to-worse,
the cue is redundant with a strong primary signal. VERDICT: not worth porting.
(Could only help as an aux feature in a RETRAINED CNN, but old experience says
redundant -- low priority.)

## 2. The genuinely useful insight: detection probe -> spotter recall is
##    SAMPLING-limited, not sensor-limited
Old probe (854 blind shots): ball seen AT the rim in 100% of SHOTS (per-shot),
even though per-FRAME ball recall is ~0.5-0.8. For SPOTTING we only need the ball
in >=1 frame per shot -> the ball IS there. Our spotter sweep confirms: stride 3
recall 0.833 -> stride 1 recall 0.946 on tuning games. So our spotter-recall gap
(0.82 at stride 2-3) is DENSE-SAMPLING-recoverable, NOT a sensor ceiling.

PARTIAL REVISION of the detector-v2 "sensor limit" finding: per-FRAME ball recall
IS blur/sensor-limited (v2 confirmed), but per-SHOT SPOTTING recall is recoverable
by sampling every frame at the rim (stride 1). The make/miss DECISION quality is
still the 30fps/120fps story; the SPOT recall has software headroom.

## Bottom line
- Noah cue: real signal, too weak to help our CNN (tested). Skip.
- Dense sampling (stride 1) is the actionable spotter-recall lever the old probe
  points to. Next: confirm stride-1 e2e on frozen games lifts recall toward ~0.92
  while make/miss holds ~0.93.
