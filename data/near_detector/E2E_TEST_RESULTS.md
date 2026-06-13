# End-to-End Shot-Detection Test (2026-06-14)

The real product metric: run the full pipeline on a FROZEN game (unseen by
detector AND classifier), with NO ground-truth windows.

## Component metrics (strong in isolation)
- Detector hoop recall 1.000, ball-at-rim 0.825 (held-out test games)
- Classifier 0.953 LOGO (on GT-windowed, DINO-anchored rim crops)
- Event-spotter (72c08cb7, no GT windows): recall 0.900, precision 0.826, 0.33 false/min

## End-to-end (FROZEN e74164e6, t=1680-2880s, 31 shots, no GT windows)
| metric | value |
|---|---|
| Spot recall | 0.871 (27/31) |
| Spot precision | 0.684 (39/57 — ~1.8 events/shot) |
| **End-to-end make/miss acc** | **0.692 (27/39)** |
| make recall | 0.593 (makes called miss) |
| miss recall | 0.917 |

## Diagnosis: TRAIN/SERVE SKEW (not a model-quality problem)
The classifier is CONFIDENTLY WRONG on missed makes (p_make ~0.04, vs ~0.97 on
correct makes) — no middling probabilities. Confident-wrong = the serving crops
are out-of-distribution vs training crops. Causes:
1. Crop geometry: training crops anchored on the DINO median hoop box; serving
   uses the detector's running-median hoop box (different size/offset) -> rim
   sits in a different place in the 160x160 -> OOD.
2. Temporal window: spotter triggers on ball-enters-zone and re-anchors motion
   in a re-decoded window; this lands on a different moment than the training
   anchor (GT end-time +/- motion peak). The drop-through frames may be cut.
3. Spotter over-triggers (~1.8 events/shot): rebounds/passes near rim each
   spawn an event; some land on non-decisive moments.

## Fix (v0.1) — serve-consistent training
1. Rebuild the classifier training set using the SAME detector+spotter pipeline
   that serves it (same hoop-box source, same crop_rect, same anchor). Retrain
   on those crops -> eliminates skew by construction. Highest-leverage fix.
2. Spotter precision: dedup multi-events per shot; trigger on rim-ARRIVAL
   (ball crossing rim plane downward) not just zone-entry; require ball to have
   approached from above.
3. Re-run this same frozen-game test as the gate; target end-to-end >= 0.90.

## Verdict
Components validated; naive integration = 0.69 due to train/serve skew. This is
the expected first-integration result and is diagnosable + fixable. The
end-to-end test (Rohit's call) correctly caught what component metrics hid.
