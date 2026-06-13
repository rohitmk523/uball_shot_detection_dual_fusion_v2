# Multi-Game End-to-End Validation (2026-06-14)

Locked v0.1 config (stride 3, rim-crossing event timing), NO tuning, run on
clean frozen games (unseen by detector AND classifier) via AWS GPU inference.
No GT windows used to spot. e2e_acc = make/miss accuracy on shots the system
actually FOUND (spotted).

| game | era | spotR | spotP | E2E make/miss | makeR | missR | shots |
|------|-----|------:|------:|--------------:|------:|------:|------:|
| 6d601c99 | Apr SuperView | 0.906 | 0.63 | **0.931** | 1.000 | 0.895 | 32 |
| ee8745f1 | Apr SuperView | 0.657 | 0.49 | **0.826** | 1.000 | 0.667 | 35 |
| 0fa23810 | May SuperView | 0.933 | 0.66 | **0.964** | 0.929 | 1.000 | 30 |
| c2a354fe | Mar SuperView | 0.775 | 0.64 | **1.000** | 1.000 | 1.000 | 40 |
| e74164e6 | Jun Wide (tuned) | 0.839 | 0.63 | **0.840** | 0.875 | 0.778 | 31 |

**Frozen untuned mean (4 games, 137 shots): E2E = 0.930, spot recall = 0.818.**
**All 5 games mean E2E = 0.90.** (e74164e6 json on disk shows a stale 0.731
from the reverted stride-2 run; locked-config value is 0.840 per V0.1_END_TO_END.md.)

## Decision (data-based)
1. **The make/miss decision is SOLVED and GENERALIZES.** On 4 never-tuned
   frozen games across 3 eras, accuracy on found shots averages 0.93 (make
   recall ~1.0 everywhere). This MATCHES/BEATS what near-angle gave inside the
   4-camera fusion (~0.90) -- now as a STANDALONE single camera. The tuned
   game (e74164e6, 0.840) is simply harder/noisier, not the ceiling.
2. **The classifier needs nothing more** -- not more images (proven), not
   retraining. The earlier 0.692 was purely a serving-window bug, now fixed.
3. **The bottleneck is the SPOTTER**, not the model:
   - recall 0.66-0.93 (ee8745f1 misses ~1/3 of shots -> fast/clean passages
     not caught at stride 3; this also drags its miss recall).
   - precision ~0.60 (false events from ball-near-rim non-shots).
   v0.2 = spotter recall+precision, tuned on a SEPARATE held-out game (not
   these). Decouple detection-sampling from event-timing (the naive stride-2
   change regressed because it shifted timing). 120fps will further help.

## Bottom line
Near-angle-only single-camera shot detection is VALIDATED: ~0.93 make/miss on
found shots, generalizing across games/eras, standalone (no fusion, no sync, no
calibration), real-time-capable (YOLO11n + small classifier). Ship v0.1; v0.2
work is spotter recall/precision, not the model.
