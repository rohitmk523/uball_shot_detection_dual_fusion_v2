# NEAR pipeline e2e: YOLO vs RF-DETR (4 frozen games, stride 3)

Identical pipeline (spotter + ResNet18 make/miss classifier + GT matching);
ONLY the ball+hoop detector swapped. Run on AWS g4dn GPU (torch cu121).
YOLO side reproduces the documented frozen baseline (mean e2e 0.930) -> setup valid.

| game | YOLO spot | RFD spot | YOLO e2e | RFD e2e | YOLO prec | RFD prec |
|---|---|---|---|---|---|---|
| ee8745f1 | 0.657 | 0.629 | 0.826 | 0.818 | 0.491 | 0.456 |
| 6d601c99 | 0.906 | 0.781 | 0.931 | 0.920 | 0.627 | 0.604 |
| 0fa23810 | 0.933 | 0.933 | 0.964 | 0.929 | 0.660 | 0.705 |
| c2a354fe | 0.775 | 0.825 | 1.000 | 1.000 | 0.639 | 0.684 |
| **MEAN** | **0.818** | **0.792** | **0.930** | **0.917** | | |

## Verdict
- Detector level (curated 486-img split): RF-DETR better (ball-recall@rim 0.902 vs 0.825).
- Pipeline level: YOLO marginally AHEAD (spot 0.818 vs 0.792, e2e 0.930 vs 0.917).
- make/miss e2e_acc ~tied (~0.92) -> classifier-bound, detector-agnostic (as predicted).
- RF-DETR's higher raw recall + lower precision -> more false events, not more real shots.
  6d601c99: RF-DETR 25/32 vs YOLO 29/32.
- Only clear RF-DETR upside for NEAR = license (Apache vs AGPL), not accuracy.

Caveats: small per-game samples (30-40 shots, 1-4 shot swings); 6d601c99 drives the gap.
