# Near-Angle Detector v1 — Results (2026-06-13)

Warm-started from old near best.pt, fine-tuned on 4,386 verified overhead frames
(Rohit annotated all 5,196), YOLO11n @1280px, AWS T4. Spot-reclaimed at epoch 47
but plateaued since ep25; best.pt saved at peak. Whole-game holdout (no leakage).

## Held-out VAL (9eb51980, never trained): mAP50 = 0.919 (peak)

## Held-out TEST (b3c1f62c SuperView + 72c08cb7 Wide, 486 imgs) — production metrics:
| metric | value | note |
|---|---|---|
| HOOP recall | **1.000** | 485/485, 1 FP — overhead hoop SOLVED (old model ~2% fire rate) |
| HOOP precision | 0.998 | |
| BALL recall | 0.786 | 232/295; ball is the harder class |
| BALL precision | 0.885 | |
| **BALL recall @ RIM MOMENT** | **0.825** | 160/194 — the frames that decide make/miss |

## Verdict
- Hoop detection is production-ready: reliable rim localization + event gating on
  the overhead view, both eras, both baskets, incl. Wide clipped-rim + teal ball.
- Ball recall 0.79 / 0.825-at-rim is good-not-perfect: fine for multi-frame
  event-spotting and as an auxiliary classifier channel; v1.1 levers = more
  rim-moment ball boxes, motion-blur aug, imgsz 1536.

## Weights
- local: near_v0/weights/near_det_v1_best.pt
- S3: s3://uball-videos-production/_tmp_tri/near_det/detector/ckpt/best.pt

## Known issue
AWS spot retry can't resume ultralytics (needs original run dir). For long GPU
trains use on-demand, or checkpoint+resume via a run-dir tar to S3, not just last.pt.
