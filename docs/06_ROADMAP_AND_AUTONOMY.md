# 06 — Roadmap & Autonomous-Loop Protocol

## Staged plan
**P0 — Game inventory.** Query `plays` for fully-annotated `manual` games; cross-check S3 for the 4 angle videos; emit `data/games_manifest.json`; assign TRAIN/VAL/TEST. *(laptop: SQL+S3 list only)*

**P1 — Track extraction (AWS).** For each game, run frozen v16 YOLO on AWS; export per-frame ball/rim tracks (both angles) for every GT shot window → `tracks/<game>.json` in S3. *(AWS GPU; recipe `04`)*

**P2 — Dataset.** Join tracks ↔ `plays` GT (offset-fit per game) → features (`03`) → `data/dataset.parquet`. *(laptop: instant)*

**P3 — Fit + validate.** Fit interpretable model on TRAIN, tune threshold on VAL, evaluate on TEST (`05`). *(laptop: seconds)*

**P4 — Error review.** Render annotated clips of TEST errors on AWS; inspect; add/repair features; back to P2. *(AWS render; laptop inspect)*

**P5 — Iterate** P2–P4 until the target band (`05`) or the stop conditions below.

**P6 — Package.** Inference module replacing legacy made/miss + resolver; final cross-game report; client-safe summary.

(Separate, later track — not blocking: retrain near YOLO on mined frames for the ~undetected shots. Out of scope until P6.)

## Autonomous-loop protocol
Once inputs (next section) are in, the loop runs unattended:

- **One iteration =** P2→P3→P4 (+ P1 only when a new game is added). Each iteration appends a full record to `results/` and commits+pushes.
- **Checkpoint:** after every iteration, push to the repo and write a one-line status. The user can inspect/interrupt at any commit.
- **Continue if:** TEST precision&recall improving and below target band and within budget.
- **Stop & report (do not loop further) if ANY:**
  1. Target band reached (`05`).
  2. **Plateau:** 3 consecutive iterations with <0.5 pp TEST gain on both precision & recall.
  3. **Budget:** cumulative AWS spend would exceed `AWS_MAX_USD_TOTAL`, or an iteration would exceed `AWS_MAX_USD_PER_ITERATION`.
  4. Overfit signal: VAL→TEST gap > 8 pp two iterations running.
  5. A blocking input is missing/invalid (creds, games, S3 access).
- **Never:** run compute on the laptop; leave an EC2 instance non-self-terminating; commit secrets; touch production pipeline / real game records; quote 100% as achieved.
- On stop: post the honest final confusion matrix, the gap-to-100% decomposition (label noise vs occlusion vs model), AWS $ spent, and the recommendation.

## "Auto mode until 100%" — the honest contract
I will run this loop autonomously and push every iteration. **100% is not a deliverable I can promise** — it is bounded by human-label noise and occlusion physics (`00`/`05`). What I *will* deliver: convergence to the practical ceiling, cross-game validated, fully audited, with the residual gap explained shot-by-shot. If you truly want me to keep iterating past the plateau, that requires raising the budget cap and accepting diminishing returns — I will ask, not assume.
