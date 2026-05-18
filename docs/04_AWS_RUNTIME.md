# 04 — AWS Runtime (NO laptop compute, ever)

All YOLO inference / track extraction / clip rendering runs on AWS GPU. The laptop only: edits code, runs lightweight JSON/feature/CSV joins & model fitting on CPU (seconds), drives AWS, reads results. **No detector/video work locally** (heat + speed).

## Proven recipe (reuse — this worked in v1 validation)
- **Instance:** `g4dn.xlarge` **spot**, AMI `ami-0cff16a9a375c81a5` (Ubuntu 22.04 + NVIDIA driver), region `us-east-1`, a default subnet in **us-east-1a/b/c** (NOT 1e — no g4dn there), default SG `sg-d1ad2a8c`, `--associate-public-ip-address`, 80 GB gp3 root.
- **Instance profile:** `uball-cv-builder-profile` — already has: read `uball-videos-production` + `uball-cv-models`, read/write `uball-cv-results`, `ec2:TerminateInstances`. No new IAM needed.
- **Self-terminate:** `--instance-initiated-shutdown-behavior terminate` + `shutdown -h now` at end of user-data + a `trap` that always uploads logs then shuts down. Never leave an instance running.
- **Code/data transport:** tar the repo/job → `s3://...dual-fusion-v2/...`; instance pulls from S3 (no GitHub auth on the box). Pull the 4 videos + frozen YOLO weights from S3 in-region (fast). Push outputs (tracks, dataset, metrics) back to `S3_WORK_PREFIX`.
- **Status:** instance writes `STATUS.txt` / `DONE` markers to `S3_WORK_PREFIX`; laptop polls S3 (cheap), never blocks on the box.

## Cost guardrails (from `.env`)
- `AWS_USE_SPOT=true`, `AWS_GPU_INSTANCE_TYPE=g4dn.xlarge` (~$0.20–0.25/hr spot).
- `AWS_MAX_USD_PER_ITERATION` / `AWS_MAX_USD_TOTAL`: before launching, estimate runtime; abort & report if an iteration would exceed per-iteration cap; **hard stop** the loop if cumulative would exceed total cap (see `06`).
- One game's 4-angle track extraction ≈ the v1 validation run (~1–2 h on T4). Fitting/eval is laptop-CPU and free. Render review clips only for error cases.

## Credentials
From `.env` (gitignored). The instance uses the **instance profile** (no keys on the box). The laptop uses the `.env` AWS keys only to call `aws ec2 run-instances` / `s3` / Supabase. Claude never writes real keys into any file.

## Standard job shape
1. Laptop: build job tarball (code + game manifest), upload to `S3_WORK_PREFIX/jobs/`.
2. Laptop: `aws ec2 run-instances …` with user-data that: installs deps → pulls code+videos+weights → runs extraction/inference → uploads tracks+dataset+metrics → `DONE` → self-terminates.
3. Laptop: poll S3 for `DONE`; on completion pull the small artifacts (parquet/JSON/metrics); fit/evaluate model locally (CPU, instant); record result.

Reference implementation to adapt: the v1 validation run used exactly this pattern (instance `i-0c85f346102f75b1a`, user-data did deps→pull→`dual_angle_fusion.py`→S3→terminate). Same skeleton here, swapped payload.
