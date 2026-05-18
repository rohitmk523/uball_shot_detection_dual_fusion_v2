# P1 Track Extraction — Infra & Run Guide

Extracts per-frame ball/rim tracks for every annotated shot window in each
selected game, using the **frozen v1 detector** (reused as-is, never
retrained). All GPU work runs on a short-lived, self-terminating EC2 spot
instance. The laptop only stages code and reads results.

## Components

| File | Purpose |
|------|---------|
| `pipeline/common.py` | Manifest, GT (Supabase REST), S3, bundle helpers. |
| `pipeline/frozen_bundle.py` | Download + **sha256-verify** + unpack the frozen detector; import v1 code. |
| `pipeline/extract_tracks.py` | One game: pull 4 videos, run frozen detector over GT windows, write `tracks.parquet`+`tracks_meta.json` to S3. Idempotent. |
| `pipeline/run_batch.py` | Iterate games sequentially; update `dual_fusion_v2.{iterations,games}` or fall back to S3 progress JSON. |
| `infra/bootstrap.sh` | EC2 user-data: arm hard kill, deps, verify bundle, run batch, **always** self-terminate. |
| `infra/launch_p1.sh` | Laptop launcher with the **hard budget guard**. |

## Run order

```bash
# 0. One-time: the frozen bundle + its .sha256 are already at
#    s3://uball-cv-results/cv-results/dual-fusion-v2/frozen_detector_v16/

# 1. Preflight everything WITHOUT spending money or hitting run-instances:
infra/launch_p1.sh --split=val --dry-run

# 2. Real launch (one spot box; self-terminates at the budget cap):
infra/launch_p1.sh --split=val
#    or:  infra/launch_p1.sh --games="<id1> <id2>"
#    splits: train | val | test | all  (from data/games_manifest.json)

# 3. Monitor (cheap, never blocks on the box) — see below.

# 4. Fetch results — see below.
```

`extract_tracks.py` / `run_batch.py` can also be run directly on any GPU
box that has the deps; the launcher is just the safe, audited path.

## Cost-safety model (read before launching)

The guard in `launch_p1.sh` is **fail-closed and conservative**:

1. `spot_price` = current lowest g4dn.xlarge spot price in-region.
2. `budget_price = max(spot_price, SPOT_FLOOR)` — never divide the budget
   by an unrealistically small/zero price.
3. `cap_hours = AWS_MAX_USD_PER_ITERATION / budget_price`.
4. `HARD_CAP_MINUTES = floor(cap_hours * 60)` — **always rounded DOWN**.
5. `worst_case_usd = (HARD_CAP_MINUTES / 60) * ON_DEMAND_PRICE` — priced at
   on-demand (which is also the spot **max-price**), so the real bill can
   never exceed this.
6. **Refuse to launch** if `cumulative_iteration_cost + worst_case_usd >
   AWS_MAX_USD_TOTAL`, or if the computed cap is `< 1 min`.
7. The instance is launched with
   `--instance-initiated-shutdown-behavior terminate`, spot
   `MaxPrice = ON_DEMAND_PRICE`, and `bootstrap.sh` arms
   `shutdown -h +HARD_CAP_MINUTES` **before any work** — so the box dies at
   the cap **regardless of job state**, and an `EXIT`/`ERR` trap always
   uploads logs and terminates even on crash or spot interruption.

Net guarantee: the maximum possible spend for one run is `worst_case_usd`,
and the launcher refuses if that would breach `AWS_MAX_USD_TOTAL`.

If Supabase is unreachable the launcher cannot read prior spend; it warns
loudly and treats prior spend as `0` — **verify cumulative spend manually**
before launching in that case.

## Monitoring

```bash
# Instances for this project / run:
aws ec2 describe-instances \
  --filters Name=tag:Project,Values=dual-fusion-v2 Name=tag:RunId,Values=<run_id> \
  --query 'Reservations[].Instances[].{Id:InstanceId,State:State.Name,Launch:LaunchTime}'

# Progress JSON (written after every game; resumable view):
aws s3 ls   s3://uball-cv-results/cv-results/dual-fusion-v2/progress/P1_extract/
aws s3 cp   s3://uball-cv-results/cv-results/dual-fusion-v2/progress/P1_extract/<run_id>.json -
# Completion marker + bootstrap log:
aws s3 cp   s3://uball-cv-results/cv-results/dual-fusion-v2/progress/P1_extract/<run_id>.DONE -
aws s3 cp   s3://uball-cv-results/cv-results/dual-fusion-v2/progress/P1_extract/<run_id>.bootstrap.log -
```

If `dual_fusion_v2` Supabase tracking is enabled, also check the
`iterations` row (`status` running→done/failed, `cost_usd`) and
`games.tracks_extracted_at` / `tracks_s3_key`.

## Fetching results

Per game (immutable, written once):

```
s3://uball-cv-results/cv-results/dual-fusion-v2/tracks/<game_id>/tracks.parquet
s3://uball-cv-results/cv-results/dual-fusion-v2/tracks/<game_id>/tracks_meta.json
```

```bash
aws s3 cp s3://uball-cv-results/cv-results/dual-fusion-v2/tracks/<game_id>/ ./tracks/<game_id>/ --recursive
```

`tracks_meta.json` records: per-angle fps/frame counts, every shot window
used, the **frozen bundle sha256**, and the git sha — enough to reproduce.

## Idempotency / immutability

- Re-running a game **skips** it if `tracks_meta.json` already exists
  (pass `--force` to overwrite).
- Source videos and the frozen bundle are read-only inputs and never
  mutated. Outputs are written once per game.
