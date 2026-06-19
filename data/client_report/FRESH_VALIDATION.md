# Fresh Out-of-Sample Validation — the 0.955 ceiling holds on unseen games

**Date:** 2026-05-24
**What this is:** The production angle-aware model (`p3_model_angleaware.joblib`)
was scored on **5 brand-new games it has never seen** (recorded 2026-05-16 /
05-19), to test whether the in-sample **0.955** test accuracy is real or
overfit. **No retraining, no threshold re-tuning** — same model, same
production decision threshold (0.310, recovered exactly: refit reproduces the
committed test predictions with max|Δprob| = 0).

## Result

| Game | Shots | Makes | Acc | Prec | Recall | FP | FN |
|------|------:|------:|----:|-----:|-------:|---:|---:|
| 77715f25 | 205 | 101 | 0.961 | 0.935 | 0.990 | 7 | 1 |
| b3c1f62c | 148 | 59 | 0.966 | 0.922 | 1.000 | 5 | 0 |
| cc1710c4 | 177 | 85 | 0.932 | 0.876 | 1.000 | 12 | 0 |
| cc5deb39 | 173 | 67 | 0.936 | 0.859 | 1.000 | 11 | 0 |
| f3e7b25a | 151 | 52 | 0.960 | 0.897 | 1.000 | 6 | 0 |
| **OVERALL** | **854** | **364** | **0.951** | **0.899** | **0.997** | **41** | **1** |

Confusion `[[TN 449, FP 41], [FN 1, TP 363]]`.

## Read

- **Out-of-sample 0.951 vs in-sample test 0.955 — a 0.4 pt drop.** The ceiling
  is **real and reproducible**, not overfit. The model generalizes to games it
  was never trained, validated, or tested on.
- **Recall 0.997 — only 1 missed make out of 364** across 5 unseen games. The
  system almost never misses a made basket.
- **42 of 43… i.e. 41 of 42 errors are false positives** (model says *make*,
  truth is *miss*). This is the **depth-illusion** signature: a ball passing in
  front of / behind the rim from an oblique 2D angle reads as "through." It is
  exactly the residual that an overhead rim-axis camera removes (see
  [`NOAH_HARDWARE_BLUEPRINT.md`](./NOAH_HARDWARE_BLUEPRINT.md)).

## Net-motion mattered (as predicted)

A preliminary score on the first 2 games **without** net-motion features was
0.951 / 0.919 (combined 0.938). Adding net-motion lifted those same two games to
**0.961 / 0.966** — net-motion cut false positives (e.g. b3c1f62c FP 12 → 5),
confirming it disambiguates real makes (net disturbance) from depth-illusion
over-calls.

## Bottom line

The **software ceiling on existing oblique game footage is ~95%** and it is
**honest** — it holds on fresh, unseen games at 0.951 with near-perfect recall.
To push past it toward ~99%+, the path is hardware (one calibrated rim-axis
camera/hoop), not more software. See
[`ROAD_TO_100.md`](./ROAD_TO_100.md) and
[`NOAH_HARDWARE_BLUEPRINT.md`](./NOAH_HARDWARE_BLUEPRINT.md).

## Reproduce

```bash
# tracks (far_v16) + net-motion already extracted to
#   s3://uball-cv-results/cv-results/dual-fusion-v2-fresh/{tracks,netmotion}/
python3 pipeline/score_fresh.py     # writes data/p3_fresh_predictions.parquet
```

Artifacts: `data/p3_fresh_predictions.parquet` (per-shot prob/pred/correct),
fresh GT at `s3://.../dual-fusion-v2/gt_fresh/gt_windows.json` (sha b190d201…).
Total AWS cost for this validation: ~$5 (track extraction ~$3 + net-motion ~$1.7).
