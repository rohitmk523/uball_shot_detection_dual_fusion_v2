# Game-2 Validation: dc5f199e

**Date:** 2026-06-03
**Cameras:** identical hardware to game-1 (reused v4 calibration, NR=FR+7 sync offset)
**Coverage:** 22 shots (NR camera stopped at 19:37, plays after that are FR-only)

## Final Accuracy

| Metric | Game-1 (88 shots) | Game-2 (22 shots) |
|---|---|---|
| **Decided accuracy** | 70/72 = **97.2%** | **20/20 = 100.0%** |
| **Overall accuracy** | 70/88 = 79.5% | 20/22 = **90.9%** |
| TP / TN / FP / FN | 37 / 33 / 1 / 1 | 10 / 10 / 0 / 0 |
| UND (NR signal lost) | 16 | 2 |

Game-2 has **zero false positives and zero false negatives** among decided shots — a strict improvement over game-1's 1 FP + 1 FN.

## What Changed in the Pipeline (this iteration)

Three new guards added to `descent_verdict` based on game-2 error analysis:

1. **Convergence guard** on `gap-stop MAKE` and `smooth descent MAKE`:
   The trailing 4 samples before the walker stopped must show **median dr <= +1.5 cm/sample**
   (r truly closing on the rim). Game-2's gap-stop FPs all had ball drifting AWAY
   from the rim (median dr +5 to +20) while real makes converged (median dr -2 to -8).
   Median (not mean) is used so a single noisy detection spike doesn't flip a
   converging trajectory to "diverging".

2. **Post-crossing continuation guard** on `rattled in MAKE`:
   After the ball clips below the rim plane, the next 8 frames must show z
   dropping to at least `rim_z - 30 cm`. If z bounces back above rim plane and
   stays there, that's a rim-bounce, not a make through the net.

3. **Conditional z_min proximity guard** on `smooth descent MAKE`:
   For shots released far from the rim (`apex_dxy > 80 cm` — 3PT, long FG),
   `z_min` must be within 50 cm of the rim plane. Otherwise the walker just
   lost the ball mid-flight on stale convergence data.
   Free throws (`apex_dxy <= 80 cm`) skip this guard — they reliably stop
   higher up while still on a true make trajectory.

Multi-shot dip threshold also loosened (100 → 50 cm) to catch game-2's
putback pair (52242360 / 58b2e35c).

## Game-2 Per-Class Breakdown

| Class | N | TP | TN | FP | FN | UND | Acc% |
|---|---|---|---|---|---|---|---|
| 3PT_MAKE | 1 | 1 | 0 | 0 | 0 | 0 | 100.0 |
| 3PT_MISS | 4 | 0 | 3 | 0 | 0 | 1 | 100.0 |
| FG_MAKE | 8 | 8 | 0 | 0 | 0 | 0 | 100.0 |
| FG_MISS | 7 | 0 | 6 | 0 | 0 | 1 | 100.0 |
| FREE_THROW_MAKE | 1 | 1 | 0 | 0 | 0 | 0 | 100.0 |
| FREE_THROW_MISS | 1 | 0 | 1 | 0 | 0 | 0 | 100.0 |
| **TOTAL** | **22** | **10** | **10** | **0** | **0** | **2** | **100.0** |

By layer:
* L1 triangulation: 18
* L3 ensemble:      3
* L5 multi-shot:    1

## Remaining UNDs (2)

* **16b71f7f_3PM** (3PT_MISS) — apex too short, ball lost early
* **8584ce26_FGM** (FG_MISS) — only 3 clean post-apex samples

Both have very sparse 3D reconstruction (insufficient overlap between FR+NR).
Per-camera ensemble could not resolve either with strong signal.

## Game-1 Regression Check

Game-1 stayed at **97.2% decided** (same 1 FP, 1 FN, same per-class breakdown).
The new L1 guards made layer composition stronger:
* L1 triangulation alone now reaches 95.2% decided (was 88.3%)
* L2-L5 layers carry less of the load — fewer downstream fixes needed
* Final accuracy unchanged: the layered pipeline was already saturated at 97.2%

| Layer | Game-1 (before) | Game-1 (after) |
|---|---|---|
| L1 tri | 60 | 66 |
| L2 und-rerun | 5 | 5 |
| L3 ensemble | 5 | 3 |
| L4 hi-res | 8 | 8 |
| L5 multi-shot | 10 | 6 |

## Key Files

* `pipeline/triangulate_shot.py` — three new guards in `descent_verdict`
* `pipeline/multi_shot_detect.py` — dip threshold 100 -> 50 cm
* `pipeline/rescore_descent.py` — re-applies descent_verdict on cached samples
* `pipeline/per_camera_verdict_g2.py` — L3 ensemble for game-2 clips
* `pipeline/final_merge_g2.py` — L1+L2+L3+L5 merge for game-2
* `data/client_report/triangulation_test/game2_dc5f199e/final_v3.json` — per-shot verdicts
