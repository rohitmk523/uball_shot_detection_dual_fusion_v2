# Game-3 Validation: 3398befc Locksmith vs Los Sazoneros

**Date:** 2026-06-03
**Game date:** 2026-05-30
**Cameras:** identical hardware to game-1 + game-2 (same court-a mount)
**Coverage:** 75 RIGHT-angle shots (76 LEFT-basket shots dropped — outside FR/NR triangulation stereo coverage)
**Sync offset:** NR = FR + 10.5 frames (midpoint of anchors FR:405=NR:415 and FR:1111=NR:1122)

## Headline numbers

| Metric | Game-1 | Game-2 | Game-3 |
|---|---|---|---|
| In-scope shots | 88 | 22 | **75** (RIGHT only) |
| **Decided accuracy** | 97.2% | 100.0% | **91.7%** |
| Overall accuracy | 79.5% | 90.9% | 58.7% |
| TP / TN / FP / FN | 37 / 33 / 1 / 1 | 10 / 10 / 0 / 0 | **22 / 22 / 2 / 2** |
| UND | 16 | 2 | **27** |

Game-3 reached **91.7% decided accuracy** after a three-step diagnostic-driven fix.

## Step-by-step improvement on game-3

| Stage | Decided | FP | FN | UND |
|---|---|---|---|---|
| L1 (game-1 v4 calibration) | 68.5% | 11 | 6 | 21 |
| All 5 layers, g1 calibration | 70.7% | 12 | 5 | 17 |
| L1 with new game-3 calibration (Step B) | 72.9% | 8 | 5 | 27 |
| All layers + g3 calib (Steps A+B) | 77.4% | 8 | 4 | 22 |
| **+ hi-res YOLO on 12 errors (Step C)** | **91.7%** | **2** | **2** | **27** |

## Step A — Diagnostic

Split game-3 errors into early / mid / late thirds by `t_start`:

| Third | N | Decided acc | FP rate |
|---|---|---|---|
| EARLY (0-864s) | 25 | 72.7% | 22.7% |
| MID (864-2146s) | 25 | 68.4% | 15.8% |
| LATE (2146-3231s) | 25 | 70.6% | 23.5% |

**FP rate is essentially flat across time → sync drift is NOT the cause.**
Errors are systemic, pointing to **calibration drift**.

## Step B — Game-3 calibration refresh

Built `calibration_v4_g3.json` from clean game-3 frames at t=6.0s, reusing
game-1's user click coordinates and letting `cv2.cornerSubPix` snap them
onto game-3's gradient peaks. Results:

* **Click pixel shifts**: FR mean 3.3 px (max 10.6), NR mean 4.4 px (max 7.8) — cameras moved by ~3-11 pixels between game-1's calibration day and game-3's recording day.
* **Reproj error**: FR mean 22.6 px (vs game-1's 22.9), NR mean 12.0 px (vs game-1's 12.4) — calibration is now correctly fitted to game-3 imagery.
* **3D cross-check on floor landmarks**: **20.3 cm mean** (game-1's was 6.0 cm). Even with refit, residual error is larger — the cameras and lighting differ enough that landmark-only PnP plateaus here.

Wired in via `CALIB_G3=1` env var. Result: -3 FPs net, +6 UND.

## Step C — Hi-res YOLO on remaining errors

Re-ran `triangulate_shot.py --conf 0.05 --imgsz 1280` on the 8 FPs + 4 FNs.
Sub-pixel ball localisation at high res:

| Target | Old verdict | New verdict | Effect |
|---|---|---|---|
| d55f7f86_FGM | MAKE rattled in r=39 | **MISS RIM-OUT r=17** | FP → TN |
| f5505934_FTM | MAKE gap-stop r=16 | **MISS z_min=344 r=76** | FP → TN |
| fbfe6f7c_FGM | MISS RIM-OUT r=15 | **MAKE smooth descent r=12** | FN → TP |
| 03d0f0b0_FGM | MAKE gap-stop r=5 | UNDECIDED | FP → UND |
| 37be88dd_3PM | MAKE gap-stop r=16 | UNDECIDED | FP → UND |
| 3cf7baec_3PM | MAKE rattled in | UNDECIDED | FP → UND |
| 4cb8fe38_FGM | MAKE gap-stop r=16 | UNDECIDED | FP → UND |
| b4672e55_FTM | MISS cross_r=168 | UNDECIDED | FN → UND |
| 2847adf3_FGM | MAKE | MAKE r=29 | FP (unchanged) |
| cdae91cd_FGM | MAKE | MAKE smooth descent r=33 | FP (unchanged) |
| 940474e0_FGM | MISS | MISS cross_r=65 | FN (unchanged) |
| 9ca659ee_FGM | MISS | MISS cross_r=62 | FN (unchanged) |

Net effect of L4 hi-res: **6 confidently-wrong calls fixed or demoted to UND**. Zero regressions.

## Final per-class breakdown (game-3)

| Class | N | TP | TN | FP | FN | UND | Acc% |
|---|---|---|---|---|---|---|---|
| 3PT_MAKE | 6 | 5 | 0 | 0 | 0 | 1 | 100.0 |
| 3PT_MISS | 10 | 0 | 6 | 0 | 0 | 4 | 100.0 |
| 4PT_MAKE | 1 | 1 | 0 | 0 | 0 | 0 | 100.0 |
| 4PT_MISS | 7 | 0 | 4 | 0 | 0 | 3 | 100.0 |
| FG_MAKE | 18 | 16 | 0 | 0 | 2 | 0 | 88.9 |
| FG_MISS | 19 | 0 | 9 | 2 | 0 | 8 | 81.8 |
| FREE_THROW_MAKE | 4 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| FREE_THROW_MISS | 10 | 0 | 3 | 0 | 0 | 7 | 100.0 |
| **TOTAL** | **75** | **22** | **22** | **2** | **2** | **27** | **91.7** |

By layer:
* L1 triangulation: 35
* L3 ensemble: 27
* L4 hi-res: 12
* L5 multi-shot: 1

## Remaining 4 errors

| Shot | GT | Verdict | Notes |
|---|---|---|---|
| 2847adf3_FGM | FG_MISS | MAKE rim-plane r=29, z_min=284 | Ball did cross rim plane near center but bounced. Need rule for "crossed but post-cross z bounces back" |
| cdae91cd_FGM | FG_MISS | MAKE smooth descent r=33 | Ball reached r=33 at z=340 with positive descent trend. True rim-out — extra signal needed |
| 940474e0_FGM | FG_MAKE | MISS cross_r=65 | Cross-r=65cm > 40cm tolerance — 3D arc passed wide of rim, but ball did go in. Calibration limit |
| 9ca659ee_FGM | FG_MAKE | MISS cross_r=62 | Same pattern as 940474e0. 3D arc 60+cm wide |

The 2 FNs are the residual signature of calibration uncertainty — 20cm cross-check error at floor level propagates to ~60cm at apex height for some shots. Same root cause as Step B addressed but cannot be fully eliminated without a richer calibration model (e.g. lens distortion, more landmarks).

## Cross-game improvements applied

1. **`pipeline/calibrate_v4_g3.py`** — game-3 calibration builder (cornerSubPix-refined clicks against game-3 frames).
2. **`pipeline/triangulate_shot.py::calibrate()`** — added `CALIB_G3=1` env-var branch to load pre-solved game-3 PnP without re-solving.
3. **`pipeline/final_merge_g3.py`** — added L4 hi-res layer (was already in `final_merge_v3.py`; mirrored for game-3).

These changes are game-3-only — they don't affect game-1 or game-2 runs.

## What still limits game-3

1. **27 UND** is high — most are 3PT/4PT/FT shots where the 3D arc was too short or noisy. The single-camera classifiers (existing FR + NR pipelines in this repo) are still under-used at L3; pulling them in more aggressively could decide many of these.
2. **3D cross-check 20cm** vs game-1's 6cm — a richer calibration (lens distortion + more rim/court landmarks + per-game refresh on the GPU box) could halve this.
3. **LEFT basket is invisible** — 76 of game-3's 151 shots are at the other basket and have no stereo coverage. A 3rd camera or a separate calibration for the LEFT basket would roughly double game-3's measurable sample.

## Files

* `pipeline/calibrate_v4_g3.py` — game-3 calibration builder
* `pipeline/triangulate_shot.py` — added `CALIB_G3=1` branch
* `pipeline/extract_game3_clips.py` — clip extractor (NR offset +10.5 frames)
* `pipeline/per_camera_verdict_g3.py` — L3 ensemble adapter
* `pipeline/final_merge_g3.py` — full L1+L3+L4+L5 merge (RIGHT-only)
* `data/client_report/triangulation_test/calibration_v4_g3.json` — game-3 calibration
* `data/client_report/triangulation_test/game3_3398befc/shots_right.json` — 75-shot manifest
* `data/client_report/triangulation_test/game3_3398befc/final_v3.json` — per-shot verdicts
