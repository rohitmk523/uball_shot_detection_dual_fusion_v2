# uball Shot Detection v2 — Client Report

**Date:** 2026-05-25
**System:** real-time make/miss detection from 4 game cameras (Far-Left/Right, Near-Left/Right), trained YOLO11n ball+rim detectors → angle-aware far/near fusion. No VLM, no per-frame heavy model, no cloud calls.

This report covers, in order: (1) our current software accuracy ceiling, (2) what Noah Basketball does better that our near angle is missing, (3) what we can do with that knowledge, (4) why we must first **synchronize all 4 cameras**, and (5) how sync unlocks **triangulation** (true 3D → ~99% with no extra sensor).

> Visuals referenced below live in this folder; paths are given so they can be embedded. `IMAGE:` markers note where a screenshot should be dropped in.

---

## 1. Current software ceiling — **~95%, and it is real (not overfit)**

Progression: v1 baseline 0.857 → iter8 0.905 → **far-detector upgrade + angle-aware fusion = 0.955** (held-out test).

We then validated on **5 brand-new games the model had never seen** (recorded 2026-05-16/19, **854 shots**), with **no retraining and no threshold change**:

| Fresh game (out-of-sample) | Shots | Acc | Prec | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 77715f25 | 205 | 0.961 | 0.935 | 0.990 | 7 | 1 |
| b3c1f62c | 148 | 0.966 | 0.922 | 1.000 | 5 | 0 |
| cc1710c4 | 177 | 0.932 | 0.876 | 1.000 | 12 | 0 |
| cc5deb39 | 173 | 0.936 | 0.859 | 1.000 | 11 | 0 |
| f3e7b25a | 151 | 0.960 | 0.897 | 1.000 | 6 | 0 |
| **OVERALL** | **854** | **0.951** | 0.899 | **0.997** | **41** | **1** |

- **Out-of-sample 0.951 vs in-sample 0.955 → 0.4-pt drop. The ceiling is genuine and reproducible.**
- **Recall 0.997 — only 1 missed make in 364.** The system almost never misses a made basket.
- **41 of 42 errors are the same failure: a MISS called a MAKE** (the "depth illusion", §2).

`VIDEO:` [fresh_error_reel/fresh_error_reel.mp4](fresh_error_reel/fresh_error_reel.mp4) — all 42 errors, far|near side-by-side, subtitled.
Details: [FRESH_VALIDATION.md](FRESH_VALIDATION.md).

---

## 2. The wall: the **depth illusion** (why ~95%, not ~99%)

Make/miss is one question: *did the ball pass through the rim?* From our **oblique, side/near 2D angles** a ball passing **in front of or behind** the rim can look identical to one going **through** it — the camera angle collapses depth. That's the entire residual error (all 41 false positives).

We **proved** it isn't a modelling gap — every attempt to fix it in 2D failed:
- far-camera "veto the make" rule → broke more true makes than it fixed;
- reliability-weighted fusion, stacked meta, blend, confidence-gated fusion → **none beat the fusion**;
- Noah's monocular depth cue on the near angle (§3) → helps near alone, redundant in fusion.

The information is **not in the pixels** of an oblique 2D view. `IMAGE:` (drop a 2-up still from the reel showing far=clear-miss, near=looks-in).

---

## 3. What Noah does better — and what our **near angle** is missing

Noah Basketball (28 NBA teams) is the shot-tracking leader. Their accuracy comes **not from better AI but from camera geometry**:

| | **Noah** | **Our near camera (NL/NR)** |
|---|---|---|
| Placement | **Overhead, on the rim axis** (rim seen as a true circle → "through" is *measured*) | **Oblique baseline** (rim seen edge-on → depth ambiguous) |
| Lens mode | Clean / calibrated | **SuperView** (widest, most distorted; softest at edges where the rim sits) |
| Per-hoop calibration | **Yes** (pixels → real inches/degrees) | None |
| Ball at the rim | Sharp | **Heavily motion-blurred** (see below) |

**The blur problem (measured on a real shot):** the near camera is close to the hoop, so a ball at the rim is ~200 px and its fall smears across many pixels per frame at 30 fps SuperView — the detected ball box is a smeared **170×127 px** blob (and unstable frame-to-frame), versus a clean **33×33 px** ball in the far camera. That blur is exactly why a size-based depth cue on the near angle is unreliable.

`IMAGE:` [calib_freethrow/blur_far_vs_near.png](calib_freethrow/blur_far_vs_near.png) — far (sharp 33px) vs near (blurred 170px) ball at the rim.

**We did apply Noah's idea to our near angle** (known-ball-size depth cue, at the rim crossing):
- Near angle **alone improved: 0.876 → 0.902** (+2.6 pts).
- But in the **full fusion it's redundant** (0.951 → 0.948) — the far angle already supplies that depth.

So with the **current cameras**, the near angle cannot lift the product. Details: [NEAR_ANGLE_NOAH_RESULTS.md](NEAR_ANGLE_NOAH_RESULTS.md), [NOAH_HARDWARE_BLUEPRINT.md](NOAH_HARDWARE_BLUEPRINT.md).

---

## 4. **First fix needed: synchronize all 4 cameras**

Today the 4 cameras are only **NTP wall-clock aligned (~0.3–0.5 s apart)** — not frame-synced. We measured it on one shot: the ball crosses the rim at **far frame 73** but **near frame 83** — a **~10-frame (0.33 s) offset.**

Why this must be fixed **before anything else**: at 8 m/s a ball travels ~3 m in 0.33 s, so the 4 views **cannot be combined frame-accurately**. No cross-camera 3D, no triangulation, no reliable far+near depth is possible until the cameras share a clock.

**Action:** hard-sync via **timecode/genlock** (shared sync signal or post-sync to a common flash/clap), targeting **sub-frame (<33 ms / <1 frame)** alignment.

`IMAGE:` [calib_freethrow/calib_freethrow_sidebyside.mp4](calib_freethrow/calib_freethrow_sidebyside.mp4) — numbered far|near frames showing the 73-vs-83 offset.

---

## 5. With sync: **triangulation becomes available → the software path to ~99%**

Once the 4 cameras are frame-synced **and** calibrated, two or more views of the ball at the rim let us **triangulate its true 3D position** — which makes "through vs in-front" a *measurement*, not a guess. **The depth illusion disappears with no extra sensor.**

- We **previously ruled triangulation out** *only because* of the ~0.5 s desync. **Hard sync puts it back on the table** as the most promising software route past 95%.
- It also yields **arc / entry-angle / depth / left-right** (Noah's coaching metrics) for free.

### 5a. We empirically tested *post-hoc* sync (from existing footage) — the signal is there, but the precision isn't
We derived per-game far↔near frame offsets from made-shot rim crossings across all 23 games (matched the client's manually-measured +10 on `b3c1f62c`), then built sync-aware cross-view consistency features into the fusion.

- **The depth-illusion signature does show up:** at the synced rim-crossing moment, far and near disagree on **44% of MISSES vs 12% of MAKES** — a 3–4× separation.
- **But fusion + sync-aware ended at 0.938 vs base 0.951 (−1.3 pts).** Reason: per-shot sync derived from events has σ 30–77 frames within a game — the median is correct, but no single shot is reliably aligned. Combined with the far angle already carrying most of that depth signal in the fusion, noisy SA features hurt more than they help.

**Implication (this is the strongest case yet for hardware sync):** the lift is in the data, but unlocking it requires **capture-time, sub-frame sync** (timecode/genlock) — not after-the-fact derivation. With hard sync at recording, each shot's SA alignment becomes accurate to <1 frame, and the 3–4× depth-illusion signature should translate into a real fusion gain. **The next recording session is the lever.**

This is the key strategic point: **sync is the unlock, but it has to happen at the camera, not in post.**

---

## 6. The capture upgrades that unlock accuracy (ranked, mostly low-cost)

| # | Change | Fixes | Cost |
|---|---|---|---|
| 1 | **Sync all 4 cameras** (timecode/genlock) | enables triangulation + true far/near 3D | low |
| 2 | **Linear mode** (not SuperView) | removes distortion; makes calibration possible | $0 (setting) |
| 3 | **Faster shutter** (≤1/480 s) | freezes the near ball → kills the rim blur | $0 (setting) |
| 4 | **Higher fps (60/120)** | fixes fast swishes under-sampled at 30 fps | $0 if supported |
| 5 | **Calibrate cameras** (intrinsics + pose) | pixels → real 3D (the Noah step) | low (one-time) |
| 6 | *(guaranteed)* **Overhead rim-axis camera per hoop** | resolves depth directly, like Noah | ~$300–500/hoop |

Items 1–5 are **software/settings on the cameras we already own**. Item 6 is the proven hardware guarantee if the client wants certainty.

---

## 7. In progress / what we need from the client

We've started **camera calibration** from a made free throw (fixed, known geometry). A numbered far|near frame set is ready for annotation: [calib_freethrow/](calib_freethrow/).

To finish it we need: **(a)** the near camera's **mounting height, distance to the hoop, and tilt angle**; **(b)** GoPro **model + mode** (confirmed near = SuperView 30 fps; **Linear is strongly recommended going forward**); **(c)** confirmed ball = **men's size 7 (9.39″)**, rim 18″ @ 10 ft (assumed).

---

## 8. Bottom line

- **Software is at ~95% and it's honest** — 0.951 on unseen games, near-perfect recall, for $0 extra hardware, in real time.
- The remaining ~5% is **one geometric problem (depth illusion)**, proven unfixable from our current oblique 2D views — including with Noah's monocular trick (which helps the near angle alone but is redundant in fusion).
- **The path past 95% is capture-side, and the first step is non-negotiable: synchronize the 4 cameras.** Sync → **triangulation** → true 3D make/miss (~99%) with no new sensor, plus Noah-style arc/depth metrics. Linear mode + faster shutter + higher fps remove the near blur and swish errors along the way.
- If the client wants a **certainty guarantee**, add **one overhead rim-axis camera per hoop** (Noah's actual method) — but with sync + calibration we can likely get most of the way there in software first.

---

## Appendix — deliverables & references

| File | Contents |
|---|---|
| [fresh_error_reel/fresh_error_reel.mp4](fresh_error_reel/fresh_error_reel.mp4) | All 42 fresh-game errors, far\|near + subtitles |
| [calib_freethrow/blur_far_vs_near.png](calib_freethrow/blur_far_vs_near.png) | Near blur vs far sharpness at the rim |
| [calib_freethrow/calib_freethrow_sidebyside.mp4](calib_freethrow/calib_freethrow_sidebyside.mp4) | Numbered far\|near free-throw frames (sync + calibration) |
| [error_highlights_FINAL/angleaware_22_problem_shots.mp4](error_highlights_FINAL/angleaware_22_problem_shots.mp4) | Earlier held-out-test error reel |
| [FRESH_VALIDATION.md](FRESH_VALIDATION.md) | Out-of-sample validation detail |
| [NOAH_HARDWARE_BLUEPRINT.md](NOAH_HARDWARE_BLUEPRINT.md) | Noah teardown + how we adopt it |
| [NEAR_ANGLE_NOAH_RESULTS.md](NEAR_ANGLE_NOAH_RESULTS.md) | Near-angle depth-cue experiments |
| [ROAD_TO_100.md](ROAD_TO_100.md) | Hardware/sensor options for ~100% |
| `0*.csv` | Per-shot predictions, errors, per-game summaries |

*Add screenshots at the `IMAGE:` markers (§2 depth-illusion still, §3 blur image is already linked, §4 sync frames).*

---

## UPDATE — 2026-06-15: New near-angle model lifts fusion accuracy to ~0.97

Since this report (2026-05-25) we rebuilt the **near-angle make/miss model** as a learned **rim-crop video classifier** (replacing the earlier geometric near-angle features). This updates one conclusion above: §3 and §5 found the near angle *redundant* in fusion — that was true of the **old, weak** near features. The **new near model is a strong, independent signal**, and re-running the far+near fusion with it now **improves accuracy**, with **both false positives and false negatives falling** (a real gain, not a threshold trade-off).

**Per-game accuracy — new far + near fusion (out-of-sample, leave-one-game-out):**

| Game | Date | Shots | Acc | Prec | Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| 29b51d57 | 2026-04-16 | 84 | 0.976 | 0.951 | 1.000 | 2 | 0 |
| 2c490f1a | 2026-04-16 | 42 | 0.976 | 1.000 | 0.923 | 0 | 1 |
| 74c4f686 | 2026-04-17 | 75 | 0.933 | 0.914 | 0.941 | 3 | 2 |
| 8dcb1330 | 2026-04-28 | 81 | 0.963 | 0.978 | 0.957 | 1 | 2 |
| 922bff3b | 2026-04-16 | 67 | 0.970 | 0.941 | 1.000 | 2 | 0 |
| 9eb51980 | 2026-04-17 | 80 | 0.950 | 0.971 | 0.919 | 1 | 3 |
| d0a9faef | 2026-04-17 | 71 | 0.972 | 0.970 | 0.970 | 1 | 1 |
| d446fe8c | 2026-05-15 | 84 | 0.964 | 0.917 | 1.000 | 3 | 0 |
| f66eb3b2 | 2026-05-15 | 77 | 0.987 | 1.000 | 0.976 | 0 | 1 |
| **OVERALL** | | **661** | **0.965** | **0.958** | **0.968** | **13** | **10** |

In a controlled head-to-head on **1,357 shots**, fusing the new near model lifted the **far-camera-alone** result **0.961 → 0.973**, with false positives 34→24 and false negatives 19→13. The near angle is now **complementary, not redundant**.

**What this means:** on the existing 4-camera setup, the software ceiling has moved from **~0.951 to ~0.965–0.973** — roughly a third of the prior error removed — at **$0 added hardware**. The path to near-perfect still runs through the §6 capture upgrades (sync, linear mode, faster shutter, and ultimately the overhead rim-axis camera), but the new near model is a real, immediate gain on footage we already have.

*Demo videos (per game, far + near detection + make/miss + shot-location map) accompany this update.*

---

## UPDATE — 2026-06-19: Detector swap — **RF-DETR vs YOLO** (Apache vs AGPL), same fusion pipeline

We evaluated replacing the **YOLO11n** ball+hoop detectors (AGPL-licensed) with **RF-DETR** (Apache-2.0) on **both** angles, keeping the rest of the v2 pipeline identical (same spotter, same near rim-crop classifier, same far trajectory/geometry features, same leave-one-game-out fusion). RF-DETR detectors were trained on the same held-out splits (near test mAP@50 0.94, ball-recall-at-rim 0.90 vs YOLO 0.83; far ball recall 0.90 vs 0.79).

**Apples-to-apples on the same 9 games / 1,364 shots** (both detectors run through one reconstructed LOGO pipeline; YOLO reconstruction = 0.969 vs the cached 0.973, so absolutes are ~0.4pt conservative — the **delta** is the trustworthy number):

| make/miss accuracy | YOLO | **RF-DETR** | Δ |
|---|---:|---:|---:|
| **Far camera alone** | 0.945 (FP 38, **FN 37**) | **0.971** (FP 26, **FN 14**) | **+2.6 pt** |
| **Near camera alone** | 0.970 | 0.970 | tie (classifier-bound) |
| **Far+near fusion** | 0.969 (FP 24, FN 19) | **0.975** (FP 20, FN 14) | **+0.66 pt** |

**Per-game fusion accuracy** (RF-DETR **wins 6, ties 2, loses 1**; FP = miss→make, FN = make→miss):

| game | shots | YOLO fusion | **RF-DETR fusion** | Δ | RF-DETR FP / FN |
|---|---:|---:|---:|---:|---:|
| 29b51d57 | 164 | 0.988 | 0.982 | −0.6 | 3 / 0 |
| 2c490f1a | 118 | 0.958 | **0.983** | +2.5 | 1 / 1 |
| 74c4f686 | 147 | 0.966 | 0.966 | 0.0 | 2 / 3 |
| 8dcb1330 | 167 | 0.970 | **0.976** | +0.6 | 1 / 3 |
| 922bff3b | 137 | 0.985 | **0.993** | +0.8 | 1 / 0 |
| 9eb51980 | 160 | 0.950 | **0.956** | +0.6 | 4 / 3 |
| d0a9faef | 142 | 0.958 | **0.972** | +1.4 | 3 / 1 |
| d446fe8c | 169 | 0.976 | **0.988** | +1.2 | 2 / 0 |
| f66eb3b2 | 160 | 0.963 | 0.963 | 0.0 | 3 / 3 |
| **OVERALL** | **1364** | **0.969** | **0.975** | **+0.66** | **20 / 14** |

(Near-solo is ~0.97 and detector-invariant on every game; the per-game spread above is driven by the far angle, where RF-DETR is stronger.)

**Findings:**
- **RF-DETR roughly halves far-side false-negatives (37 → 14)** — its cleaner, denser trajectory makes the far "did the ball pass through the hoop" line-intersection more reliable, catching made baskets the YOLO trajectory dropped. That gain carries into the fusion (**fewer FP *and* FN**).
- **The near angle is detector-invariant** — near make/miss is decided by the rim-crop *classifier*; both detectors locate the rim at ~1.0 recall, so the detector doesn't move it.
- **Net: a modest but consistent fusion gain (+0.66 pt) plus a clearly stronger far angle**, on top of a **permissive Apache-2.0 license** (vs YOLO's AGPL-3.0, which is restrictive for commercial/networked deployment).

**Recommendation:** RF-DETR is a viable, **better-licensed** drop-in that **holds or slightly improves** fused accuracy and **meaningfully strengthens the far angle** (valuable when the near view is occluded or down-weighted). The remaining ceiling is still the **depth illusion** (§2) — a geometry problem no detector fixes; the path past it remains capture-side (sync → triangulation, §4–5).

We also re-tested **disagreement-aware fusion** (when the two cameras strongly disagree, trust the more reliable one) — four variants, all leave-one-game-out (max-confidence, asymmetric far→near, LOGO logistic stack, LOGO tuned weight). **None beat the simple mean-blend (0.975)**, and most hurt the weakest games. Reason: the disagreement errors go *both* ways (sometimes far is wrong, sometimes near), and the depth illusion fools *both* cameras *confidently* — so there is no reliable signal for *which* camera to trust. The mean-blend is already optimal; the tie-breaker has to come from new information (3D), not a smarter weighting.

### What triangulation needs (the route to ~99%)

Triangulation — recovering the ball's **true 3D position** from 2+ synced, calibrated cameras — turns "through vs in front of the rim" into a *measurement* and dissolves the depth illusion. It is **not** part of the current fusion (the 0.975 is pure 2D); a prototype exists (`pipeline/triangulate_shot.py`, ~84% standalone) but isn't shipped. To make it production-grade we need, in priority order:

1. **Sub-frame time sync (the blocker).** Current cameras are NTP wall-clock aligned (~0.3–0.5 s), and post-hoc audio/event sync is **σ 30–77 frames** per shot — the median is right, but no single shot is aligned to the **< 1 frame** triangulation needs (a ball at 8 m/s moves ~27 cm per 30 fps frame). → **hardware timecode/genlock**, or a sharp **sync flash/clap at recording**, or **60–120 fps** to shrink per-frame motion.
2. **Linear capture mode (not SuperView).** SuperView is heavily distorted and effectively un-calibratable; Linear is a prerequisite for any lens calibration.
3. **Intrinsic calibration** — each camera's focal length + lens distortion, from a one-time **checkerboard** (or exact GoPro model/mode/resolution). The prototype used FOV *guesses* (FR 73°, NR 92°) → ~11–18 px reprojection error.
4. **Extrinsic calibration (pose)** — accurate **court + rim pixel↔world correspondences** per camera (click/detect 6+ known points) plus the **rim's exact 3D position**.

**What we already have vs. need:** ✅ court measurements (the world-coordinate side of pose) and ⚠️ coarse audio sync are a real head start — but neither replaces the two true unlocks: **capture-time sub-frame sync** and **Linear-mode + checkerboard calibration**. Those two are what move triangulation from ~84% standalone to the measured-depth ~99% path.
