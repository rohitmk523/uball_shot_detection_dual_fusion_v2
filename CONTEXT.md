# Session Context — Triangulation Pipeline Handoff

**Last updated:** 2026-06-10
**Repo:** `/Users/rohitkale/Cellstrat/GitHub_Repositories/uball_shot_detection_dual_fusion_v2`

This file is the single source of truth for a fresh session. Read top to bottom.

---

## 1. What this project does

3D basketball shot detection by triangulating from two camera angles per court:
- **FR (Far-Right)** — wide-angle camera mounted far from the rim
- **NR (Near-Right)** — wide-angle camera mounted near/above the rim

Both record one half-court. Per-frame YOLO ball detections from each camera get
combined via Direct Linear Transform (DLT) triangulation into a 3D ball
trajectory. Then a descent-verdict rule judges MAKE/MISS.

---

## 2. Current accuracy state (validated)

| Game | Shots in scope | Decided accuracy | Notes |
|---|---|---|---|
| Game-1 (5a5f1aae) | 88 | **97.2%** | original calibration source |
| Game-2 (dc5f199e) | 22 | **100.0%** | calibrated cleanly with v4 |
| Game-3 (3398befc) | 75 (RIGHT-basket only) | **92.0%** | calibration drift required refit + SAM3 + L4 hi-res |
| **June 4 games aggregate** | 280 (RIGHT-only) | **87.5% dec / 80.4% ovr** | with DESCENT_BOUNDS+SKIP_NOISY+Y_GUARD (2026-06-10, IN-SAMPLE — needs out-of-sample validation); was 69.1%/54% — see §7 |

**Production blocker:** floor cross-check error ~20 cm on all post-May-29 games
(was 6 cm on game-1). This puts ball positions near the rim within ±20 cm of
truth and breaks the 40 cm MAKE/MISS threshold.

**Established negative results** (do not re-try):
- Raising cross_r threshold from 40 to 50 cm → regresses all games
- Lens distortion (k1) refit on assumed pinhole → k1 ≈ 0, no help
- FR-strong-MAKE ensemble override rule → inconsistent gains/losses
- End-to-end gradient-boosted classifier on 185-shot dataset → 78% on held-out (worse than rules)
- SAM3 ball segmentation on rim crossings → −19 cm cross_r but didn't flip verdicts
- SAM3 rim-center calibration on June games → +0.9 pts (essentially flat)
- Auto-sync detection (offsets 5–25 frame sweep) → REGRESSED 3 of 4 games
- Fisheye lens model swap (cv2.fisheye.solvePnP at 122°) → fails badly without calibrated distortion D vector
- Paint-corner-only auto calibration (`pipeline/calibrate_auto.py`) → game 4692eb2b dropped from 74.5% to 50.0% decided. Root cause: NR has too few landmarks (#3, #4, #10, #20, #34 only). Without #7/#8/#9 centerline points, NR pose is underconstrained. To make this work needs adding 3PT arc apex detection (gives NR #8) + centerline detection (gives #7, #9).
- **Hybrid calibration (AUTO FR + SAM3 NR)** via `pipeline/build_hybrid_calib.py` + `rerun_june_hybrid.py` → essentially a wash with SAM3 baseline. Aggregate over 4 June games (280 shots): HYBRID 68.7% decided / 55.7% overall vs SAM3 70.0% / 54.3%. HYBRID decides 10 more shots but adds 9 false negatives. Per game: 2 wins (72c08cb7, e74164e6), 1 mixed (4692eb2b), 1 loss (454da9cf). No clear winner. Conclusion: improving calibration alone does NOT reach the user's 85-90% target — the bottleneck is elsewhere (likely YOLO detection quality, descent_verdict tuning, or sync).

### Positive results this session

- **SAM3+HYBRID verdict-level ensemble** via `pipeline/ensemble_sam3_hybrid.py` →
  **+1.4 pts decided accuracy (70.0% → 71.4%)** and 6 fewer FP (29 → 23) across
  the 280-shot June set. Mechanism: agreement = high-confidence shared verdict;
  disagreement = downgrade to UND; one-UND = use the other's decision. Per-game
  decided improves on 3 of 4 (72c08cb7 +3.0, e74164e6 +0.3, 454da9cf +2.3;
  4692eb2b −1.2). Surfaces 35 ambiguous shots (12.5%) where the two calibrations
  disagree — useful as a confidence signal for downstream review. Fills 46 UND
  gaps. Not transformative but the best decided accuracy of the three calibration
  variants. **Recommend as the production verdict source going forward.**

- **Global MAKE_R_CM threshold tuning** via `pipeline/threshold_sweep.py` →
  **+1.7 pts decided (67.9% → 69.6%) and +1.4 pts overall** using 35cm vs
  current 40cm. Mechanism: 35cm converts 5 borderline-MAKE FPs to TNs without
  losing TPs (TP gain remains +1). Not stackable with ensemble — both target
  the same boundary cases.

- **Per-class threshold sweep** via `pipeline/_per_class_threshold.py` →
  reveals the BOTTLENECK. Per-class optimum thresholds:
  - FG (n=114): **35cm best** → 70.7% decided
  - 3PT (n=68): **17cm best** → 75.0% decided (very tight)
  - 4PT (n=35): **15cm best** → 72.0% decided
  - FT (n=63): **threshold-INSENSITIVE** → 62.5% ceiling regardless
  Applying per-class optimums aggregates to 70.1% dec / 56.1% ovr — only +0.1
  decided vs production SAM3. Per-class threshold tuning is NOT a meaningful
  lever to break the 70-72% ceiling.

### The real bottleneck (revealed by per-class analysis)

The 70% decided ceiling is driven by structural problems in specific classes,
not calibration or thresholds:

- **3PT MAKE**: only 4 TP out of 19 ground-truth MAKEs (12 FN). The MAKE balls
  cross the rim at r > 35cm because trajectory triangulation accumulates noise
  over long arcs. Threshold tightening makes 3PT MISS detection cleaner but
  can't recover the 12 FNs.
- **4PT MAKE**: only 1 TP out of 9 (6 FN). Same trajectory-noise issue, worse
  on the longest shots.
- **FT MAKE**: 9 TP out of 38 (16 FN). FT-specific issue — free throws drop
  nearly vertically and the descent_verdict model (which assumes parabolic
  descent with apex_r constraint) doesn't fit them well.

To break through 70-72%, the next session needs to tackle:
1. **3PT/4PT trajectory smoothing** — Kalman or RANSAC arc fitting with
   tighter inlier criteria; reject noisy samples that pull cross_r away from
   the true rim crossing
2. **FT-specific descent_verdict** — bypass the parabolic-descent rule; use
   the NR rim crossing detection alone for FTs (since FT trajectory drops
   almost straight)
3. **YOLO recall improvement** — many UND shots come from gappy ball tracks;
   either retrain YOLO with more in-air ball samples or implement a tracker
   that fills gaps between detections

---

## 3. Critical new findings (the recent session)

### 3.1 GoPro Hero 12 cameras are in WIDE mode
- Both FR + NR are GoPro Hero 12
- Recording mode **Wide** (after May 28) / SuperView (before)
- Horizontal FOV ≈ 122°, recorded at 4K then compressed to 1080p
- **We've been calibrating with pinhole assumption FR=73° / NR=92°** — wrong model
- This likely explains a large chunk of the 20 cm drift
- Need to use `cv2.fisheye` or proper Hero 12 lens profile

### 3.2 Shutter speed changed Auto → 1/120
- Auto was likely 1/60 → significant motion blur on fast ball
- 1/120 halves the blur — expected to lift detection rate and decided accuracy by ~3–7 pts

### 3.3 We have empty-court calibration frames for all 4 June games
- Extracted using "mid-FT-gap" trick: a moment between two consecutive LEFT FTs
  (action moves to the other basket; calibrated key is briefly empty)
- See section 5

---

## 4. Repo structure (what is where)

### 4.1 Top-level dirs
```
pipeline/                     # all Python pipeline code
scripts/                      # smaller utility scripts
data/client_report/triangulation_test/   # all calibration + per-game artifacts
data/reports/                 # TRIANGULATION_REPORT.md + measurement_guide/
CONTEXT.md                    # THIS FILE
```

### 4.2 Per-game data layout

For game `<id>` (one of `72c08cb7 454da9cf e74164e6 4692eb2b` for June, or `full_game game2_dc5f199e game3_3398befc` for earlier):

```
data/client_report/triangulation_test/june_<id>/
├── <id>_FR_full.mp4          # full FR video (~4 GB, downloaded from S3)
├── <id>_NR_full.mp4          # full NR video (~4 GB, downloaded from S3)
├── shots_all.json            # all 100+ GT shots with start/end timestamps
├── shots_right.json          # filtered to RIGHT-angle shots (those our cameras see)
├── calib/
│   ├── FR_t30.jpg            # FR calibration frame (empty court — JUST UPDATED 2026-06-06)
│   ├── NR_t30.jpg            # NR calibration frame
│   ├── FR_t30_old.jpg        # previous frame (had players occluding lines)
│   └── empty/                # exploratory empty-court frame candidates
├── clips/
│   ├── shots_pipeline.json   # clip-relative manifest used by triangulate_shot.py
│   ├── <name>_FR.mp4         # per-shot FR clip
│   └── <name>_NR.mp4         # per-shot NR clip
├── results/                  # L1 triangulation outputs (one .json per shot)
├── results_hires/            # L4 hi-res YOLO on UND shots
├── results_sam3/             # results with SAM3-augmented calibration
├── results_hires_sam3/       # hi-res with SAM3 calibration
├── results_sync/             # results with auto-sync offset (FAILED experiment)
├── results_hires_sync/       # hi-res with auto-sync offset
├── ensemble_results.json     # L3 per-camera ensemble verdicts
├── multi_shot_results.json   # L5 multi-shot detector outputs
├── final.json                # current final per-shot verdicts (baseline)
├── final_sam3.json           # SAM3-calibrated final verdicts
├── final_sync.json           # sync-corrected final verdicts (REGRESSED — ignore)
└── sync_detect.json          # raw output of pipeline/auto_sync_detect.py
```

### 4.3 Calibration JSONs (shared, one per source)

```
data/client_report/triangulation_test/
├── calibration_v4.json                       # game-1 original (best: 6cm cross-check)
├── calibration_v4_g3.json                    # game-3 refit (20.3 cm)
├── calibration_v4_sam3_g3.json               # game-3 + SAM3 rim center (4.6 cm rim, 20.3 cm floor)
├── calibration_june_<id>.json                # per-game (4 of them, ~20 cm cross-check each)
├── calibration_june_<id>_sam3.json           # per-game + SAM3 rim (~5-7 cm rim, ~20 cm floor)
└── calibration_v5.json                       # FAILED lens-distortion attempt
```

### 4.4 Key pipeline scripts

| Script | What it does |
|---|---|
| `pipeline/triangulate_shot.py` | THE core L1 triangulation. Loads calibration via env vars (`CALIB_JUNE_SAM3=1`, `CALIB_JUNE_JSON=<path>`, etc.) |
| `pipeline/calibrate_v4.py` | game-1 calibration (uses hardcoded user-clicked landmarks) |
| `pipeline/calibrate_v4_g3.py` | game-3 calibration (cornerSubPix-refines clicks against game-3 frame) |
| `pipeline/calibrate_v4_sam3_g3.py` | game-3 + SAM3 rim center landmark |
| `pipeline/calibrate_june.py` | per-game June calibration (--game-id) — basic |
| `pipeline/calibrate_june_sam3.py` | per-game June + SAM3 rim center (--game-id) |
| `pipeline/extract_game2_clips.py` `_g3.py` | per-game clip extractors (FR/NR sync-corrected) |
| `pipeline/per_camera_verdict.py` `_g2.py` `_g3.py` | L3 ensemble (FR/NR independent verdicts) |
| `pipeline/multi_shot_detect.py` | L5 detects plays with >1 shot attempt (putbacks) |
| `pipeline/sam3_ball_refine.py` | SAM3 per-frame ball-mask centroid refinement |
| `pipeline/final_merge_v3.py` `_g2.py` `_g3.py` | per-game final tiered merge |
| `pipeline/run_june_game.py` | end-to-end runner for a June game (extract + L1 + hi-res + merge) |
| `pipeline/rerun_june_sam3.py` | re-runs L1 + hi-res with SAM3 calibration |
| `pipeline/rerun_june_sync.py` | re-runs with auto-sync offset (FAILED experiment) |
| `pipeline/auto_sync_detect.py` | auto-detects NR sync offset (5..25 frame sweep) — produced regression |
| `pipeline/rescore_descent.py` | re-applies descent_verdict on cached samples with new env-var threshold |
| `pipeline/calibrate_auto.py` | EXPERIMENTAL auto paint-corner calibration. Floor 5cm cross-check but NR rim 38cm; end-to-end −24pts vs SAM3. Needs 3PT arc + centerline detection to be production-ready. |
| `pipeline/build_hybrid_calib.py` | merges AUTO FR + SAM3 NR into `calibration_june_<id>_hybrid.json` |
| `pipeline/calibrate_intrinsics_probe.py` | diagnostic: sweeps pinhole FOV + fisheye D=0; saved intrinsics_probe.json |
| `pipeline/rerun_june_auto.py` | re-runs pipeline with CALIB_JUNE_AUTO; companion to calibrate_auto.py |
| `pipeline/rerun_june_hybrid.py` | re-runs pipeline with CALIB_JUNE_HYBRID; companion to build_hybrid_calib.py |
| `pipeline/ensemble_sam3_hybrid.py` | merges SAM3 + HYBRID verdicts; produces `final_ensemble.json`. Best decided accuracy of the three: 71.4% |
| `pipeline/threshold_sweep.py` | global MAKE_R_CM sweep on SAM3 samples; sweet spot 35cm |
| `pipeline/threshold_sweep_ensemble.py` | combined threshold + ensemble sweep (modest, not stackable) |
| `pipeline/_per_class_threshold.py` | per-class threshold optimums; shows 70-72% ceiling on threshold tuning alone |
| `pipeline/_detect_paint_fr.py`, `_detect_paint_nr.py` | early-iteration paint detectors (kept for reference) |
| `pipeline/_visualize_landmarks.py` | overlays cornerSubPix landmark positions on each June frame for visual QA |
| `pipeline/_debug_auto.py` | per-game debugging: compares detected paint corners to SAM3-projected expected positions |
| `pipeline/extract_features.py` `train_classifier.py` | end-to-end classifier experiment (FAILED) |
| `pipeline/calibrate_v5.py` | lens-distortion attempt (FAILED) |

### 4.5 Key constants / conventions

- Court coordinate origin: rim baseline corner + scoreboard-side sideline
- `RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8` cm (in pipeline/triangulate_shot.py)
- `RIM_RADIUS = 22.86` cm (NBA 18-inch)
- `BALL_RADIUS = 12.0` cm
- `FPS = 29.97`
- MAKE/MISS cross_r threshold = 40 cm (rim_radius + ball_radius + tolerance)
- `WORLD_FLOOR` dict has 10 floor landmarks (line intersections, painted-line corners)
- `SYNC_OFFSET_NR` per game (NR ahead of FR by N frames):
  - game-1: +1
  - game-2: +7
  - game-3: +10.5
  - all 4 June games: assumed +13 (auto-sync detector said this was approximately right)

### 4.6 Calibration loader (in triangulate_shot.py)

The `calibrate()` function reads env vars to pick which calibration to use:

| Env var | Loads |
|---|---|
| (none) | game-1 v4 calibration (default) |
| `CALIB_V5=1` | calibration_v5.json (lens distortion attempt — fails) |
| `CALIB_G3=1` | calibration_v4_g3.json (game-3) |
| `CALIB_SAM3=1` | calibration_v4_sam3_g3.json (game-3 + SAM3 rim center) |
| `CALIB_JUNE=1` + `CALIB_JUNE_JSON=<path>` | a June game's basic calibration |
| `CALIB_JUNE_SAM3=1` + `CALIB_JUNE_JSON=<path>` | a June game's SAM3 calibration (**production for June**) |
| `CALIB_JUNE_AUTO=1` + `CALIB_JUNE_JSON=<path>` | EXPERIMENTAL: auto paint-corner calibration. Currently underconstrained on NR; do not use in production. |
| `CALIB_JUNE_HYBRID=1` + `CALIB_JUNE_JSON=<path>` | EXPERIMENTAL: AUTO FR + SAM3 NR. Wash with SAM3 (+1.4 overall, −1.3 decided). Schema matches SAM3, see `pipeline/build_hybrid_calib.py`. |

---

## 5. Empty-court calibration frames (NEW — IMPORTANT)

We extracted clean empty-court frames from S3 using the mid-FT-gap trick:
between two consecutive LEFT free throws in the same possession, players
briefly leave the calibrated key area.

| Game | Timestamp (sec) | FR/NR frames saved at |
|---|---|---|
| 4692eb2b | t=1102 | `data/client_report/triangulation_test/june_4692eb2b/calib/FR_t30.jpg` / `NR_t30.jpg` |
| 72c08cb7 | t=2273 | `data/client_report/triangulation_test/june_72c08cb7/calib/FR_t30.jpg` / `NR_t30.jpg` |
| e74164e6 | t=390  | `data/client_report/triangulation_test/june_e74164e6/calib/FR_t30.jpg` / `NR_t30.jpg` |
| 454da9cf | t=955  | `data/client_report/triangulation_test/june_454da9cf/calib/FR_t30.jpg` / `NR_t30.jpg` |

Previous calibration frames (with players occluding lines) backed up to
`<game_id>/calib/FR_t30_old.jpg`.

---

## 6. How to fetch data

### 6.1 Supabase (via Claude MCP) — basketball game ground truth

**Project ID:** `mhbrsftxvxxtfgbajrlc` (uball.ai)

Use the MCP tool: `mcp__claude_ai_Supabase__execute_sql`

Example queries you might re-run:

```sql
-- list all games for a date
SELECT id::text, date, video_name, status
FROM public.games
WHERE date BETWEEN '2026-06-01' AND '2026-06-10'
ORDER BY date;

-- get all shots for a game
SELECT id::text AS play_id, classification, start_timestamp, end_timestamp, angle
FROM public.plays
WHERE game_id = '<UUID>'
  AND classification IN ('FG_MAKE','FG_MISS','3PT_MAKE','3PT_MISS','4PT_MAKE','4PT_MISS','FREE_THROW_MAKE','FREE_THROW_MISS')
  AND start_timestamp IS NOT NULL AND end_timestamp IS NOT NULL
ORDER BY start_timestamp;

-- find FTs with short gap to next play (good for empty-court calibration)
SELECT p.start_timestamp, p.end_timestamp, p.angle, p.classification,
       LEAD(p.start_timestamp) OVER (PARTITION BY p.game_id ORDER BY p.start_timestamp) - p.end_timestamp AS gap_to_next
FROM public.plays p
WHERE p.game_id = '<UUID>'
  AND p.classification LIKE 'FREE_THROW%'
  AND p.start_timestamp IS NOT NULL
ORDER BY p.start_timestamp;
```

**Key tables:** `public.games`, `public.plays`. Schema details in the working
memory but `plays.angle` is `'LEFT'` or `'RIGHT'` (convention is per-game —
sometimes LEFT = the FR-watched basket, sometimes the opposite — VERIFY VISUALLY).

### 6.2 S3 (AWS CLI, no MCP) — videos

**Bucket:** `s3://uball-videos-production`

**Path pattern:** `s3://uball-videos-production/court-a/<YYYY-MM-DD>/<game-uuid>/<YYYY-MM-DD>_<game-id-prefix>_<FR|NR|FL>.mp4`

Example listing:
```bash
aws s3 ls s3://uball-videos-production/court-a/2026-06-03/4692eb2b-83bd-4976-a370/
```

**Fast frame extraction without full download** (via presigned URL streaming):
```bash
URL=$(aws s3 presign s3://uball-videos-production/court-a/<date>/<game-prefix>/<file>.mp4 --expires-in 600)
ffmpeg -y -ss <seconds> -i "$URL" -frames:v 1 out.jpg
```
This is how we got the empty-court frames without downloading 4 GB.

**Full download** (only when needed):
```bash
aws s3 cp s3://uball-videos-production/court-a/<date>/<game-prefix>/<file>.mp4 ./local.mp4
```

### 6.2b Four-angle game inventory (compiled 2026-06-10)

**S3 game uuid == annotation-tool `games.id`** (verified — the S3 dir name is
the first 23 chars of the full uuid). Annotation tool = uball.ai project
`mhbrsftxvxxtfgbajrlc` (NOT Uball-core `kjgnswlxsqhayabdpheh`, which is the
consumer app).

62 court-a games have all 4 angles (FR/NR/FL/NL); 53 are real recordings
(>500MB/angle); **29 of those have annotation shots**. Tier-1 = May cluster
(14 games, closest era to the June calibration; CONTEXT §3: cameras barely
moved 05-28 → 06-03):

| date | S3/game id prefix | shots |
|---|---|---|
| 2026-05-22 | fb677f72 / 0ad23700 / 0af6ae02 | 170 / 162 / 148 |
| 2026-05-21 | b7956f81 | 166 |
| 2026-05-19 | 77715f25 / cc1710c4 / cc5deb39 | 207 / 177 / 173 |
| 2026-05-18 | 329042f1 | 149 |
| 2026-05-16 | b3c1f62c / f3e7b25a | 148 / 151 |
| 2026-05-15 | d446fe8c / f66eb3b2 / 49b3873e / 0fa23810 | 169 / 160 / 147 / 146 |

Tier-2 (April, camera era unverified): 04-16 ×4 (164/153/137/160),
04-17 ×3 (147/160/142), 04-18 ×3 (158/136/113), 04-28 ×3 (201/171/167),
04-29 ×2 (179/162). Tier-3 (March): c2a354fe (189), e6fba750 (142).
Jan/Feb 4-angle games are mostly unannotated.

≈2,240 tier-1 shots + ≈2,000 tier-2 — out-of-sample validation data AND
left-basket (FL/NL) doubling. Left-side calibration trick (from Rohit):
pick a play happening on the RIGHT basket → left basket is empty → grab
FL/NL frames for calibration; mirror for right.

### 6.2c Camera sync — MEASURED offsets (Rohit, frame-marked FT clips, 2026-06-10)

**Sync varies per game** — the June uniform +13 assumption was wrong:

| game | NR = FR + | NL = FL + |
|---|---|---|
| val 77715f25 | **+17** | −1 |
| val cc1710c4 | **+11** | 0 |
| val fb677f72 | **+14** (typo-corrected from "22408", pending confirm) | 0 |
| june ×4 | sync clips built (`validation_sync/JUNE_*_FRNR.mp4`), awaiting marks | |

June marks (2026-06-10): 4692eb2b **+7**, e74164e6 **+6** (both re-extracted
+ re-run), 72c08cb7 unmarked (clip playback issues — keep +13),
454da9cf unmarked (keep +13).

Protocol: side-by-side FR|NR clip of a made FT with absolute frame numbers
burned in (`extract_val_clips.py` / `validation_sync/` clips); Rohit marks
the same event in both panels. A ±4-frame sync error (0.13s) at descent
speeds ≈ 40-65cm along-trajectory error — confirmed cause of June depth bias.

**Audio cross-correlation sync (`pipeline/audio_sync_detect.py`) —
INCONCLUSIVE so far:** best-confidence windows match manual marks to ~0.1-1
frame (4692eb2b t=2120 peak 27×: +7.01 vs manual +7), but windows scatter
±5 frames across a game. Two hypotheses: (a) gym music creates false
correlation peaks (beat periodicity), (b) REAL within-game clock drift
between cameras (offset trends upward over game time in 2 of 4 games —
~5 frames / 50 min ≈ 60 ppm, plausible for consumer clocks). Drift-test
clip built: `validation_sync/JUNE_4692eb2b_DRIFTTEST_lateFT_FRNR.mp4`
(late FT @2572.8 in the game Rohit marked +7 @804). If the late mark ≈ +11,
drift is real and sync must be TIME-VARYING (linear model per game,
two marks or robust audio fit); if ≈ +7, music artifacts explain scatter
and best-peak audio sync is usable.

### 6.3 SAM3 model location

Local: `/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt` (3.2 GB)

Load via:
```python
from ultralytics import SAM
sam = SAM("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt")
```

### 6.4 YOLO model locations

```python
FR_WEIGHTS = "/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt"
NR_WEIGHTS = "/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt"
```

Both are YOLO11n (5.5 MB, ~2.6 M params), 2 classes: `0=Basketball, 1=Basketball Hoop`.

---

## 7. Checkpoints / what to do next

### Session 2026-06-08 — what was tried + findings

Built `pipeline/calibrate_auto.py` (auto paint-corner detection) + supporting
diagnostics. Outcomes:

1. **Lens model is NOT the dominant error.** `pipeline/calibrate_intrinsics_probe.py`
   swept FR FOV 60..125° pinhole + fisheye Kannala-Brandt at the predicted
   Hero 12 Wide 122°. Fisheye fails badly without a calibrated D vector
   (cross-check 100-500cm). Pinhole at FOV=75/90 already near-optimal at
   ~25cm cross-check. So the FOV-mismatch hypothesis was wrong — the
   pinhole-with-wrong-FOV pose absorbs distortion via tilt/offset.

2. **Cameras barely moved between game-1 and June.** `pipeline/_visualize_landmarks.py`
   showed cornerSubPix shifts of 5-15 px (well within window). The existing
   CLICKS_FR_FLOOR / CLICKS_NR_FLOOR are essentially valid for June frames.

3. **Paint-corner auto detection works perfectly** for FR (4/4 games) and NR
   (4/4 games). Red HSV mask → largest contour → 4 extreme corners. See
   `pipeline/_detect_paint_fr.py` and `_detect_paint_nr.py`.

4. **BUT auto-PnP underperforms SAM3 baseline.** End-to-end test on game
   4692eb2b: AUTO 50.0% decided vs SAM3 74.5%. NR reproj 38px vs SAM3 18px,
   rim cross-check 38cm vs SAM3 5cm. Root cause: NR has only 5 auto
   landmarks (#3, #4, #10, #20, #34) vs SAM3's 10+, so NR pose is
   underconstrained.

5. **NR cannot see baseline.** Existing SAM3 calibration projects the
   actual baseline (X=2145) to NR image y=2830 (off-screen). The visible
   "bottom of NR paint" at y=1025 is at world X≈1850, not the baseline.
   So NR's BL/BR paint corners are invalid as #5/#6 landmarks.

### Session 2026-06-08 (later) + 2026-06-10 — calibration track CONCLUDED

6. **Hybrid calibration built and tested** (`pipeline/build_hybrid_calib.py`):
   AUTO FR pose (paint corners, reproj 10-12px) + SAM3 NR pose. Preserves
   rim cross-check 5.6-6.4cm. Re-ran all 4 games (`pipeline/rerun_june_hybrid.py`).

7. **All calibration variants land in the same accuracy band** (280 shots):

   | Variant | Decided acc | Overall acc | UND |
   |---|---|---|---|
   | sam3 (baseline) | 70.0% | 54.3% | 63 |
   | hybrid (AUTO FR + SAM3 NR) | 68.7% | 55.7% | 53 |
   | ensemble (sam3 ∩ hybrid) | 71.4% | 53.6% | 70 |

   They trade coverage vs precision; none breaks out. The ~70% decided /
   ~55% overall ceiling on June games is robust to calibration choice.

8. **Threshold tuning is played out** (`pipeline/threshold_sweep.py`,
   `pipeline/_per_class_threshold.py`, run 2026-06-10):
   - Global MAKE_R_CM sweep 25-50cm: best 69.6% dec at 35cm (vs 69.2% at 25).
   - Per-class optima (FG@35, 3PT@17, 4PT@any, FT@15): combined 70.1% dec /
     56.1% ovr — only +0.5pt over global, with small-n overfit risk. SKIP.

9. **FT is the broken class — this is the real signal.** Per-class analysis:
   - FG: 70.7% dec. 3PT: 75.0% dec (but only 4/19 makes detected). 4PT:
     72.0% (1/9 makes). **FT: 62.5% dec, 16/38 makes scored MISS**, and
     completely threshold-INSENSITIVE 15→43cm — FT makes triangulate >43cm
     from rim center. Systematic geometry/sampling error on free throws,
     NOT a threshold problem. Long-range makes (3PT/4PT) are mostly FN/UND too.

### Session 2026-06-10 — FT FN diagnosis → DEPTH-DEGENERACY FIX (+13.9pt overall)

Diagnosed the FT FN cluster (`pipeline/_diagnose_ft_fn.py`). **Root cause:
depth-axis stereo degeneracy.** In the FT/3PT-arc region (ball high, far from
rim) the FR/NR rays are near-parallel → depth (X) is unobservable while
lateral (Y) stays solid. Observed: 3px of pixel motion → X jumping from
1148cm to 64731cm (647 m). Consequences in `descent_verdict`:

1. `apex_idx = argmax(z)` selects the most degenerate garbage sample
   (z up to 97 m) → `apex_dxy > 1000` → "CLEAN MISS" for real makes.
2. The post-apex walker's 250cm-hop/25 m/s teleport guard trips on
   transient depth spikes → "only 1 clean post-apex sample" → UNDECIDED.
3. Near the rim, NR looks steeply down → geometry is well-conditioned →
   that's why all FT TPs came from near-rim gap-stop rules.

**Fix (two env-gated guards in `pipeline/triangulate_shot.py`, default OFF
so game-1/2/3 production verdicts are untouched):**

- `DESCENT_BOUNDS=1` — `sanitize_samples()` drops physically-impossible
  triangulations (X∉(800,2600), Y∉(0,1430), z∉(-50,1100)) before apex
  selection. Only 3.1% of samples drop; 77-100% of every "garbage" shot's
  samples were fine — a handful hijacked the verdict.
- `DESCENT_SKIP_NOISY=1` — walker skips hop-violating samples (depth spikes
  are transient) instead of terminating; terminates only after
  `DESCENT_MAX_SKIPS` (default 8, swept: plateau ≥8) consecutive violations
  (real wrong-ball grabs persist).

**Results on 280 June shots, sam3 calibration, cached rescore
(`pipeline/_ab_descent_robust.py`):**

| Variant | Decided | Overall | UND |
|---|---|---|---|
| baseline | 67.9% | 54.3% | 56 |
| bounds only | 71.4% | 52.5% | 74 |
| **bounds+skip (skips=8)** | **74.3%** | **68.2%** | **23** |

Per class (overall acc): FT 47.6→66.7%, 3PT 52.9→72.1%, 4PT 51.4→82.9%,
FG 59.6→61.4%. Flips: 48 improvements vs 5 regressions.

**Watch-item:** FP grew 23→30 (FG-heavy: 16→22). Previously-dead miss tracks
now walk and some produce make-looking "smooth descent" verdicts → resolved
by the Y-guard below.

### Session 2026-06-10 (later) — FP audit → Y-GUARD (81.3% dec / 74.6% ovr)

Audited all 30 FPs (multi-agent workflow, results in
`triangulation_test/fp_audit_results.json` + manifest in
`fp_audit_manifest.json`; 17/30 deep analyses completed). Quantitative
TP-vs-FP feature study (`pipeline/_tp_fp_features.py`,
`tp_fp_features.json`): **the lateral (Y) axis is the discriminator** —
Y is the well-conditioned stereo axis, and true makes are laterally centered
at the walk end (TP y_off med 5.5cm / p90 21.9) while FPs are off-center
(med 24.7 / p90 59.3). Depth noise can fake radial convergence; it cannot
fake lateral centering.

**`DESCENT_Y_GUARD=1`** (in `descent_verdict`, env-gated default OFF) —
four per-rule lateral-centering guards, thresholds measured per-rule:

| MAKE rule | guard | June FP/TP kill |
|---|---|---|
| gap-stop | y_off_stop > 22cm → fall through | 7+3 FP, 0/45 TP |
| rattled-in | cross_y ≥ 15 AND y_off_stop ≥ 20 → MISS | 6/6 FP, 0/7 TP |
| smooth-descent (rule 5) | y_off_stop > 20 → MISS | 5/7 FP, 0/7 TP |
| clean pass-through | y_off_stop > 40 → MISS | 3/4 FP, 1/27 TP |

**Result (280 shots): TP=85 TN=124 FP=11 FN=37 UND=23 → 81.3% decided /
74.6% overall.** Day's full arc: 67.9%/54.3% → 74.3%/68.2% (bounds+skip)
→ 81.3%/74.6% (+Y-guard). Per class dec: FT 76.8, FG 77.6, 3PT 87.1,
4PT 90.6. Per game dec: 77.9 / 86.4 / 81.0 / 80.6.

**Production flag set:** `DESCENT_BOUNDS=1 DESCENT_SKIP_NOISY=1
DESCENT_MAX_SKIPS=8 DESCENT_Y_GUARD=1` (June games; legacy default-off
path verified bit-identical to pre-change baseline).

**TESTED AND REJECTED (do not re-try):** post-gap reappearance
contradiction guards (ball reappearing at rim height r>60 / below plane
r>45 / on floor after the gap → rim-out) + walk-bridge cap (dt>0.6s).
Net regression: 6 FP killed but 9+ TP killed — after a TRUE make the ball
exits the net, bounces on the floor and rolls away, producing identical
reappearance signatures on sparse tracks. Predicted in advance by the
feature study (reappear_rim_h: TP 23% vs FP 20% — not discriminative).

**Residual 11 FPs** are kinematically make-like (laterally centered,
converging, descending): audit classifications = slow centered rim rattles
that fell off (threshold_edge), 2 putbacks (a90a5cda, 60aefff5 — clip
window contains a subsequent made putback; GT refers to the first shot —
unfixable from cached samples), 1 manufactured gap, 4 unanalyzed (session
limit). Further FP reduction needs denser sampling (hi-res re-extraction
for l1-only shots) or clip-extent fixes, not more verdict rules.

### Session 2026-06-10 (final) — FN audit → lateral-crossing override (87.5% / 80.4%)

FN audit (`fn_audit_manifest.json`): 37 FN = 18 cross_r_too_big + 9 rim-out
bounce + 6 rule-6 default + 3 bounce_back + 1 y-guard edge.

**Finding 1 — cross_r is depth-poisoned, cross_y is not:** 17/18 cross FNs
were laterally centered (cross_y ≤ 24cm) vs only 4/23 cross TNs. In
454da9cf, EVERY crossing (makes and misses alike) sits at dx ≈ −140cm — a
per-game systematic depth bias that makes cross_r meaningless. Fix (under
`DESCENT_Y_GUARD`): **lateral-crossing override** — a crossing with
|cy − rim_y| ≤ 25cm whose tail does not bounce back above the rim plane is
MAKE regardless of cross_r. Recovers 17 FN, flips 4 TN (FT bricks falling
straight down are kinematically inseparable from makes — accepted cost).

**Finding 2 — bounce_back rule only killed makes on June data** (3 FN,
0 TN): dead-center crossings (r=6-20cm) where the net holds the ball at rim
level and the tail-rise check misreads it as a rim-out. Fix: centered
crossings (r<25, cy<20) override the bounce-back tail.

**Final June state (280 shots): TP=105 TN=120 FP=15 FN=17 UND=23 →
87.5% decided / 80.4% overall.** Per game dec: 82.4 / 93.2 / 88.9 / 86.6
(game-2 of June now exceeds the game-3 92% reference). Per class dec:
FT 91.1, FG 83.2, 3PT 91.9, 4PT 87.5.

Day's arc: 67.9/54.3 → 74.3/68.2 (bounds+skip) → 81.3/74.6 (Y-guard) →
**87.5/80.4** (crossing override). All via verdict logic; zero calibration
changes. Legacy no-flag path re-verified bit-identical after every change.

### Session 2026-06-10 (validation) — OUT-OF-SAMPLE VERDICT: 84.2% dec / 80.6% ovr

Ran the full pipeline on 3 never-seen May games (284 right-basket shots,
measured sync +17/+11/+14, fresh per-game SAM3 calibration, production
flag set). Per game (dec/ovr): 77715f25 81.4/75.5 · cc1710c4 87.4/85.6 ·
fb677f72 84.0/81.8. **Aggregate: TP=101 TN=128 FP=17 FN=26 UND=12 →
84.2% decided / 80.6% overall.** June in-sample was 87.5/80.4 → the
guards generalize (≈3pt decided shrinkage, overall holds).

**Sync findings (Rohit's frame marks + audio):**
- Offsets vary per game: val +17/+11/+14; June 4692eb2b +7, e74164e6 +6
  (June's assumed +13 was 6-7 frames wrong → re-extracted + re-run; the
  −140cm crossing bias was sync-driven).
- Within-game clock drift is REAL but small: +7@t=804 → +8@t=2573 in
  4692eb2b (~1 frame/game ≈ 20ppm). Single per-game offset is adequate.
- `pipeline/audio_sync_detect.py`: high-confidence windows cluster at the
  true offset (±1 frame); music creates outlier windows → use a
  density-cluster estimator over many windows, validated against the 5
  manual marks. Sync is now automatable for all held-out games.

**HYBRID with angle-aware fusion (CLIENT_REPORT system, 95.1%) — the
strategic direction:** fusion's only failure is depth-illusion FPs;
triangulation's unique strength is depth. On the 207-shot overlap:
fusion alone 93.7%; triangulation flags 10 of fusion's 12 FPs as MISS
(P(miss|fusion-MAKE ∧ tri-MISS) = 40% vs 10% prior = 4× lift). A naive
unconditional veto fails (−12 TP/+8 FP — tri's own false-MISSes fire on
the same verdict types); the improvement-phase task is a GATED arbiter
(features: fusion prob, tri verdict type + quality metrics — y_off,
cross_y, apex_r, n_samples). Target: 93.7% → ~96-97% on right-basket.
Also: CLIENT_REPORT §5a's "post-hoc sync impossible (σ30-77 frames)"
is OBSOLETE — FT-mark/audio sync gives ±1 frame post-hoc.

**Improvement-phase setup (agreed with Rohit):** dev set = 5
measured-sync games (3 val + 2 re-synced June, ~539 tri shots + fusion
preds); test pool = remaining ~24 annotated 4-angle games, untouched.

### Session 2026-06-11 — ARBITER v1 (hybrid 94.2% → 95.6% on dev overlap)

Expanded the fusion∩tri overlap to **449 shots / 5 games** by running
triangulation on b3c1f62c, cc5deb39, f3e7b25a (full automated chain:
clips → clip-audio sync → re-cut → SAM3 calib → pipeline + repair pass;
scripts: `extract_val_clips.py`, `clip_audio_sync.py`,
`repair_val_game.py`, `run_val_games.py`).

**Clip-audio sync is now production equipment** (`clip_audio_sync.py`):
cross-correlate audio of already-downloaded clip pairs → density cluster
→ +1.3-frame acoustic correction (rim sound reaches NR mic first; venue
constant, calibrated vs 5 manual marks, ±1-2 frames). EVERY new game
measured so far had a wrong default: b3c1f62c +12, cc5deb39 +7,
f3e7b25a +7 (assumed +13). Per-game measurement is mandatory.

**Six-game out-of-sample triangulation aggregate: 83.8% dec / 79.7% ovr
(526 shots)** — stable 79-87% dec across every unseen game.

**Arbiter zones (449-shot overlap):** fus=MISS → 99.6% right (never
touch). fus=MAKE ∧ tri=MAKE → 93.6% makes. ∧ tri=UND → 90.6% makes.
∧ tri=MISS (n=48, 27% real misses) → the gate zone.

**Gate v1 (per-game safe, validated on 2.5× the data it was found on):**
- `fus_prob < 0.99` → +9/−3 → hybrid **95.55%** (max gain)
- `apex_r ≤ 150 ∧ fus_prob < 0.99` → +6/−1 → hybrid **95.32%**
  (86% veto precision — RECOMMENDED: protects fusion's 0.997 recall)
- oracle +13 → 97.1%. Untouchable remainder: 4 good vetoes where fusion
  is ≥0.99-confident-and-wrong; needs stronger tri-confidence features.

Caveat: gates were SELECTED on these 5 games. Honest final number needs
the untouched test pool (~24 games). Artifacts: `arbiter_dataset.json`
(builder: `pipeline/arbiter_dataset.py`).

**Next:** (1) test-pool eval of fusion+gate end-to-end on 2-3 fresh
games; (2) tri-confidence features for the ≥0.99 zone; (3) left-basket
(FL/NL) bring-up doubles all of this.

### Session 2026-06-11 (later) — FROZEN-GATE TEST VERDICT: gate holds (+1.3pt)

Test set: the 3 fusion-TEST games that also have 4 angles (6d601c99,
ee8745f1, c2a354fe — March/April; cameras verified unmoved since March,
drift 0-18px). Frozen pipeline + frozen gates, zero retuning. 236-shot
fusion∩tri overlap (`test_` dirs; fusion preds from
`01_per_shot_predictions_all_test_shots.csv`).

- Audio sync caught big errors again: 6d601c99 **+2**, ee8745f1 +10,
  c2a354fe +7 (vs +13 default). Sync step is mandatory and works.
- Tri standalone (frozen): 91.9/87.7, 79.7/77.6, 84.3/78.9 →
  **84.9% dec / 80.9% ovr aggregate** — matches dev (83.8/79.7).
  9-game out-of-sample total now ~762 shots at ~84/80.
- **Hybrid frozen-gate test result:**
  | | dev (449) | test (236) |
  |---|---|---|
  | fusion alone | 94.21% | 89.83% |
  | hybrid gate A (prob<0.99) | 95.55% (+1.34) | **91.10% (+1.27)** |
  | hybrid gate B (∧apex≤150) | 95.32% | 90.68% (c2a354fe −1.1 ❌) |
  | oracle | 97.10% | 94.49% |
- **Gate A is the production gate**: delta stable ~+1.3pt in AND out of
  sample; per-game never negative on all 8 games (test: +3.1/+1.3/0.0).
  Gate B regressed on c2a354fe → dropped.
- Veto precision drops out-of-sample (75%→59%) — tri false-MISSes are
  the cost driver; the tri-confidence score (roadmap #1) and audio
  rim-clang cue (#2) are the levers to close the +3.4pt oracle gap.

(Fusion's 89.8% here is its held-out-TEST-era baseline on right-basket
shots — not comparable to the 95.1% fresh-validation headline; the
meaningful number is the stable +1.3pt hybrid delta.)

### Session 2026-06-11 (cont.) — HYBRID v2: hi-res escalation (+3.0pt on test)

Root cause analysis of the oracle gap: hi-res triangulation only ran on
UND shots, so gate-zone shots were judged on sparse 640px L1 tracks
(hr_available ~10% in the zone) — tri's false-MISSes were a track-density
artifact. **v2 PIPELINE POLICY: when fusion=MAKE and L1-tri says
confident-MISS, escalate that shot to hi-res (1280px) triangulation;
the hi-res verdict replaces L1's. Gate A (fus_prob<0.99) stays FROZEN
on top.** Hi-res results live in `results_hires_arbiter/` (separate dir;
standalone tri numbers untouched).

| | dev (449) | test (236, frozen) |
|---|---|---|
| fusion alone | 94.21% | 89.83% |
| hybrid v1 (L1 tracks) | 95.55% | 91.10% |
| **hybrid v2 (hi-res escalation)** | 95.55% | **92.80%** |
| oracle (v2 zones) | ~96.9% | 94.07% |

- Hi-res self-corrects most tri false-MISSes BEFORE any veto: dev gate
  zone 48→27 candidates, test 26→15; veto precision 59%→**82%** on test.
- Per-game test: 92.3→95.4, 85.5→89.5, 91.6→93.7 — positive everywhere.
- Hybrid v2 captures ~70% of the oracle gap (v1: 27%).
- Cost: hi-res triangulation on ~6-11% of shots (the disagreements) —
  small, bounded compute.
- Residual ~1.3pt to oracle = fusion-≥0.99-confident-and-wrong +
  remaining tri false-MISSes → audio rim-clang cue is the next lever.

### Caveats / next steps

1. **OVERFIT WARNING (critical):** three rounds of threshold fitting on the
   same 280 shots (bounds, y-guard 22/20/15∧20/40, cross_y≤25, bounce_back
   r<25∧cy<20). Mechanisms are physics-grounded but cutoffs hug this data.
   The 87.5/80.4 number is IN-SAMPLE. Before client-facing claims: validate
   on game-1/2/3 (pull cached results) or the next game batch. Treat
   ~80-85% dec as the honest expectation.
2. **Remaining errors are mostly data-limited, not logic-limited:**
   UND 23 = 13 shots with <3 dual-camera samples + 10 walker-died (sparse
   l1, no hi-res file). FP 15 / FN 17 = centered slow rattles, putbacks in
   clip window, FT bricks falling straight down — per-frame kinematics
   can't separate these. Next data levers: hi-res re-extraction for
   l1-only shots; audio cue (`pipeline/audio_signal.py` exists); NR pixel
   rebound features.
3. Re-run the June pipeline end-to-end with the flag set to regenerate
   final_*.json artifacts for the client report.
4. **Production flag set:** `DESCENT_BOUNDS=1 DESCENT_SKIP_NOISY=1
   DESCENT_MAX_SKIPS=8 DESCENT_Y_GUARD=1`.

### Previously planned (deprioritized): 3PT arc + centerline detection on NR

NOTE (2026-06-10): the value of this is **automation for new venues** (no
manual clicks), NOT accuracy — the hybrid test proved even a perfect NR pose
doesn't lift end-to-end accuracy. Do this when productionizing, not for the
accuracy push.

To make auto-calibration production-viable, need to add more NR landmarks:

1. **3PT arc apex on centerline (#8)** at world (1074, 711, 0).
   - In NR view, the 3PT line is a curved red arc visible above the painted lane
   - Detect via HSV red mask outside the paint region + HoughCircles or fitEllipse
   - The apex on the lane centerline projects to image ~(940, 337) per SAM3 calibration

2. **Lane centerline landmarks (#7, #9)** at world (1370, 708, 0) and (1733, 718, 0).
   - These are virtual centerline points along the lane (image-x ≈ 940)
   - Sample at known image-y positions if there's a detectable centerline marker

3. **Re-run `pipeline/calibrate_auto.py`** with these added landmarks
4. **Target**: NR reproj < 25px, rim cross-check < 10cm
5. **Re-run pipeline** and verify accuracy match or beat SAM3 baseline

### Skip these (already disproven this session)

- Auto paint-corner-only calibration on NR (insufficient landmarks)
- FOV sweep without distortion model (fisheye needs calibrated D)
- Hero 12 Wide 122° FOV swap (fisheye solver fails without D)

### Sub-steps once auto-calibration works

7. **Apply to game-3** for comparison (we know game-3 hit 92% with the old approach — does auto match or beat?)
8. **Re-run all 4 June games' clip extractions** if intrinsics change clip framing
9. **Generate per-shot CSV** with features + verdicts for the 280 shots — for future single-multi-angle weights fine-tuning

### Skip these (already disproven)

- SAM3 ball segmentation (mild effect)
- SAM3 rim center alone (+0.9 pts at most)
- Auto-sync detection (regressed)
- Threshold tuning (global sweep: +0.4pt; per-class: +0.5pt — both noise-level)
- Calibration variant swapping (sam3/hybrid/ensemble all ~70% dec / ~55% ovr)
- Lens distortion k1 with WRONG pinhole assumption
- ML classifier on 185 shots

### Still pending on user side

- **Physical camera position measurements** — 6 numbers (FR x/y/z, NR x/y/z) — see `data/reports/measurement_guide/`
- **Game-1 mode confirmation** — was it Wide or SuperView? Affects whether to include in training data

---

## 8. Common one-liners

```bash
# Sanity-check a game's current accuracy
python3 pipeline/final_merge_g3.py  # game-3
python3 pipeline/final_merge_v3.py  # game-1
python3 pipeline/final_merge_g2.py  # game-2

# Aggregate all 4 June games (current best baseline: 69.1%)
python3 -c "
import json
from collections import defaultdict
tot = defaultdict(int)
for g in ['72c08cb7','454da9cf','e74164e6','4692eb2b']:
    p = f'data/client_report/triangulation_test/june_{g}/final.json'
    for r in json.loads(open(p).read()): tot[r['cat']] += 1
n = sum(tot.values()); dec = tot['TP']+tot['TN']+tot['FP']+tot['FN']
print(f\"TP={tot['TP']} TN={tot['TN']} FP={tot['FP']} FN={tot['FN']} UND={tot['UND']}\")
print(f\"Decided: {tot['TP']+tot['TN']}/{dec} = {100*(tot['TP']+tot['TN'])/dec:.1f}%\")
"

# Re-run a single game with SAM3 calibration
python3 pipeline/rerun_june_sam3.py --game-id 4692eb2b

# Re-build per-game basic calibration
python3 pipeline/calibrate_june.py --game-id 4692eb2b

# Re-build per-game SAM3 calibration
python3 pipeline/calibrate_june_sam3.py --game-id 4692eb2b
```

---

## 9. Things to KNOW (gotchas)

- **Python 3.9 (Xcode)** has cv2 but not sklearn. **Python 3.14 (Homebrew)** has neither YOLO nor cv2 properly. Use `/Applications/Xcode.app/.../python3.9/bin/python3` for everything OpenCV-based. The Xcode python is what `python3` resolves to via PATH in our shell.
- **macOS file paths with spaces** (e.g., "Uball Far Angle") need quoting in shell.
- **The angle field convention in plays varies per game** — sometimes 'LEFT' means the FR-watched basket, sometimes the opposite. Always verify with a sample frame.
- **NR rebound detection unreliable on some games** — NR's view of ball-through-net can look like a rebound depending on camera angle.
- **L1 results dir vs hi-res dir**: the final merge reads BOTH and picks the better verdict. If you re-run L1, also re-run hi-res or accuracy will revert.
- **Final merge reads cached verdict STRINGS** — changing `MAKE_R_CM` env var without re-running L1 has no effect (use `pipeline/rescore_descent.py`).

---

## 10. Open questions for the next session

1. Does `cv2.fisheye.solvePnP` with Hero 12 Wide intrinsics produce cross-check <8 cm? (THE big unknown)
2. Are the empty-court frames actually clean enough for Hough line detection? (Should be — see the 4 frames in `<game>/calib/FR_t30.jpg`)
3. Once auto-calibration works on June games, does it also clean up game-3's residual 4 errors?
4. Can we build a single multi-angle YOLO weights file using the 280 + 185 = 465 shot dataset?

---

## 11. Reports already written

- `data/reports/TRIANGULATION_REPORT.md` — main status report (10 sections, no "client" usage)
- `data/reports/measurement_guide/` — annotated images showing what tape-measure numbers we need from the court (FR camera, NR camera, top-down schematic)

---

**END OF CONTEXT.** Read once, then start building the automated court-line calibrator
using section 7 as the spec.
