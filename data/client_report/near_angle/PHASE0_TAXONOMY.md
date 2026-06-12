# Phase 0 — Eyes on Data: Failure Taxonomy (Near-Angle Overhead Rim View)

**Date:** 2026-06-12/13 · **Plan:** `NEAR_ANGLE_PLAN.md` (Phase 0, §2 Step 0)
**Data:** 32 NR shot clips — 24 June era (Wide, 1/120 shutter; games 4692eb2b,
454da9cf, 72c08cb7) + 8 old era (SuperView, auto shutter; 77715f25 May,
29b51d57 April). Frozen test pool untouched.

## 0. Headline findings

1. **The "near" camera is an overhead rim camera.** Mounted high behind the
   basket, looking nearly straight down the rim axis (Noah geometry). The
   rim+net is a large static mesh circle (~420 px) at bottom-center of frame.
   Mount is fixed: identical position across games to <1 cm (old calib data).
   June crop: `x∈[640,1330], y∈[700,1080]`; old era: `x∈[650,1250], y∈[750,1080]`.
2. **GT-blind human-level read of 30fps rim montages: 20/32 (62.5%)** — but
   the errors are *structured, not random*:
   - 5/12 errors = event-capture failures, **all in teal-ball game 454da9cf**
     (windowing anchored on global motion; nothing to classify in the strip).
   - 3-4/12 = **suspected GT label noise** (double-blind reads contradict GT
     with concrete physical evidence): `c04=77715f25_3bfe1b31 (3PT_MISS?)`,
     `c18=77715f25_30521420 (FT_MISS?)`, `c20=29b51d57_ad927eb7 (3PT_MISS?)`,
     `c26=454da9cf_b38d7ebd (4PT_MAKE?)`.
   - Rest = 30fps rattle aliasing/blur + behavioral-aftermath traps.
   - **Clips where ball-through-net or exit-carom physics was visible: 13/13
     correct.** The signal is there; the failures are capture + labels, not
     the view. Rim-crop classifier = GO.
3. **Existing YOLO11n near weights are unusable on this view**: hoop fires on
   ~2% of frames (193 dets / ~8,200 frames); only 20% of ball dets are near
   the rim (FPs on heads/jerseys); **blue/teal ball in 454da9cf → ~0 ball
   dets**. → Detector retrain approved by Rohit (in progress, parallel track).

## 1. Taxonomy (32 clips, ≥1 category each)

| # | Category | n | Example cids | Classifier implication |
|---|----------|---|--------------|------------------------|
| 1 | Clean make (ball through net / sustained net deformation) | 6 | c05 c06 c08 c09 c14 c16 | Easiest class; net-sway + in-circle shrink-darken are localized learnable features. 120fps makes trivial. |
| 2 | Clean miss (exit carom visible, net undisturbed) | 7 | c01 c02 c13 c19 c22 c23 c24 | Easy; conjunction (exit + static net + scramble) robust. Ball-size parallax = unique overhead cue. |
| 3 | Rattle / decisive instant lost to 30fps aliasing+blur | 8 | c03 c12 c15 c20 c26 c28 c29 c30 | **Core 30fps ceiling.** 4 of 12 errors. The 120fps camera attacks exactly this. Until then: integrate weak evidence + abstain path. |
| 4 | Rim occluded by players at critical moment | 6 | c03 c12 c25 c26 c29 c32 | fps does NOT fix. Irreducible single-view residual. Needs post-event net-state evidence in-window + calibrated low confidence. |
| 5 | Ball invisible by color/contrast (camouflage) | 2+5 | c17 c27 (+454da9cf cluster) | Color cannot be load-bearing: add frame-diff/flow channel, hue/gray augmentation, teal balls in detector training. Net dynamics are color-independent. |
| 6 | Event not captured in montage window | 5 | c07 c10 c11 c21 c31 (all 454da9cf) | **Upstream failure, not classifier failure.** Event anchor must use motion INSIDE rim crop (never global). Add explicit `no_event` class. |
| 7 | Behavioral-aftermath traps | 6 | c04 c12 c17 c18 c20 c30 | All wrong "make" calls leaned on calm-aftermath cues. Dead-ball FT misses mimic makes. Keep crop tight + core window short; aftermath = separate ablatable input. |
| 8 | Suspected GT label noise | 4 | c04 c18 c20 c26 | At a ~100% target, label noise dominates. Adjudicate these 4 from source video; double-blind verify eval labels; noise-robust loss. |

## 2. Era comparison (Wide/1-120 vs SuperView/auto)

Raw blind accuracy identical (62.5% both) but composition differs entirely:
- **All 3 old-era errors are confident wrong "makes" on GT misses** — auto-shutter
  smear + dark ball makes roll-out look like drop-through (and all 3 are
  label-noise candidates → era and label quality confounded).
- June errors are pipeline/content (teal-ball cluster, camouflage, rattle).
  **Excluding the no-event cluster, June reads 15/19 = 79% vs old 62.5%.**
- June 1/120 shutter delivers: sharper ball blobs, readable net mesh,
  trackable exit caroms. → Train primarily on June-era; treat old era as
  domain-shift augmentation source; re-adjudicate old-era GT before eval use.

## 3. Risks to ~100% (ranked)

1. **GT label noise** — can't claim (or even measure) ~100% until eval labels
   are adjudicated from source video.
2. **Event localization** — 15.6% of clips had no event in-window; anchor on
   rim-crop-local motion (or audio onset) only.
3. **Teal/low-contrast ball** — entire games silently fail in any
   ball-detection-dependent stage.
4. **30fps rattle aliasing** (physics ceiling; fixed by 120fps camera).
5. **Player occlusion** (not fixed by fps; needs net-state + abstain;
   the residual argument for keeping fusion as fallback).
6. **Behavioral-shortcut learning** (dataset design must prevent).
7. Hoop/ball detector OOD (retrain in progress).
8. Small sample (32 clips) — wide error bars; Phase 1 uses whole-game holdouts.

## 4. Phase 1 spec (locked from these findings)

- **Rim localization:** NOT YOLO. Rim is static per game → locate once per
  game (Hough/median-frame fit or one manual click), freeze crop.
- **Crop:** square, centered on rim, ~2.5-3× rim diameter (≈800-1500 px native
  → 224-320 px), must include net cone below + entry arc above.
- **Event anchor:** motion energy inside rim crop only. Re-extract 454da9cf
  no-event clips with wider windows as the regression test.
- **Window:** anchor −0.5s → +1.0s at **stride 1** (Phase 0 stride-2 lost
  decisive frames). Aftermath +1.0→+2.5s stored separately (ablatable).
- **Labels:** {make, miss, no_event, unreadable} + metadata (occlusion level,
  ball color, era, shot type, rattle flag). Double-blind verify eval labels.
  Adjudicate c04/c18/c20/c26 first.
- **Composition:** oversample FT misses, rattles, teal-ball; split by whole
  game; ≥1 old-era game held out.

## 5. Phase 2 spec (model)

- Small 3D video classifier on rim crops (X3D-S / R(2+1)D-18 / MViT-tiny, or
  TSM-2D for tightest real-time budget), 16-32 frames @224 px.
- **Frame-difference or flow channel** alongside RGB (color-independent).
- Heads: 3-way make/miss/no_event + calibrated confidence/abstain; auxiliary
  net-deformation + ball-heatmap heads (forces physics over aftermath).
- Augment: hue/sat jitter, random grayscale, synthetic motion blur + darkening
  (bridges old era), temporal jitter ±5 frames, small rotations.
- Report per-category (this taxonomy) + per-era + per-game; production metric
  = decided-accuracy + abstain-rate (abstains escalate).

## 6. Artifacts

- Clips/detections/strips/blind set: `data/client_report/near_angle/phase0/`
  (`clips/`, `clips_old_era/`, `detections/`, `annotated/`, `rimstrip/`,
  `blind/`, `blind_map.json`, `blind_results_full.json`, `detector_stats.json`)
- Scripts: `near_v0/phase0_cut_clips.py`, `near_v0/phase0_detect.py`,
  `near_v0/phase0_rimstrip.py`
- Blind protocol: 32 GT-blind readers + independent second read on
  wrong/unsure + synthesis (51 agents total).

## 7. Detector retrain track (parallel, approved 2026-06-12)

Goal: production shot-event spotter + ball channel + rim auto-localization.
Pipeline = far-angle loop (`Training_frameworks/Uball Far Angle/src`):
mine frames (Supabase plays, both angles) → DINO/SAM2-assisted prelabels →
human verify in annotate_server → YOLO11 fine-tune → era/blue-ball gates.
Status & data: `data/near_detector/` (2,857 plays mined across 18 dev games).
Frozen test pool (6 games) excluded from detector training too.
Key adaptations vs far angle: disable REJECT_BALL_INSIDE_HOOP (ball legally
inside hoop bbox at rim moment), static-hoop-box propagation per game/angle,
teal-ball frames mandatory, motion-blur augmentation.
