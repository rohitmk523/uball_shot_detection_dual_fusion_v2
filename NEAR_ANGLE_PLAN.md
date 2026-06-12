# NEAR-ANGLE-ONLY Shot Detection — Fresh-Start Plan

**Created:** 2026-06-12 · **Owner:** Rohit + Claude
**This file is the COMPLETE context for a new session. Read top to bottom before doing anything.**

---

## 1. Mission

Build a make/miss detector that uses **ONLY the near-angle camera** (NL or NR — the
camera mounted near/above each rim) and pushes toward **~100% accuracy**.

Constraints & context:
- Current footage: **30 FPS** GoPro Hero 12 (SuperView early games, Wide after
  2026-05-28; shutter Auto early, 1/120 after ~June). Optimize for 30fps NOW.
- Client will later install a **dedicated 120 FPS near camera per hoop purely for
  shot detection** — the architecture must scale to that (more frames = more
  evidence; design nothing that assumes 30fps).
- **Real-time, low-cost, no VLM, no cloud calls per shot** (long-standing constraint).
- Single camera ⇒ **no sync, no cross-camera calibration, no triangulation**.
  This kills the three hardest problems of the previous system at the root.

**Existence proof this target is reachable:** HomeCourt (NEX Team) claims ~99%
make/miss with a single phone camera at 30fps, no sensors. See §4.

---

## 2. CRITICAL METHODOLOGY RULE (from Rohit)

**Do NOT read the current near-angle implementation at first.** Fresh eyes.

- Do NOT open: `pipeline/near_fusion_gated.py`, `pipeline/near_noah_compare.py`,
  `pipeline/near_detect_quality.py`, `data/client_report/NEAR_ANGLE_NOAH_RESULTS.md`,
  or any `p2_features*/p3_*` fusion feature code.
- The ONLY numbers to carry as context (for later comparison, not as design
  input): old near-angle-alone logic scored **0.876**, with a Noah-style
  ball-size depth cue **0.902**; the 4-camera fusion system is at **0.951**
  fresh / the hybrid ~**96-97%**. Near-only must eventually beat 0.902 by a lot
  to matter; near-100% is the goal with the 120fps camera.
- AFTER a v0 exists and is measured, THEN read the old implementation and the
  Noah results to compare failure modes.

### Step 0 of the new session — look before you build
Pull 10-20 shot clips (§6 has exact commands), run the existing YOLO
ball+hoop detector on them (weights in §7 — using the detector is fine, it's
not "near-angle logic"), and **watch the detections frame by frame**:
- Where is the ball lost? (entering net? blur? frame-top? occlusion by rim/board?)
- How stable is the hoop box? How does the ball look in the 3-5 frames around
  the rim crossing — for a make vs a miss vs a rim-bounce?
- Build the failure taxonomy FIRST, from your own eyes. Then design.

---

## 3. Why near-angle-only can win (physics of this camera)

The near camera is mounted near/above the rim looking down-court. Facts measured
in previous sessions (these transfer; they are camera facts, not logic):
- The rim is **huge and dead-center** in frame (hoop bbox ~300-900px wide).
  At the rim, geometry is the BEST of any camera — it looks nearly down the
  rim axis (Noah's overhead-camera principle, approximated).
- The ball at the rim is ~170px but **motion-blurred at 30fps/slow shutter**
  (the single biggest measured enemy). 1/120 shutter (post-June) halves it;
  the future **120fps camera mostly eliminates it** — 4× temporal samples.
- 51% of shots have the ball exiting the **frame top** during high arcs —
  irrelevant for rim-crossing classification (the decision happens AT the rim),
  but it kills approaches needing full trajectories. Don't need them.
- Audio track exists on every video (rim clang vs swish) — optional extra
  channel, currently deferred by Rohit.

---

## 4. Web research — what beats hand-crafted geometry (2024-2026 survey)

Researched 2026-06-12. The consistent winning pattern for single-camera
make/miss is **NOT trajectory geometry** (what we did before) but
**learned classification of the rim region over a short time window**:

1. **Rim-crop video classifier (RECOMMENDED CORE).** Detect hoop once (it's
   static per game), crop a fixed window around/below the rim, feed the
   ~1-2s clip of crops around the shot event to a small video classifier
   (made / missed / no-shot). Patent US11400355 does exactly this from
   under-rim images with a 3-class training set; an open-source project pairs
   YOLOv5 hoop detection with a **ResNet50 scoring-action classifier**
   ([isBre/Automated-Basketball-Highlights](https://github.com/isBre/Automated-Basketball-Highlights-with-Deep-Learning)).
   This sidesteps ball-tracking fragility entirely — the net's motion, the
   ball's passage through the cylinder, and occlusion patterns ARE the signal.
2. **HomeCourt existence proof:** single phone camera, 30fps, ~99% claimed,
   real-time on-device — trained end-to-end on hours of footage
   ([CNBC](https://www.cnbc.com/2019/04/17/how-artificial-intelligence-is-making-better-basketball-shooters-with-just-your-iphone.html),
   [VentureBeat](https://venturebeat.com/ai/homecourt-helps-basketball-players-improve-their-game-with-ai/)).
3. **Efficient video backbones:** X3D (lightweight 3D CNN, real-time-friendly)
   or simple frame-stacked 2D CNN (ResNet18 with 8-16 stacked gray crops) —
   start with the simplest; rim crops are small (e.g., 224×224) so even 3D
   models run real-time on modest hardware. MViT if accuracy-hungry later.
4. **Motion-blur augmentation** for the ball/hoop detector measurably helps on
   fast footage ([abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis)).
5. **Trajectory+hoop-intersection heuristics** (e.g.
   [nitinhemaraj/Basketball-shot-detection](https://github.com/nitinhemaraj/Basketball-shot-detection))
   are what we're moving AWAY from — they're the 2D analog of the old logic.

Sources:
- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11400355
- https://github.com/isBre/Automated-Basketball-Highlights-with-Deep-Learning
- https://github.com/abdullahtarek/basketball_analysis
- https://github.com/nitinhemaraj/Basketball-shot-detection
- https://www.homecourt.ai/ · CNBC + VentureBeat articles above
- https://dl.acm.org/doi/10.1145/3448823.3448882 (hoop detection + scoring NN)
- X3D/MViT: https://pmc.ncbi.nlm.nih.gov/articles/PMC11645029/

---

## 5. Proposed roadmap (each phase has a measurable gate)

**Phase 0 — Eyes on data (½ day).** §2 Step 0. Output: failure taxonomy doc +
30 hand-picked example clips covering make/swish, make/rattle, miss/rim-out,
miss/airball, blocked, occluded.

**Phase 1 — Rim-crop dataset builder (1 day).**
- Hoop localization: run the existing hoop detector once per game on an
  empty-court frame → fixed rim box per game (it never moves). Define crop =
  rim box expanded (e.g., 1.5× width, 2.5× height downward to include net +
  below-net region).
- For every GT shot (§6): cut the NL/NR clip (timestamps direct, NO sync
  needed), extract the crop sequence around the shot window, save as compact
  tensors/videos + label. ~37 annotated games × ~160 shots × 2 baskets ≈
  **potential 6,000+ labeled rim-crop clips** (start with 10 games ≈ 1,600).
- Gate: visually verify 50 random crops contain the rim event.

**Phase 2 — v0 classifier (1-2 days).**
- Baseline A: frames-stacked ResNet18 (8-16 crops, gray) — hours to train.
- Baseline B: X3D-S on the same crops.
- Protocol: LOGO over games (the established honest protocol), frozen test
  games NEVER touched during dev (§8).
- Gate: beat old near-alone 0.902 on LOGO. If rim-crop v0 can't beat 0.902,
  stop and rethink (it should — this is the industry-standard approach).

**Phase 3 — iterate on failure modes (ongoing).**
- Hard-negative mining (rebounds, passes near rim, dunks-hanging-on-rim).
- Add ball-detector channel as an auxiliary input (crop + ball heatmap).
- Blur augmentation; class-balanced sampling; temporal jitter.
- Optional: two-stage — cheap shot-event spotter (ball enters rim zone) →
  classifier only on candidate windows (this is the real-time architecture).
- Gate: ≥0.95 LOGO on 30fps before touching anything fancy.

**Phase 4 — 120fps readiness.**
- Keep the temporal input parametric (n frames, stride) so the same model
  retrains on 120fps crops with zero architecture change.
- Expect the blur/undersampling failure classes to mostly vanish; re-run the
  taxonomy and re-train.

**Explicitly NOT in scope:** triangulation, cross-camera sync, court
calibration, audio (deferred by Rohit), VLMs.

---

## 6. Data resources (everything verified working in past sessions)

### 6.1 Ground truth — Supabase MCP
- Tool: `mcp__claude_ai_Supabase__execute_sql`, project **`mhbrsftxvxxtfgbajrlc`**
  (uball.ai — the annotation tool; NOT `kjgnswlxsqhayabdpheh`/Uball-core).
- Tables: `public.games` (id, date, video_name, status), `public.plays`
  (game_id, classification, start_timestamp, end_timestamp, angle).
- Shot classes: `FG_MAKE/FG_MISS/3PT_MAKE/3PT_MISS/4PT_MAKE/4PT_MISS/
  FREE_THROW_MAKE/FREE_THROW_MISS`.
- `plays.angle` ∈ {LEFT, RIGHT} = which basket. **Convention is per-game;
  VERIFY VISUALLY once per game** (extract a frame at a RIGHT-FT and check
  which basket has the action). RIGHT basket → NR camera; LEFT → NL.
- Timestamps are on the shared game timeline; for single-camera clips just cut
  `[start-1s, end+2.5s]` from the NL/NR video directly. Camera-to-camera sync
  is IRRELEVANT here. (Cameras start within ~±0.5s of each other; if a clip
  window looks shifted, pad ±1s more — never chase frame sync.)
- **Results >50KB auto-save to a file** — parse the saved file with
  `re.search(r'<untrusted-data-[^>]*>\\s*(\\[.*\\])\\s*</untrusted-data', ...)`
  on `json.loads(file)['result']`. Add a pad column to force file mode.
- GT caveat: some labels are wrong — see
  `data/client_report/02_candidate_mislabels_high_conf_FPs.csv`. Budget a
  label-cleanup pass; near-100% claims require clean GT.

### 6.2 Videos — S3 (AWS CLI works, creds active: account 840102831548)
- Bucket: `s3://uball-videos-production/court-a/<YYYY-MM-DD>/<game-uuid-23ch>/`
  files `<date>_<uuid23>_<FL|FR|NL|NR>.mp4` (~3-5GB each, 1920×1080@29.97).
- **S3 game uuid prefix == annotation `games.id`** (verified).
- Stream-extract clips without downloading full videos:
  `URL=$(aws s3 presign <s3path> --expires-in 7200)` then
  `ffmpeg -ss <t0> -i "$URL" -t <dur> -c copy out.mp4`.
  **ALWAYS validate clips with ffprobe** (duration>0.5s) — interrupted copies
  leave moov-less corrupt files that pass size checks. Retry failed ones
  (S3 timeouts are routine). Reference implementation (clip cutting only — OK
  to reuse, it contains no near-angle logic): `pipeline/extract_val_clips.py`.
- **Game inventory** (compiled & verified): ~29 annotated 4-angle games +
  more NR/FR-only games. Lists live in `CONTEXT.md` §6.2b. Key sets:
  - May cluster (14 games, 2026-05-15..22): 0ad23700, 0af6ae02, fb677f72,
    b7956f81, 77715f25, cc1710c4, cc5deb39, 329042f1, b3c1f62c, f3e7b25a,
    d446fe8c, f66eb3b2, 49b3873e, 0fa23810
  - April (15 games): 29b51d57, 2c490f1a, 922bff3b, ee8745f1, 74c4f686,
    9eb51980, d0a9faef, 6d601c99, 78acaf33, d186e25e, 2399cfac, 8dcb1330,
    b68967fe, cd045da8, 95d2ea95
  - March: e6fba750, c2a354fe · June: 4692eb2b, 454da9cf, 72c08cb7, e74164e6
    (+June games have local full FR/NR videos in
    `data/client_report/triangulation_test/june_*/`)
  - `data/s3_games_extra.json` maps 15 prefixes → (date, full-uuid).
- Per-shot manifests (`shots_right.json`: name, gt, t_start, t_end, play_id)
  already exist for 23 games under
  `data/client_report/triangulation_test/{june_,val_,test_,train_}<gid8>/`.
  **LEFT-basket manifests don't exist yet** — same Supabase query with
  `angle='LEFT'` doubles the dataset (NL camera).

### 6.3 Existing model weights (reusable tools, not "logic")
- Ball+hoop detector (near): `/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt`
  (YOLO11n, classes: 0=Basketball, 1=Basketball Hoop) — fine to use for rim
  localization + optional ball channel; consider retraining with blur aug later.
- Far detector (if ever needed): `.../Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt`
- SAM3 (heavy, optional): `/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt`

---

## 7. Compute resources

### M4 MacBook (default)
- Python with CV stack: `/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3`
  (cv2 4.13, ultralytics, scipy, numpy). Pandas/sklearn/torch live in
  `/Users/rohitkale/miniconda3/bin/python3` (py3.13). ffmpeg/ffprobe via homebrew.
- Disk is tight (~10-20GB free): delete clips after feature/crop extraction;
  keep only crops + results. ffprobe-validate everything.

### AWS Batch (validated 2026-06-11, ~$0.30/game, ~10min/game)
- Queue `cv-shot-detection-queue` (us-east-1), job def `ffmpeg-nvenc-transcode`
  (g4dn spot GPU, 2 concurrent). In-region S3 extraction is near-instant.
- Pattern: tar a bundle (scripts+weights+manifests) →
  `s3://uball-videos-production/_tmp_tri/bundle.tar.gz`; job bootstrap:
  apt ffmpeg+libgl, pip awscli+ultralytics+scipy (+torch preinstalled in the
  CUDA image), pull bundle, run, `aws s3 cp` results tarball to
  `_tmp_tri/out/` (do NOT use boto3 — not installed; use awscli).
- **Spot reclaims are routine** — submit with `--retry-strategy attempts=3`,
  poll S3 outputs, resubmit stragglers; M4 as backup lane.
- GPU training of the crop classifier can also run there (or any GPU box);
  the dataset of crops is small (GBs) — upload once, iterate fast.

---

## 8. Evaluation protocol (non-negotiable, inherited from this project's hard lessons)

1. **Freeze a test pool BEFORE any modeling**: pick 4-6 games (mix of eras/
   modes, both baskets), never tune on them. Suggest: 6d601c99, ee8745f1,
   c2a354fe (already frozen test games before) + 2 May games + 1 June game.
2. Dev work uses **LOGO (leave-one-game-out)** — per-game accuracy ALWAYS
   reported; a method that wins on average but loses on some game is suspect.
3. Frozen-test evaluations are **one-shot per milestone** — no peeking/iterating.
4. Report decided/overall + per-class (FT/FG/3PT/4PT) + per-game. Watch FT
   and rim-bounce classes — they were the historical sore spots.
5. Negative results get written down (this file or successor) so they're not
   re-tried. The previous project's biggest accelerator was its falsification log.

## 9. Practical gotchas (hard-won, will bite otherwise)

- Clip names: `<play_id8>_<FGM|3PM|4PM|FTM>` (suffix = class family, NOT make/miss).
- 2026-03-17 S3 games are 0-byte stubs; some games miss angles — check sizes.
- A stray ball at the lens / FT shooter can ruin a chosen "empty" frame —
  always eyeball extracted reference frames.
- macOS `/tmp` gets cleaned — keep scripts in the repo, not /tmp.
- Background bash + `&` children die with the parent task — use separate
  background tasks per long runner.
- `bc` rejects leading `+` (use awk for arithmetic in shell).
- Supabase row counts: games ~172, plays ~18k — queries are cheap, go wide.

## 10. First-session checklist

1. Read this file. Do NOT open the excluded files (§2).
2. Freeze the test pool (§8.1). Write it down in this file.
3. Phase 0: pull ~20 clips across classes (`extract_val_clips.py` pattern or
   plain ffmpeg presign), run YOLO ball+hoop, WATCH them, write the taxonomy.
4. Phase 1: build the rim-crop dataset for ~10 dev games (both baskets).
5. Phase 2: train v0 (ResNet18-stack first), LOGO, compare to 0.902.
6. Report to Rohit with per-game numbers + example reels before going further.

---
*Predecessor system (for later comparison only): 4-camera fusion 0.951 fresh /
hybrid+triangulation ~96-97% equivalent; full history in `CONTEXT.md` and
memory. The near-only track is a fresh build on the same data substrate.*
