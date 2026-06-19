# Noah-Style Hardware Blueprint: Pushing Shot Detection From ~95.5% (Software) Toward ~99%+ (Hardware)

**Audience:** uball engineering + client decision-makers
**Author:** dual-fusion v2 team
**Date:** 2026-05-23
**Status:** proposal / reference

> TL;DR — Our 2D software model on existing game footage tops out around
> **95.5%** make/miss accuracy (the "software ceiling"). The remaining ~4–5%
> is dominated by *depth ambiguity*: from oblique game angles we cannot always
> tell whether the ball passed **through** the rim or just **in front of / behind**
> it. Noah Basketball does not beat this with smarter software — it beats it with
> **camera placement and calibration**. By adding **one calibrated camera per
> hoop, mounted high and looking down the rim axis**, the "through vs in-front"
> question becomes directly observable, and make/miss approaches certainty. The
> same camera gives us **arc / entry-angle / depth / left-right** tracking for
> free. This document explains what Noah uses, *why* it works, and exactly how
> we adopt the principle **without** their per-hoop cost or proprietary lock-in.

---

## 1. The problem we are actually solving

Our system (`uball_shot_detection_dual_fusion_v2`) classifies every shot as
**make or miss** from **4 GoPro HERO12 game cameras** at oblique angles:

| Camera | Role |
|--------|------|
| FL / FR | **Far** side, left/right of court |
| NL / NR | **Near** side, left/right of court |

Current best model (angle-aware fusion on `far_v16` detector tracks):

| Metric | Held-out test | LOGO (leave-one-game-out) |
|--------|---------------|---------------------------|
| Accuracy | **0.955** | 0.949 |
| Precision | 0.909 | 0.912 |
| Recall | 0.990 | 0.982 |

We are validating this ceiling right now on **5 fresh, never-seen games**.

### Why we can't software our way to ~100%
The residual errors are **not** detector misses or labeling noise — they are a
**geometry problem**:

- A ball passing **in front of** the rim (near camera) looks, in 2D, almost
  identical to a ball passing **through** the rim → false "make" (depth illusion / parallax).
- A clean **swish** sometimes shows little rim interaction → the model hesitates.
- Oblique angles compress the vertical axis, so "did it drop through the hoop
  plane" is inferred, never measured.

No amount of fusion fully removes this, because **the information is not in the
pixels** of an oblique 2D view. This is the wall Noah's hardware is designed
around.

---

## 2. What Noah Basketball actually uses

Noah is the category leader in shot tracking (≈28 NBA teams, 200+ NCAA, 1,000+
high schools; ~498M shots tracked). Confirmed specifics:

| Aspect | Noah |
|--------|------|
| **Sensor** | Dedicated camera(s); **Noah Pro uses multiple 4K cameras** |
| **Placement** | Mounted **on / above the backboard**, a sensor **~13 ft above the basket** looking down at the rim |
| **Frame rate** | **~30 fps** capture of ball position |
| **Calibration** | Per-hoop; everything normalized to the **shooter's perspective** ("front of rim" = point nearest the shooter) — patented calibration method |
| **Outputs (3 measurements)** | **Left-Right** (−9…0…+9), **Depth** (0=front rim … 18=back rim, ideal ~11), **Arc / entry angle** (most jumpers 35–55°, ideal ~45°) |
| **Feedback** | **Real-time verbal** callout of depth/angle |
| **Extras** | Auto-tags shots to players (no wearables/special ball), 3-sec clip per shot, Noahlytics dashboard, Rim Maps, API |
| **Products** | Noah Pro, Noah Backboard, Noah Broadcast, HOOPS app |

### The single most important fact
Noah runs at **the same ~30 fps we already record at.** Frame rate is **not**
their advantage (this answers the earlier "should we go 30→60 fps?" question —
it helps marginally, but it is **not** what gets Noah to near-100%). Their
advantage is **(a) one camera looking down the rim axis** and **(b) per-hoop
calibration to the rim plane.** That combination *removes the depth ambiguity
that caps our 2D model.*

---

## 3. Why their hardware beats our software ceiling

Think of make/miss as one question: **did the ball pass through the rim circle
(the "hoop plane")?**

- **Our oblique game cameras** see the rim circle edge-on / squashed. The ball's
  path crosses *near* it; "through vs in-front-of vs behind" differ by a few
  pixels and depend on depth we can't measure. → ambiguity → ~4–5% error.
- **A camera looking down the rim axis** (Noah's placement) sees the rim circle
  as an actual **circle**, and the ball either lands **inside that circle and
  descends** (make) or **outside it** (miss). Depth ambiguity collapses because
  we are now looking *along* the very axis that was ambiguous.

With calibration, that camera also yields the 3 quality metrics directly:
- **Left-Right**: the ball's lateral offset across the calibrated rim diameter.
- **Depth**: where along the front-back rim axis the ball crosses the plane.
- **Arc / entry angle**: fit the ball's trajectory over the last frames before
  the plane and take the velocity vector's angle to horizontal.

This is why Noah achieves **near-100% on the in/out question and precise arc/
depth/L-R**: not better ML, **better-posed geometry.**

---

## 4. How we adopt the principle — without Noah's cost or lock-in

We do **not** need to buy Noah (per-hoop dedicated rig + subscription conflicts
with our **low-cost** constraint). We replicate the *winning idea* — a
calibrated rim-axis camera — on our own stack.

### 4.1 The "rim-cam" add-on (core recommendation)
Add **one inexpensive camera per hoop**, mounted **high and behind/above the
backboard, angled down at the rim**, approximating a top-down view of the rim
circle.

| Spec | Recommendation | Why |
|------|----------------|-----|
| Camera | 1080p–4K action cam or fixed IP/USB cam, **60 fps if cheap, 30 fps min** | We already own GoPro HERO12s; one extra per hoop is low-cost |
| Mount | Behind backboard, ~12–13 ft, downward tilt framing the **whole rim circle** | Looks *down the rim axis* → removes depth illusion |
| Lens | Wide enough to keep rim + ~2 ft above it in frame | Need approach trajectory for arc |
| Sync | Same NTP wall-clock sync we already use across the 4 GoPros | Aligns rim-cam events to game-cam plays |
| Power/data | PoE or battery + local capture to the Jetson Nano we already deploy | Reuses existing capture infra |

**Cost order of magnitude:** ~1 action cam + mount per hoop (~$300–500 one-time),
vs Noah's installed, subscription rig. No new servers — inference reuses our
existing AWS GPU pipeline.

### 4.2 Keep the 4 game cameras
The rim-cam answers **make/miss + arc**. The 4 oblique game cameras still do
what they're best at and what a single rim-cam can't:
- **Player attribution** (who shot it) and **play context** (catch-and-shoot vs
  on-the-move) — like Noah's auto-tagging, but from footage we already have.
- **Highlight/broadcast angles** and the existing make/miss model as a
  **redundant cross-check**.

### 4.3 Fusion logic (where the accuracy comes from)
```
final_make_miss =
    rim_cam_decision           # AUTHORITY when rim-cam confidence is high
    if rim_cam_conf >= τ
    else angle_aware_software   # FALLBACK: current 0.955 model on 4 game cams
```
- Rim-cam is the **authority** for in/out (it removes the ambiguity).
- The 4-camera software model is the **fallback** when the rim-cam is occluded
  (rare: a defender's hand, net occlusion) — exactly the cases our software
  already handles at 95.5%.
- Net effect: errors only survive when **both** the rim-axis view **and** all 4
  oblique views fail simultaneously → expected accuracy **~99%+**.

---

## 5. Arc / entry-angle tracking (the "Noah metrics" on our stack)

From the calibrated rim-cam, per shot:

1. **Calibrate once per hoop:** compute a homography mapping rim-cam pixels to
   real rim-plane coordinates (rim diameter = 18 in known scale). Define
   "front of rim" = side nearest the shooter (from the play's `angle` LEFT/RIGHT
   we already store), matching Noah's shooter-relative convention.
2. **Detect the ball** each frame approaching the rim (reuse our YOLO11n ball
   detector — already trained, already running).
3. **Fit the trajectory** (parabola) over the last ~10–15 frames before the
   ball reaches the hoop plane.
4. **Derive the three metrics:**
   - **Entry angle (arc)** = angle of the fitted velocity vector to horizontal
     at the plane (report vs the 35–55° band, ideal ~45°).
   - **Depth** = crossing point along the front→back rim axis (map to 0–18).
   - **Left-Right** = lateral offset across the rim (map to −9…+9).
5. **Optional real-time feedback** (training mode): we can mirror Noah's verbal
   callout, but this is **secondary** to game make/miss.

This gives the client **the full Noah-style coaching metric set** as a
by-product of the same rim-cam we add for accuracy.

---

## 6. Accuracy expectation & honest framing

| Approach | Make/Miss accuracy | Arc/Depth/L-R | Added cost | Real-time | Notes |
|----------|-------------------|---------------|-----------|-----------|-------|
| **Current software (4 game cams)** | ~95.5% | ✗ (no reliable depth) | $0 | ✓ | Today's system; depth illusion is the wall |
| **+60 fps recording only** | ~95.5–96% | ✗ | ~$0 | ✓ | Frame rate is *not* the bottleneck |
| **+1 calibrated rim-cam/hoop (this proposal)** | **~99%+** | **✓** | low (1 cam/hoop) | ✓ | Removes depth ambiguity; adds Noah metrics |
| **Buy Noah** | ~99%+ (in/out is implicit) | ✓ | high (rig + subscription/hoop) | ✓ | Proven, but cost + proprietary + IP |

**Honest caveat:** we should *measure*, not promise, the exact rim-cam number.
~99%+ is the well-founded expectation because the residual errors we see today
are specifically the depth cases a rim-axis view resolves; but a 1-hoop pilot
must confirm it before we quote a figure to the client.

---

## 7. What Noah is "using to make it nearly 100%" — the precise answer

It is **not** a secret sensor or 1000-fps camera. The recipe is:

1. **Camera on the rim axis** (above/behind backboard, ~13 ft) so the rim is a
   true circle and in/out is *seen*, not inferred.
2. **Per-hoop calibration** to the rim plane (their patented step) so pixel
   positions become real-world depth/left-right/arc in inches and degrees.
3. **Shooter-relative normalization** ("front of rim" = nearest the shooter) so
   every shot is analyzed in one consistent frame.
4. **~30 fps trajectory + parabola fit** for arc and crossing point.
5. **Dedicated, unobstructed view** (one job, one camera) → very low occlusion.

We can reproduce **1–4** directly. **5** we approximate and back up with our
4-camera fallback.

---

## 8. Phased rollout

| Phase | Action | Outcome |
|-------|--------|---------|
| **0** | Finish fresh-games validation of the 0.955 software ceiling | Honest baseline to beat |
| **1** | **1-hoop pilot:** install one rim-cam, calibrate, sync to game cams | Prove the depth-illusion shots flip to correct |
| **2** | Build rim-cam detector + homography + arc/depth/L-R extraction | Noah-style metrics on our stack |
| **3** | Implement rim-cam-authority / software-fallback fusion | Measure combined accuracy (target ~99%+) |
| **4** | Roll out per-hoop; add optional real-time verbal feedback (training) | Product parity with Noah on metrics, lower cost |

---

## 9. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| **Net occlusion** on a clean swish from rim-cam | 4-camera software fallback; mount angle tuned above net line |
| **Calibration drift** if a hoop is bumped | Periodic auto-recalibration from rim circle detection |
| **IP** — Noah holds patents on calibration + KPI methods | Legal clearance before commercializing a calibrated-arc product; our make/miss fusion + reuse of game cams is differentiated, but **review Noah's patents** (e.g. US 12,288,344; 11,305,176; 10,343,015) |
| **Mounting logistics** per venue | Use existing Jetson Nano capture + battery action cams to avoid wiring |
| **Cost creep** | Stay action-cam class; do **not** replicate a full dedicated rig per hoop |

---

## 10. Bottom line for the client

- Our **software** gets you to **~95.5%** on footage you already capture, for
  **$0 extra hardware**, in real time, with no VLM/cloud cost.
- To go **near-100% AND unlock arc/depth/left-right coaching metrics**, the
  proven path is Noah's *idea*, not Noah's *price*: **one calibrated rim-axis
  camera per hoop**, fused as the in/out authority with our existing
  4-camera model as fallback.
- This respects every constraint we set: **real-time, low-cost, no VLM**, and
  reuses our detectors, sync, Jetsons, and AWS pipeline.

*See also:* [`ROAD_TO_100.md`](./ROAD_TO_100.md) (software path + triangulation
analysis) and [`CLIENT_REPORT.md`](./CLIENT_REPORT.md) (current accuracy).
