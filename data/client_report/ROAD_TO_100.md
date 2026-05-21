# Road to 100% — Why Software Has Maxed Out, and What Closes the Gap

**Companion to:** `CLIENT_REPORT.md`
**TL;DR:** The video-only software approach has hit a real ceiling at ~88–92% accuracy. The remaining 8–12 % is **structurally invisible** to bounding-box computer vision. Closing the gap requires giving the system information that the pixels do not contain — through sensors. This document ranks the realistic options.

---

## 1. Where the software ceiling actually lives

After 7 model iterations over 20 fully-annotated games (3,161 shots) we converged on **0.899 / 0.881 / 0.876** held-out (0.851 weighted across all 20 under fair LOGO). The remaining errors decompose into three structurally distinct buckets:

| Cause | Share of errors | Solvable in software? |
|---|---|---|
| **Annotation ambiguity** on rim in-and-outs | ~25% (~12/49 in test set) | **No** — humans disagree on these too; the *labels* are noisy |
| **Sub-pixel rim-contact information** (swish vs rattle-out) | ~50% (~25/49) | **No, structurally** — a bounding box can't see whether the ball touched iron |
| **Camera-mount / quality degradation** (e.g. 13e1ffad's broken FR) | ~25% game-level outlier risk | **No** — a model can't compensate for a broken sensor |

None of these are model defects. **They are properties of the input signal.** That is why the last five iterations (interactions, post-rim trajectory, calibrated ensemble, hyperparameter search) all plateaued: we have already squeezed essentially all the information out of the available representation (test AUC 0.957).

---

## 2. The information that pixels do not contain

Three physical events fully determine make vs miss:

1. **Did the ball pass through the rim plane?** — Visible in 1080p video, *if* the ball is detected at the right millisecond (often not, because the rim, net, and players occlude it).
2. **Did the ball contact iron on the way?** — Sub-pixel. Indistinguishable in any standard-FPS camera.
3. **Did the net deform in the swish-pattern or the rattle-pattern?** — *Partially* visible (we already extracted net-motion energy — it gave the biggest single jump in iter4) but degrades at far angles and is corrupted by player motion in the foreground.

Every remaining model error is one of these three signals being unrecoverable from the current sensor array.

---

## 3. Hardware options — ranked

### Tier 1 — Cheap, retrofit, immediate wins (no court change)

| Solution | Expected lift | Install cost | Real-time | Notes |
|---|---|---|---|---|
| **Camera-mount audit + stabilization** | **+25 pts on degraded games**, ~0 on good ones | **~$0** (process change) | yes | The catastrophic 0.62 / 0.62 games (`2399cfac`, `13e1ffad`) had broken or vibrating mounts — fix mounts, re-record, accuracy recovers immediately. This is the single highest-ROI fix in the entire document. |
| **Microphone(s) near the rim/backboard** | +1–3 pt | ~$50 / court (lavalier + USB capture) | yes | Net swish, ball-on-iron, ball-on-glass each have **distinct audio signatures** (~50ms window). A tiny audio classifier (real-time, interpretable) runs *parallel* to the visual model; fuse the two. Adds redundancy in noisy gyms but works in the typical recording environment. Cheap, retrofittable, complementary. |
| **Higher frame-rate primary cameras (120 fps → 240 fps)** | +2–5 pt | $300–800 / camera × 4 = $1.2–3.2k / court | yes | The swish/rim-out signal **exists** at the rim — we just sample too rarely. 240 fps doubles temporal resolution and captures the decisive 1–2 frames most current cameras miss. Plug-and-play with the existing pipeline (just re-encode video). |

### Tier 2 — Bigger install, materially higher lift

| Solution | Expected lift | Install cost | Real-time | Notes |
|---|---|---|---|---|
| **Dedicated high-FPS rim camera** (zoomed, behind backboard) | +5–10 pt | ~$500–1.5k / court | yes | A 5th camera *aimed exclusively at the rim region*, 240+ fps, narrow FOV. Removes the resolution and detection-occlusion problems on the rim simultaneously. The pipeline already supports per-angle feature fusion; this is one more angle. |
| **Depth/IR camera at backboard** (e.g. RealSense / Azure Kinect) | +5–7 pt | $300–600 / court | yes | Gives ball Z-position relative to rim plane in **3D**, not 2D bounding boxes. Rim-grazer / in-and-out decisions become geometrically determined: did the ball center cross below the rim plane and continue, or did it bounce above? Trivial classifier on the 3D signal. |
| **Event-based camera at the rim** (Prophesee / iniVation) | +5–10 pt | ~$2–4k / court | yes | Microsecond temporal resolution. Captures the *exact* moment of net contact / iron contact. Overkill in single-court setups; transformative in pro-grade installs. |

### Tier 3 — Instrumented hardware → near-100% by construction

| Solution | Expected lift | Install cost | Real-time | Notes |
|---|---|---|---|---|
| **IR break-beam / proximity sensor through the hoop** (Bhavesh's suggestion) | **→ ~100% on MAKE detection, by construction** | **$20–100 / court** (emitter+receiver pair or a small ring + microcontroller) | yes (latency µs) | **The most direct possible make-detector.** An infrared beam (or short-range proximity ring) spans the hoop opening; a ball passing through **physically breaks the beam** → deterministic "ball went through." This is the canonical hardware solution (used in smart hoops: Huupe, Siq, DribbleUp). Combined with our vision system's shot-attempt detection: *shot attempted + beam broken = MAKE; shot attempted + no break = MISS* — deterministic make/miss. **Cheapest deterministic solution in the entire document.** **Placement caveat:** mount the beam at the **net choke (15–20 cm below the rim)**, not at rim level — otherwise a deep in-and-out (ball dips below the rim then pops back out) can false-trigger. At the net choke the ball must fully pass to break the beam. A 2–3 beam vertical stack + simple debounce eliminates false triggers from players reaching through the net. |
| **Instrumented rim** (accelerometer + strain gauge on the rim bracket) | **+8–12 pt → effectively 100% on rim-touch detection** | $200–500 / court (one-time bracket swap) | yes (latency < 10 ms) | The rim **directly senses** when it has been touched and how hard. Combined with our existing ball-trajectory model, this is a *deterministic* signal: ball entered rim region + rim *was not* touched = clean make; ball entered + rim touched + ball trajectory continued downward = made-with-iron; ball entered + rim touched + ball came back out = miss. Cheap. Retrofittable in an afternoon. Complements the break-beam: the **beam answers "did it go in,"** the **rim sensor answers "did it rattle"** (analytics / shot-quality). |
| **Net-vibration sensor** (piezo on net hook) | +3–6 pt | $50–150 / court | yes | A make and a rim-out perturb the net differently; a tiny sensor distinguishes them cleanly. Complementary to instrumented rim. |
| **Smart basketball** (IMU + pressure inside, e.g. Wilson X / DribbleUp / NBA Series One) | **~100% by construction** | $80–300 / ball; requires using their ball exclusively | yes | The ball **knows** if it passed through a basket (down-then-still trajectory, plus optional NFC tag at the net). If your client controls ball supply (training, league play), this is the cleanest path to 100%. **Limitation**: can't be used for ad-hoc / game-feed analysis with arbitrary balls. |
| **Court-wide multi-camera triangulation** (NBA SportVU style, ≥8 calibrated cameras + 3D ball reconstruction) | **~100% for ball position** | $50k–200k / court | yes | The gold standard. Used by every pro league. Massively over-engineered for the typical use case but listed for completeness. |

---

## 3a. Special note: the 30 → 60 fps recording upgrade

This deserves its own section because it is **the cheapest single lever** the client can pull, and it directly attacks the largest model-fixable error class (long-range swishes that the model currently misses).

**The arithmetic that explains why the model misses swishes:**

| Frame rate | ms / frame | Ball travel / frame | Frames inside the rim region on a swish |
|---|---|---|---|
| **30 fps (current)** | 33 ms | ~25 cm | **1–2, sometimes 0** |
| **60 fps** | 17 ms | ~12 cm | 2–3 reliably |
| 120 fps | 8 ms | ~6 cm | 5–6 |
| 240 fps | 4 ms | ~3 cm | 8+ |

A basketball moves about 25 cm between consecutive frames near the rim at 30 fps, while the rim is only ~45 cm across. **The camera physically cannot see the ball inside the rim for most of a clean swish.** That is why our weakest class is 4PT_MAKE at 73 % — the swishes that should be the *easiest* makes are the ones the model misses, because the input data does not contain them.

**Expected impact of just doubling to 60 fps:**
- 3PT_MAKE accuracy 0.83 → likely **0.88–0.92**
- 4PT_MAKE accuracy 0.73 → likely **0.80–0.88**
- Overall held-out test accuracy 0.89 → likely **0.91–0.93** under fair LOGO
- Net-motion features get twice the temporal resolution → small additional lift

**Costs:**
- **Camera change**: $0 if the existing cameras already support 1080p 60 fps as a recording mode — **the most likely case**. Check with your install vendor first.
- 2× storage and bandwidth (the videos roughly double in size)
- ~2× extraction GPU time
- One-time re-extraction over existing games on AWS to get matched 60 fps tracks: ~$7–10

**Verdict:** if cameras support 60 fps natively, **flip the switch.** It is the highest-leverage single change in this entire document for accuracy, costs essentially nothing, and stacks cleanly with the instrumented-rim install in Tier 3.

---

## 4. Recommended path for the client

**Option A — Fast, cheap, retrofit (no court rebuild):**
1. **Switch existing cameras to 1080p 60 fps (if they support it, $0).** See §3a — directly attacks the swish-misread error class. Expected +2–4 pt accuracy on its own.
2. **Camera-mount audit (do today, $0).** Fixes the catastrophic outlier games. Another +1–3 pt on the corpus average, much more on a per-court basis.
3. **IR break-beam / proximity sensor at the net choke (~$20–100/court) — Bhavesh's suggestion, and the single best value here.** A beam through the hoop opening gives **deterministic, ~100% make detection**. Fused with the vision system's shot-attempt detection, make/miss becomes essentially exact. This one item does more for make/miss accuracy than everything else in Option A combined.
4. *(optional)* **Microphones near each rim (~$50/court)** — net/iron audio classifier, adds redundancy. +1–3 pt.
5. *(optional)* **Instrumented-rim bracket (~$300/court)** — adds "did it rattle?" shot-quality analytics on top of the beam's "did it go in?".

Total per-court budget for **~99 % make/miss accuracy**: **~$100** (60 fps switch + break-beam), or **~$450** with the optional mic + instrumented-rim analytics layer. No court reconstruction, all retrofit, all real-time.

> **Why the break-beam is the headline:** every other approach in this document *infers* make/miss from indirect signals (pixels, sound, vibration) and therefore has residual error. The break-beam **measures the actual event** — the ball passing through the hoop — so it is deterministic by construction. It is also the cheapest. If the client only does one hardware thing, it should be this.

**Option B — Pro-grade install (new court / premium tier):**
1. Tier-1 items above (mounts, mics, 240-fps primary cameras).
2. Dedicated rim camera + depth camera at backboard.
3. Instrumented rim.

Per-court budget: **~$2.5–5k**, accuracy near 99%, fully real-time, still interpretable.

**Option C — Total certainty (only viable if client controls the ball):**
- Smart basketball + Option A. **Effectively 100%** by construction; loses generality (must use that ball).

---

## 5. Honest closing read

We have built and validated the strongest **software-only**, **real-time**, **interpretable**, dual-angle shot-detection pipeline that the current sensor array supports. The model:

- Beats v1 across the board (+4.2 / +8.6 acc/prec).
- Recall is at target on every working-camera game.
- Test AUC 0.957 — the *decision boundary* is essentially as good as it can be.

**The remaining 8–12 % gap is not a software problem.** It is a sensor-information problem. Closing it requires *giving the system information it does not currently have* — and the most direct, cheapest way to do that is an **IR break-beam / proximity sensor through the hoop** (~$20–100/court), which *measures* the ball passing through the rim instead of inferring it from pixels. That single addition takes make/miss detection to ~100% by construction. Mics, an instrumented rim, or a smart ball are complementary options for shot-quality analytics and redundancy. Each is mature, off-the-shelf, retrofit-compatible, and real-time. We are happy to spec and integrate any of them.
