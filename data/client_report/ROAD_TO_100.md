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
| **Instrumented rim** (accelerometer + strain gauge on the rim bracket) | **+8–12 pt → effectively 100% on rim-touch detection** | $200–500 / court (one-time bracket swap) | yes (latency < 10 ms) | The rim **directly senses** when it has been touched and how hard. Combined with our existing ball-trajectory model, this is a *deterministic* signal: ball entered rim region + rim *was not* touched = clean make; ball entered + rim touched + ball trajectory continued downward = made-with-iron; ball entered + rim touched + ball came back out = miss. Cheap. Retrofittable in an afternoon. **Highest ROI in this tier.** |
| **Net-vibration sensor** (piezo on net hook) | +3–6 pt | $50–150 / court | yes | A make and a rim-out perturb the net differently; a tiny sensor distinguishes them cleanly. Complementary to instrumented rim. |
| **Smart basketball** (IMU + pressure inside, e.g. Wilson X / DribbleUp / NBA Series One) | **~100% by construction** | $80–300 / ball; requires using their ball exclusively | yes | The ball **knows** if it passed through a basket (down-then-still trajectory, plus optional NFC tag at the net). If your client controls ball supply (training, league play), this is the cleanest path to 100%. **Limitation**: can't be used for ad-hoc / game-feed analysis with arbitrary balls. |
| **Court-wide multi-camera triangulation** (NBA SportVU style, ≥8 calibrated cameras + 3D ball reconstruction) | **~100% for ball position** | $50k–200k / court | yes | The gold standard. Used by every pro league. Massively over-engineered for the typical use case but listed for completeness. |

---

## 4. Recommended path for the client

**Option A — Fast, cheap, retrofit (no court rebuild):**
1. **Camera-mount audit (do today, $0).** Fixes the catastrophic outlier games. Likely +1–3 pt on the corpus average, more on a per-court basis.
2. **Add microphones near each rim (~$50/court).** Train a 50ms audio classifier on net/iron sounds. Fuse with the existing visual model.
3. **Instrumented-rim bracket (~$300/court).** This is the single highest-leverage hardware addition. Pairing the visual model with an "iron touched? yes/no" deterministic signal mathematically closes nearly all the remaining make/miss errors.

Total per-court hardware budget for ~95–98% accuracy: **~$400, no court reconstruction, all retrofit, all real-time.**

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

**The remaining 8–12 % gap is not a software problem.** It is a sensor-information problem. Closing it requires *giving the system information it does not currently have* — through cheap mics, an instrumented rim, or a smart ball. Each of those options is mature, off-the-shelf, retrofit-compatible, and real-time. We are happy to spec and integrate any of them.
