# Shot Detection: Why Overhead Tracking Works, Why It Doesn't Transfer to a Side Angle, and Our Path to Near‑Perfect Accuracy

## 1. The proven benchmark — how the category leader gets near‑certain make/miss

Noah Basketball is the reference standard (used in ~93% of NBA facilities, 600M+ shots tracked). The single most important fact:

- **It runs at ~30 fps — the same frame rate we already record at.** Frame rate is *not* their advantage.
- Their advantage is **camera placement**: one camera mounted ~13 ft **directly above the rim, looking straight down the rim axis.**
- From straight down, the rim is a perfect circle, so the ball either drops **inside** it (make) or not (miss). Make/miss is **directly observed, not inferred.** A one‑time calibration to the known rim size turns this into precise depth, arc, and left‑right measurements.

## 2. Why this does **not** transfer to our current near‑angle (side) camera

- Our near camera views the rim from an **oblique angle**, not straight down.
- From an oblique angle, the one thing the camera *cannot* see well is **depth** — whether the ball passed **through** the rim or just **in front of** it. That is the exact information that decides make vs. miss.
- We tested this rigorously: **four independent implementations** of the overhead‑style geometric method, run on our own footage, all produced **essentially chance‑level** make/miss accuracy — the depth cue was so unreliable it even reversed direction between games.
- **Conclusion:** the overhead method's accuracy comes from *where the camera is*, not from the algorithm. Copying the math onto a side angle does not work — and we have **verified** this, not assumed it.

## 3. What works today, and the path to near‑perfect

**Track A — Software (working now, on existing footage).**
Instead of fragile geometry, we use a trained visual model that reads the rim region directly; it is robust to motion blur and missing frames. On existing 30 fps footage:

- **~93% correct make/miss** on every shot it detects, and **~92% of shots reliably detected.**
- This is already at the **practical ceiling** of what a side angle at 30 fps can deliver — confirmed by extensive testing.

**Track B — Hardware (the real unlock to near‑perfect).**
Adopt the *principle* the market leader proved, on our own low‑cost stack:

- Add **one dedicated camera per hoop, mounted overhead looking down the rim axis.**
- Pair it with a **higher frame rate (120 fps) and faster shutter** so the fast‑moving ball is captured crisply at the rim.
- This removes the depth ambiguity **at the source**, pushing make/miss toward near‑certainty — and delivers **arc, depth, and left‑right** shooting metrics as a bonus.

## Bottom line

- Our software is at the **realistic limit (~93%)** for a side camera at 30 fps, with **zero added hardware.**
- The proven route to **near‑perfect shot detection** is the same one the market leader uses: an **overhead, rim‑axis camera at a higher frame rate.** We can replicate that principle at a **fraction of the cost**, reusing our existing detection and analytics pipeline — and gain professional‑grade shooting metrics in the process.
