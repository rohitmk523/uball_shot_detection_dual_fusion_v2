#!/usr/bin/env python3
"""SAM3 ball-centroid refinement for the descent phase of a shot.

For each shot:
  1. Load cached samples (frame, fr_px, nr_px, X_cm)
  2. Find apex (max z) — that's where MAKE/MISS verdict is sensitive
  3. For each sample in [apex_idx - 4, apex_idx + 10]:
     a) Seek FR + NR clip to that frame
     b) Run SAM3 with point prompt at cached (cx, cy)
     c) Compute mask centroid -> sub-pixel ball position
  4. Re-triangulate refined points -> new X_cm
  5. Replace those samples in result.json; rewrite verdict via descent_verdict

YOLO bbox centers have ~3-5 px noise. Mask centroids are sub-pixel which
should tighten cross_r by 5-10 cm at the rim plane.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from triangulate_shot import (   # noqa: E402
    calibrate, triangulate, descent_verdict, nr_rebound_check,
    RIM_X, RIM_Y, RIM_Z, FPS,
)

SAM3_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt")
G3 = ROOT / "data/client_report/triangulation_test/game3_3398befc"
CLIPS = G3 / "clips"

WINDOW_BEFORE = 2    # frames before apex
WINDOW_AFTER = 30    # frames after apex — covers full descent through rim


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Sub-pixel centroid of a binary mask via moments."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.sum() < 5:
        return None
    M = cv2.moments(mask)
    if M["m00"] < 1e-3: return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def refine_shot(name: str, sam, fr_cal, nr_cal,
                results_dir: Path, fr_clip: Path, nr_clip: Path,
                verbose: bool = True) -> dict:
    rp = results_dir / f"{name}.json"
    if not rp.exists():
        return dict(name=name, ok=False, reason="no result.json")
    d = json.loads(rp.read_text())
    samples = d.get("samples", [])
    if len(samples) < 6:
        return dict(name=name, ok=False, reason=f"only {len(samples)} samples")

    zs = np.array([s["X_cm"][2] for s in samples])
    apex_idx = int(np.argmax(zs))
    lo = max(0, apex_idx - WINDOW_BEFORE)
    hi = min(len(samples), apex_idx + WINDOW_AFTER + 1)
    target_idxs = list(range(lo, hi))
    if verbose:
        print(f"\n=== {name}  apex_idx={apex_idx}  refining {len(target_idxs)} "
              f"samples [{lo}..{hi-1}]  z_apex={zs[apex_idx]:.0f}cm ===")
        print(f"  pre-refine verdict: {d.get('verdict','')[:80]}")

    capFR = cv2.VideoCapture(str(fr_clip))
    capNR = cv2.VideoCapture(str(nr_clip))
    refined = {}
    t0 = time.time()
    for ti in target_idxs:
        s = samples[ti]
        f = int(s["frame"])
        # FR seek + read
        capFR.set(cv2.CAP_PROP_POS_FRAMES, f)
        okF, imgF = capFR.read()
        capNR.set(cv2.CAP_PROP_POS_FRAMES, f)
        okN, imgN = capNR.read()
        if not (okF and okN): continue

        # YOLO-cached centers as point prompts; SAM3 produces a mask per point
        cxF, cyF = float(s["fr_px"][0]), float(s["fr_px"][1])
        cxN, cyN = float(s["nr_px"][0]), float(s["nr_px"][1])
        # SAM3 with points=[[x,y]] labels=[1] (positive prompt)
        try:
            rF = sam(imgF, points=[[cxF, cyF]], labels=[1], verbose=False)
            rN = sam(imgN, points=[[cxN, cyN]], labels=[1], verbose=False)
        except TypeError:
            # Some Ultralytics versions need points kwarg differently
            rF = sam(imgF, points=[[[cxF, cyF]]], labels=[[1]], verbose=False)
            rN = sam(imgN, points=[[[cxN, cyN]]], labels=[[1]], verbose=False)
        if not rF or rF[0].masks is None or not rN or rN[0].masks is None:
            continue
        mF = rF[0].masks.data[0].cpu().numpy()
        mN = rN[0].masks.data[0].cpu().numpy()
        # Only accept small "ball-sized" masks. The ball is 24cm; in FR it's
        # ~6-12 px diameter (~40-110 px²). In NR it's ~20-40 px diameter
        # (~300-1200 px²). If the mask is huge, SAM3 latched onto a player/net.
        if mF.sum() > 2000 or mF.sum() < 30: continue
        if mN.sum() > 5000 or mN.sum() < 30: continue
        cF = mask_centroid(mF.astype(np.uint8))
        cN = mask_centroid(mN.astype(np.uint8))
        if cF is None or cN is None: continue
        # Sanity: refined centroid must be within ~30 px of YOLO center
        if abs(cF[0]-cxF) > 30 or abs(cF[1]-cyF) > 30: continue
        if abs(cN[0]-cxN) > 40 or abs(cN[1]-cyN) > 40: continue

        # Re-triangulate
        X = triangulate(fr_cal["P"], nr_cal["P"],
                        np.array(cF, float), np.array(cN, float))
        refined[ti] = dict(fr_px=cF, nr_px=cN, X_cm=X.tolist())
    capFR.release(); capNR.release()
    dt = time.time() - t0
    if verbose:
        print(f"  SAM3 done {dt:.1f}s; refined {len(refined)}/{len(target_idxs)} samples")

    # Splice refined samples back
    new_samples = []
    for i, s in enumerate(samples):
        if i in refined:
            new = dict(s)
            new["fr_px"] = refined[i]["fr_px"]
            new["nr_px"] = refined[i]["nr_px"]
            new["X_cm"] = refined[i]["X_cm"]
            new["sam3"] = True
            new_samples.append(new)
        else:
            new_samples.append(s)

    # Re-run descent_verdict
    zs2 = np.array([s["X_cm"][2] for s in new_samples])
    apex_idx2 = int(np.argmax(zs2))
    apex = new_samples[apex_idx2]["X_cm"]
    apex_dxy = float(np.hypot(apex[0] - RIM_X, apex[1] - RIM_Y))
    info, descent_v = descent_verdict(new_samples, apex_idx2, apex_dxy, apex[2])
    # NR rebound check
    if descent_v.startswith("MAKE"):
        post_apex_z_min = min(
            (new_samples[i]["X_cm"][2]
             for i in range(apex_idx2, min(apex_idx2+30, len(new_samples)))),
            default=9999)
        rebound, max_cy, min_cy_after, dt_reb = nr_rebound_check(
            new_samples, apex_idx2)
        if rebound and post_apex_z_min > 150:
            descent_v = (f"MISS (NR-rebound: cy {max_cy:.0f}->{min_cy_after:.0f}, "
                         f"D={max_cy-min_cy_after:.0f}px)")

    new_v = f"{descent_v}  [SAM3-refined: apex r={apex_dxy:.0f}cm, z_peak={apex[2]:.0f}]"
    if verbose:
        print(f"  post-refine verdict: {new_v[:80]}")

    out = dict(d)
    out["samples"] = new_samples
    out["verdict"] = new_v
    out["sam3_refined"] = dict(n_refined=len(refined),
                                target_range=[lo, hi-1],
                                pre_verdict=d.get("verdict",""))
    return dict(name=name, ok=True, old=d.get("verdict",""), new=new_v,
                n_refined=len(refined), out=out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, help="comma-separated shot names")
    ap.add_argument("--results-dir", default=str(G3 / "results"))
    ap.add_argument("--write", action="store_true",
                    help="write refined samples back into result.json")
    args = ap.parse_args()

    names = args.only.split(",")
    print(f"[load] SAM3 (3.2 GB)...")
    from ultralytics import SAM
    t0 = time.time()
    sam = SAM(str(SAM3_W))
    print(f"  SAM3 loaded in {time.time()-t0:.1f}s")

    print(f"[load] calibration...")
    import os
    os.environ["CALIB_SAM3"] = "1"
    fr_cal, nr_cal = calibrate()

    rdir = Path(args.results_dir)
    rows = []
    for name in names:
        fr_clip = CLIPS / f"{name}_FR.mp4"
        nr_clip = CLIPS / f"{name}_NR.mp4"
        if not fr_clip.exists() or not nr_clip.exists():
            print(f"  {name}: missing clip"); continue
        r = refine_shot(name, sam, fr_cal, nr_cal, rdir, fr_clip, nr_clip)
        rows.append(r)
        if r.get("ok") and args.write:
            (rdir / f"{name}.json").write_text(
                json.dumps(r["out"], indent=2, default=str))
    print("\n=== SUMMARY ===")
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['name']:18s}  SKIPPED: {r.get('reason','?')}")
        else:
            print(f"  {r['name']:18s}  refined={r['n_refined']}")
            print(f"    OLD: {r['old'][:80]}")
            print(f"    NEW: {r['new'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
