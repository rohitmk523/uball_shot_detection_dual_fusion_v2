#!/usr/bin/env python3
"""Per-game calibration for June games. Reuses calibrate_v4 logic with
game-specific frames. Writes calibration_june_<id>.json.

Usage:
  python pipeline/calibrate_june.py --game-id 72c08cb7
"""
from __future__ import annotations
import argparse, sys, cv2, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from calibrate_v4 import (   # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    WORLD_FLOOR, WORLD_RIM, FR_W, NR_W,
    refine_clicks, yolo_hoop_bbox, fr_rim_landmarks_from_bbox,
    nr_rim_landmarks_from_bbox, solve_pnp, triangulate,
)
from ultralytics import YOLO   # noqa: E402

RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    fr_video = G / f"{gid}_FR_full.mp4"
    nr_video = G / f"{gid}_NR_full.mp4"
    calib_dir = G / "calib"
    calib_dir.mkdir(exist_ok=True)
    fr_frame = calib_dir / f"FR_t30.jpg"
    nr_frame = calib_dir / f"NR_t30.jpg"

    # Extract frames at t=30s if not present
    import subprocess
    for src, out in [(fr_video, fr_frame), (nr_video, nr_frame)]:
        if not out.exists() or out.stat().st_size < 10000:
            print(f"  extracting {out.name}...")
            subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                            "-ss","30","-i",str(src),"-frames:v","1",str(out)],
                           timeout=60, check=True)

    print(f"[load] {fr_frame.name}, {nr_frame.name}")
    fr_img = cv2.imread(str(fr_frame))
    nr_img = cv2.imread(str(nr_frame))

    print("[step 1] refining clicks")
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)
    fr_shifts = [float(np.hypot(refined_fr[k][0]-CLICKS_FR_FLOOR[k][0],
                                 refined_fr[k][1]-CLICKS_FR_FLOOR[k][1]))
                 for k in CLICKS_FR_FLOOR]
    nr_shifts = [float(np.hypot(refined_nr[k][0]-CLICKS_NR_FLOOR[k][0],
                                 refined_nr[k][1]-CLICKS_NR_FLOOR[k][1]))
                 for k in CLICKS_NR_FLOOR]
    print(f"  FR mean shift: {np.mean(fr_shifts):.1f}px max={max(fr_shifts):.1f}px")
    print(f"  NR mean shift: {np.mean(nr_shifts):.1f}px max={max(nr_shifts):.1f}px")

    print("[step 2] YOLO hoop bbox")
    fr_model = YOLO(str(FR_W)); nr_model = YOLO(str(NR_W))
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)
    rim_fr = fr_rim_landmarks_from_bbox(bb_fr)
    rim_nr = nr_rim_landmarks_from_bbox(bb_nr)

    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    for k,(_,w) in rim_fr.items(): world_all[k] = w
    for k,(_,w) in rim_nr.items(): world_all[k] = w
    px_fr = {**refined_fr, **refined_fr_rim, **{k: px for k,(px,_) in rim_fr.items()}}
    px_nr = {**refined_nr, **refined_nr_rim, **{k: px for k,(px,_) in rim_nr.items()}}

    print("[step 3] PnP solve")
    fr = solve_pnp(world_all, px_fr, fov=73)
    nr = solve_pnp(world_all, px_nr, fov=92)
    print(f"  FR: reproj={fr['mean']:.1f}px cam=({fr['cam'][0]:.0f},{fr['cam'][1]:.0f},{fr['cam'][2]:.0f})")
    print(f"  NR: reproj={nr['mean']:.1f}px cam=({nr['cam'][0]:.0f},{nr['cam'][1]:.0f},{nr['cam'][2]:.0f})")

    print("[step 4] cross-check")
    common = sorted(set(px_fr) & set(px_nr) & set(WORLD_FLOOR))
    errs = []
    for k in common:
        X = triangulate(fr['P'], nr['P'], np.array(px_fr[k], float),
                                          np.array(px_nr[k], float))
        wt = np.array(WORLD_FLOOR[k])
        errs.append(float(np.linalg.norm(X - wt)))
    print(f"  mean: {np.mean(errs):.1f}cm  max: {max(errs):.1f}cm")

    out = {
        "FR": dict(K=fr['K'].tolist(), rvec=fr['rvec'].tolist(),
                   tvec=fr['t'].tolist(), cam_cm=fr['cam'].tolist(),
                   reproj_mean=fr['mean']),
        "NR": dict(K=nr['K'].tolist(), rvec=nr['rvec'].tolist(),
                   tvec=nr['t'].tolist(), cam_cm=nr['cam'].tolist(),
                   reproj_mean=nr['mean']),
        "fov": {"FR":73,"NR":92},
        "hoop_bbox": {"FR":bb_fr,"NR":bb_nr},
        "cross_check_mean_cm": float(np.mean(errs)),
        "click_shifts": {"FR_mean":float(np.mean(fr_shifts)),
                         "NR_mean":float(np.mean(nr_shifts))},
        "source_game": gid,
    }
    out_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
