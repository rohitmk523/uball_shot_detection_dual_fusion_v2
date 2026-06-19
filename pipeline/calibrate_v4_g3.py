#!/usr/bin/env python3
"""Calibration v4 - game-3 variant.

Same logic as calibrate_v4.py, but uses game-3 frames so cornerSubPix
can refine the starting clicks against the new imagery. If the camera
shifted between game-1 calibration day and game-3 recording day, the
refined click positions and resulting PnP solve will track that drift.

Writes calibration_v4_g3.json next to calibration_v4.json.
"""
from __future__ import annotations
import sys, cv2, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

# Reuse everything from calibrate_v4: world coords, starting clicks, helpers.
from calibrate_v4 import (   # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    WORLD_FLOOR, WORLD_RIM,
    FR_W, NR_W,
    K_from_fov, refine_clicks, yolo_hoop_bbox,
    fr_rim_landmarks_from_bbox, nr_rim_landmarks_from_bbox,
    solve_pnp, triangulate,
)
from ultralytics import YOLO   # noqa: E402

G3 = ROOT / "data/client_report/triangulation_test/game3_3398befc"
FR_FRAME = G3 / "calib" / "FR_t6.jpg"
NR_FRAME = G3 / "calib" / "NR_t6.jpg"


def main() -> int:
    print(f"[load] game-3 frames")
    print(f"  FR: {FR_FRAME}")
    print(f"  NR: {NR_FRAME}")
    fr_img = cv2.imread(str(FR_FRAME))
    nr_img = cv2.imread(str(NR_FRAME))
    if fr_img is None or nr_img is None:
        print(f"  ERROR: failed to read frames"); return 1
    fr_model = YOLO(str(FR_W))
    nr_model = YOLO(str(NR_W))

    # Subpixel-refine the game-1 click coordinates against game-3 frames.
    # If a click sits within ~15 px of the true corner, cornerSubPix will
    # snap onto the local gradient peak (court line intersection); if the
    # camera shifted by more than that, the refinement is essentially a
    # no-op and the calibration won't improve.
    print("\n[step 1] subpixel-refining game-1 click coords against game-3 frames")
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    fr_shifts = []
    for k in sorted(CLICKS_FR_FLOOR):
        ox, oy = CLICKS_FR_FLOOR[k]; nx, ny = refined_fr[k]
        d = float(np.hypot(nx-ox, ny-oy)); fr_shifts.append(d)
        print(f"  FR #{k}: ({ox},{oy}) -> ({nx:.1f},{ny:.1f})  shift={d:.1f}px")
    print(f"  FR mean shift: {np.mean(fr_shifts):.1f} px  max: {max(fr_shifts):.1f} px")

    nr_shifts = []
    for k in sorted(CLICKS_NR_FLOOR):
        ox, oy = CLICKS_NR_FLOOR[k]; nx, ny = refined_nr[k]
        d = float(np.hypot(nx-ox, ny-oy)); nr_shifts.append(d)
        print(f"  NR #{k}: ({ox},{oy}) -> ({nx:.1f},{ny:.1f})  shift={d:.1f}px")
    print(f"  NR mean shift: {np.mean(nr_shifts):.1f} px  max: {max(nr_shifts):.1f} px")

    # YOLO hoop bbox in game-3 frames
    print("\n[step 2] YOLO hoop -> 3D-spread rim landmarks (game-3)")
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)
    if bb_fr is None or bb_nr is None:
        print(f"  ERROR: missing hoop bbox  FR={bb_fr}  NR={bb_nr}"); return 1
    print(f"  FR hoop bbox: ({bb_fr['x1']:.0f},{bb_fr['y1']:.0f})-"
          f"({bb_fr['x2']:.0f},{bb_fr['y2']:.0f}) conf={bb_fr['conf']:.2f}")
    print(f"  NR hoop bbox: ({bb_nr['x1']:.0f},{bb_nr['y1']:.0f})-"
          f"({bb_nr['x2']:.0f},{bb_nr['y2']:.0f}) conf={bb_nr['conf']:.2f}")
    rim_fr = fr_rim_landmarks_from_bbox(bb_fr)
    rim_nr = nr_rim_landmarks_from_bbox(bb_nr)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    # Assemble + solve
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    for k, (_, w) in rim_fr.items(): world_all[k] = w
    for k, (_, w) in rim_nr.items(): world_all[k] = w

    px_fr_all = {**refined_fr, **refined_fr_rim,
                 **{k: px for k,(px,_) in rim_fr.items()}}
    px_nr_all = {**refined_nr, **refined_nr_rim,
                 **{k: px for k,(px,_) in rim_nr.items()}}

    print(f"\n[step 3] PnP with {len(px_fr_all)} FR + {len(px_nr_all)} NR landmarks")
    fr = solve_pnp(world_all, px_fr_all, fov=73)
    nr = solve_pnp(world_all, px_nr_all, fov=92)
    print(f"  FR: reproj mean={fr['mean']:.1f}px max={fr['max']:.1f}px")
    print(f"  NR: reproj mean={nr['mean']:.1f}px max={nr['max']:.1f}px")
    print(f"  FR cam: X={fr['cam'][0]:+7.0f}cm Y={fr['cam'][1]:+7.0f}cm "
          f"|Z|={abs(fr['cam'][2]):.0f}cm  (h={abs(fr['cam'][2])/30.48:.1f}ft)")
    print(f"  NR cam: X={nr['cam'][0]:+7.0f}cm Y={nr['cam'][1]:+7.0f}cm "
          f"|Z|={abs(nr['cam'][2]):.0f}cm  (h={abs(nr['cam'][2])/30.48:.1f}ft)")

    # Cross-check on shared floor landmarks
    common = sorted(set(px_fr_all) & set(px_nr_all) & set(WORLD_FLOOR))
    errs: list[float] = []
    print(f"\n[step 4] triangulation cross-check on {len(common)} floor landmarks:")
    for k in common:
        X = triangulate(fr['P'], nr['P'], np.array(px_fr_all[k], float),
                                          np.array(px_nr_all[k], float))
        wt = np.array(WORLD_FLOOR[k])
        e = float(np.linalg.norm(X - wt))
        errs.append(e)
        print(f"    #{k:>2}  truth ({wt[0]:.0f},{wt[1]:.0f},{wt[2]:.0f}) -> "
              f"({X[0]:+7.0f},{X[1]:+7.0f},{X[2]:+6.0f})  err {e:5.1f}cm")
    if errs:
        m = float(np.mean(errs))
        print(f"  MEAN 3D cross-check error: {m:.1f}cm  ({m*0.394:.1f}in)")

    # Save
    out = {
        "FR": {"K": fr['K'].tolist(), "rvec": fr['rvec'].tolist(),
               "tvec": fr['t'].tolist(), "cam_cm": fr['cam'].tolist(),
               "reproj_mean": fr['mean']},
        "NR": {"K": nr['K'].tolist(), "rvec": nr['rvec'].tolist(),
               "tvec": nr['t'].tolist(), "cam_cm": nr['cam'].tolist(),
               "reproj_mean": nr['mean']},
        "fov": {"FR": 73, "NR": 92},
        "hoop_bbox": {"FR": bb_fr, "NR": bb_nr},
        "refined_px": {"FR": refined_fr, "NR": refined_nr},
        "cross_check_mean_cm": float(np.mean(errs)) if errs else None,
        "source_game": "3398befc",
        "click_shifts": {"FR_mean_px": float(np.mean(fr_shifts)),
                         "NR_mean_px": float(np.mean(nr_shifts))},
    }
    out_path = ROOT / "data/client_report/triangulation_test/calibration_v4_g3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
