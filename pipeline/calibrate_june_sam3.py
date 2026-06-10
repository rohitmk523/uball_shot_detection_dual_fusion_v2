#!/usr/bin/env python3
"""SAM3-assisted calibration for any June game.

Reuses the calibrate_v4_sam3_g3 logic but parametrized by --game-id.

Adds a single precise rim-centre landmark (SAM3 segments the orange rim ring,
fits an ellipse, takes the centre). Drops rim cross-check from ~20cm to <5cm
without touching floor landmarks. Saves calibration_june_<id>_sam3.json.
"""
from __future__ import annotations
import argparse, sys, cv2, json, time
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
from ultralytics import YOLO, SAM   # noqa: E402

RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8
RIM_RADIUS = 22.86
SAM3_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt")


def segment_rim_ellipse(sam, img, bbox):
    """SAM3 mask + red HSV filter + ellipse fit. Returns ellipse dict + mask."""
    res = sam(img, bboxes=[list(bbox)], verbose=False)
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    sam_mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    if sam_mask.shape[:2] != img.shape[:2]:
        sam_mask = cv2.resize(sam_mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0,100,20], np.uint8),
                                np.array([10,255,255], np.uint8))
    high_red = cv2.inRange(hsv, np.array([160,100,20], np.uint8),
                                 np.array([179,255,255], np.uint8))
    red = cv2.bitwise_or(low_red, high_red)
    ring = cv2.bitwise_and(red, sam_mask)
    ring = cv2.dilate(ring, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 30 or len(contour) < 5: return None
    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
    return dict(cx=float(cx), cy=float(cy), w=float(w), h=float(h),
                angle=float(angle),
                contour_area=float(cv2.contourArea(contour)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--dir-prefix", default="june_",
                    help="game dir prefix under triangulation_test/ "
                         "(june_ for June games, val_ for May validation)")
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/{args.dir_prefix}{gid}"
    fr_frame = G / "calib" / "FR_t30.jpg"
    nr_frame = G / "calib" / "NR_t30.jpg"
    if not fr_frame.exists() or not nr_frame.exists():
        print(f"  ERROR: missing calib frames in {G/'calib'}"); return 1

    print(f"[load] {fr_frame.name}, {nr_frame.name}")
    fr_img = cv2.imread(str(fr_frame))
    nr_img = cv2.imread(str(nr_frame))

    print("[step 1] refining clicks")
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    print("[step 2] YOLO hoop bbox + SAM3 rim segmentation")
    fr_model = YOLO(str(FR_W)); nr_model = YOLO(str(NR_W))
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)

    t0 = time.time()
    print(f"  loading SAM3 ...")
    sam = SAM(str(SAM3_W))
    print(f"  SAM3 loaded in {time.time()-t0:.1f}s")

    fr_ell = segment_rim_ellipse(sam, fr_img,
                                  (bb_fr['x1'], bb_fr['y1'], bb_fr['x2'], bb_fr['y2']))
    nr_ell = segment_rim_ellipse(sam, nr_img,
                                  (bb_nr['x1'], bb_nr['y1'], bb_nr['x2'], bb_nr['y2']))
    if fr_ell is None or nr_ell is None:
        print(f"  ERROR: ellipse fit failed FR={fr_ell} NR={nr_ell}")
        return 1
    print(f"  FR rim ellipse centre: ({fr_ell['cx']:.1f},{fr_ell['cy']:.1f})  contour={fr_ell['contour_area']:.0f}px2")
    print(f"  NR rim ellipse centre: ({nr_ell['cx']:.1f},{nr_ell['cy']:.1f})  contour={nr_ell['contour_area']:.0f}px2")

    # Build world + pixel dicts (same as v4-sam3-g3 — only rim centre landmark #34)
    rim_fr = fr_rim_landmarks_from_bbox(bb_fr)
    rim_nr = nr_rim_landmarks_from_bbox(bb_nr)
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    for k,(_,w) in rim_fr.items(): world_all[k] = w
    for k,(_,w) in rim_nr.items(): world_all[k] = w
    world_all[34] = (RIM_X, RIM_Y, RIM_Z)

    px_fr = {**refined_fr, **refined_fr_rim, **{k: px for k,(px,_) in rim_fr.items()}}
    px_nr = {**refined_nr, **refined_nr_rim, **{k: px for k,(px,_) in rim_nr.items()}}
    px_fr[34] = (fr_ell['cx'], fr_ell['cy'])
    px_nr[34] = (nr_ell['cx'], nr_ell['cy'])

    print(f"[step 3] PnP with {len(px_fr)} FR + {len(px_nr)} NR landmarks (+1 SAM3 rim centre each)")
    fr = solve_pnp(world_all, px_fr, fov=73)
    nr = solve_pnp(world_all, px_nr, fov=92)
    print(f"  FR: reproj={fr['mean']:.1f}px  cam=({fr['cam'][0]:.0f},{fr['cam'][1]:.0f},{fr['cam'][2]:.0f})")
    print(f"  NR: reproj={nr['mean']:.1f}px  cam=({nr['cam'][0]:.0f},{nr['cam'][1]:.0f},{nr['cam'][2]:.0f})")

    common = sorted(set(px_fr) & set(px_nr) & set(WORLD_FLOOR))
    errs = []
    for k in common:
        X = triangulate(fr['P'], nr['P'], np.array(px_fr[k], float),
                                          np.array(px_nr[k], float))
        wt = np.array(WORLD_FLOOR[k])
        errs.append(float(np.linalg.norm(X - wt)))
    rim_X = triangulate(fr['P'], nr['P'], np.array(px_fr[34], float),
                                           np.array(px_nr[34], float))
    rim_err = float(np.linalg.norm(rim_X - np.array((RIM_X, RIM_Y, RIM_Z))))
    print(f"[step 4] cross-check:")
    print(f"  floor (n={len(common)}): mean={np.mean(errs):.1f}cm max={max(errs):.1f}cm")
    print(f"  rim centre: {rim_err:.1f} cm")

    out = {
        "FR": dict(K=fr['K'].tolist(), rvec=fr['rvec'].tolist(),
                   tvec=fr['t'].tolist(), cam_cm=fr['cam'].tolist(),
                   reproj_mean=fr['mean']),
        "NR": dict(K=nr['K'].tolist(), rvec=nr['rvec'].tolist(),
                   tvec=nr['t'].tolist(), cam_cm=nr['cam'].tolist(),
                   reproj_mean=nr['mean']),
        "fov": {"FR":73,"NR":92},
        "hoop_bbox": {"FR":bb_fr,"NR":bb_nr},
        "sam3_ellipse": {"FR":fr_ell,"NR":nr_ell},
        "cross_check_mean_cm": float(np.mean(errs)),
        "rim_center_cross_check_cm": rim_err,
        "source_game": gid,
        "method": "v4 + SAM3 rim ellipse centre",
    }
    out_path = ROOT / (f"data/client_report/triangulation_test/"
                       f"calibration_{args.dir_prefix}{gid}_sam3.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
