#!/usr/bin/env python3
"""Game-3 SAM3-assisted calibration.

Workflow:
  1. SAM3 segments the hoop region (YOLO bbox prompt)
  2. HSV color filter isolates the orange rim ring from the SAM3 mask
  3. cv2.fitEllipse on the orange contour -> precise ellipse parameters
  4. Extract 4 cardinal points (top/bottom/left/right of the image-space
     ellipse) -> map to known 3D rim circle points
  5. Re-solve PnP with SAM3-refined rim landmarks + existing floor clicks
  6. Save calibration_v4_sam3_g3.json

The rim is a 3D circle of radius 22.86 cm at z=304.8, centered at
(RIM_X, RIM_Y). Image-space ellipse cardinal points -> 3D circle cardinal
points using a heuristic based on camera orientation.

For FR (mounted low-left, looking up at rim):
  - ellipse TOP    -> rim BACK edge  (-X direction, toward backboard)
  - ellipse BOTTOM -> rim FRONT edge (+X)
  - ellipse LEFT   -> rim's +Y edge  (toward Iverson side wall)
  - ellipse RIGHT  -> rim's -Y edge  (toward scoreboard side)
For NR (mounted above rim looking down):
  - ellipse TOP    -> rim BACK edge  (camera is on +X side, BACK is -X)
  - ellipse BOTTOM -> rim FRONT edge
  - ellipse LEFT   -> rim's -Y edge
  - ellipse RIGHT  -> rim's +Y edge
The image-space mapping is verified by reproj error after PnP solve.
"""
from __future__ import annotations
import sys, cv2, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_v4 import (   # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    WORLD_FLOOR, WORLD_RIM,
    FR_W, NR_W,
    K_from_fov, refine_clicks, yolo_hoop_bbox,
    fr_rim_landmarks_from_bbox, nr_rim_landmarks_from_bbox,
    solve_pnp, triangulate,
)
from ultralytics import YOLO, SAM   # noqa: E402

RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8
RIM_RADIUS = 22.86

G3 = ROOT / "data/client_report/triangulation_test/game3_3398befc"
FR_FRAME = G3 / "calib" / "FR_t6.jpg"
NR_FRAME = G3 / "calib" / "NR_t6.jpg"
SAM3_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/DEMO_UBALL/demo/sam3.pt")


# 3D points on the rim circle for the 4 image-space cardinal points.
# These are mapped per camera based on camera position relative to rim.
FR_CARDINAL_3D = {
    "top":    (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z),   # back of rim (-X)
    "bottom": (RIM_X + RIM_RADIUS, RIM_Y, RIM_Z),   # front of rim (+X)
    "left":   (RIM_X, RIM_Y + RIM_RADIUS, RIM_Z),   # +Y (Iverson side)
    "right":  (RIM_X, RIM_Y - RIM_RADIUS, RIM_Z),   # -Y (scoreboard side)
}
NR_CARDINAL_3D = {
    "top":    (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z),   # back of rim
    "bottom": (RIM_X + RIM_RADIUS, RIM_Y, RIM_Z),   # front of rim
    "left":   (RIM_X, RIM_Y - RIM_RADIUS, RIM_Z),   # -Y
    "right":  (RIM_X, RIM_Y + RIM_RADIUS, RIM_Z),   # +Y
}
# Land-mark IDs we'll assign: 30=top, 31=bottom, 32=left, 33=right, 34=center


def segment_rim_ellipse(sam_model: SAM, img: np.ndarray,
                        bbox: tuple[float, float, float, float]
                        ) -> tuple[dict, np.ndarray] | None:
    """Run SAM3 with bbox prompt, then HSV-filter for orange rim pixels,
    then fit ellipse to the largest orange contour.

    Returns (ellipse_dict, mask) on success; None on failure.
    ellipse_dict has keys: cx, cy, w, h, angle, top, bottom, left, right.
    """
    res = sam_model(img, bboxes=[list(bbox)], verbose=False)
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    sam_mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    # SAM3 may downsample; resize to image size if needed
    if sam_mask.shape[:2] != img.shape[:2]:
        sam_mask = cv2.resize(sam_mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # HSV filter for the DARK RED rim ring. Color samples on game-3 FR/NR
    # frames show H=170-179, S>200, V=30-110 (the rim is a deep red, not
    # orange). Hue wraps so we OR the low-end and high-end:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 100, 20], dtype=np.uint8),
                                np.array([10, 255, 255], dtype=np.uint8))
    high_red = cv2.inRange(hsv, np.array([160, 100, 20], dtype=np.uint8),
                                 np.array([179, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(low_red, high_red)
    # restrict to SAM3 mask, then dilate so a thin rim contour is connected
    ring = cv2.bitwise_and(red, sam_mask)
    ring = cv2.dilate(ring, np.ones((3, 3), np.uint8), iterations=1)
    # morph close to fill the ring contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 30:
        return None
    if len(contour) < 5:
        return None

    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
    # axis-aligned cardinal points of the rotated ellipse:
    # convert angle to radians (cv2.fitEllipse returns degrees, rotation of
    # the MINOR axis). For our use, just take ellipse extremes in image x/y.
    pts = contour.reshape(-1, 2)
    top = tuple(pts[pts[:, 1].argmin()])
    bottom = tuple(pts[pts[:, 1].argmax()])
    left = tuple(pts[pts[:, 0].argmin()])
    right = tuple(pts[pts[:, 0].argmax()])

    info = dict(cx=float(cx), cy=float(cy), w=float(w), h=float(h),
                angle=float(angle),
                top=(float(top[0]), float(top[1])),
                bottom=(float(bottom[0]), float(bottom[1])),
                left=(float(left[0]), float(left[1])),
                right=(float(right[0]), float(right[1])),
                contour_area=float(cv2.contourArea(contour)),
                contour_points=int(len(contour)))
    return info, ring


def main() -> int:
    print("[load] frames + models")
    print(f"  FR: {FR_FRAME}")
    print(f"  NR: {NR_FRAME}")
    fr_img = cv2.imread(str(FR_FRAME))
    nr_img = cv2.imread(str(NR_FRAME))
    if fr_img is None or nr_img is None:
        print(f"  ERROR: failed to read frames"); return 1
    print(f"  loading YOLO FR/NR...")
    fr_model = YOLO(str(FR_W))
    nr_model = YOLO(str(NR_W))
    print(f"  loading SAM3 (3.2 GB) — this can take 20-40s...")
    import time
    t0 = time.time()
    sam = SAM(str(SAM3_W))
    print(f"  SAM3 loaded in {time.time()-t0:.1f}s")

    # Step 1: floor clicks refined against game-3 frames (same as v4_g3)
    print("\n[step 1] subpixel-refine floor clicks against game-3 frames")
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    # Step 2: YOLO hoop bbox in game-3 frames
    print("\n[step 2] YOLO hoop bbox in game-3 frames")
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)
    print(f"  FR hoop bbox: ({bb_fr['x1']:.0f},{bb_fr['y1']:.0f})-"
          f"({bb_fr['x2']:.0f},{bb_fr['y2']:.0f})  conf={bb_fr['conf']:.2f}")
    print(f"  NR hoop bbox: ({bb_nr['x1']:.0f},{bb_nr['y1']:.0f})-"
          f"({bb_nr['x2']:.0f},{bb_nr['y2']:.0f})  conf={bb_nr['conf']:.2f}")

    # Step 3: SAM3 rim ring segmentation + ellipse fit
    print("\n[step 3] SAM3 + HSV orange filter + ellipse fit")
    fr_seg = segment_rim_ellipse(
        sam, fr_img, (bb_fr['x1'], bb_fr['y1'], bb_fr['x2'], bb_fr['y2']))
    nr_seg = segment_rim_ellipse(
        sam, nr_img, (bb_nr['x1'], bb_nr['y1'], bb_nr['x2'], bb_nr['y2']))
    if fr_seg is None or nr_seg is None:
        print(f"  ERROR: ellipse fit failed  FR={fr_seg}  NR={nr_seg}")
        return 1
    fr_ell, fr_ring = fr_seg
    nr_ell, nr_ring = nr_seg
    print(f"  FR ellipse: center=({fr_ell['cx']:.1f},{fr_ell['cy']:.1f}) "
          f"w={fr_ell['w']:.1f} h={fr_ell['h']:.1f} angle={fr_ell['angle']:.0f}°  "
          f"contour={fr_ell['contour_area']:.0f}px²")
    for k in ("top", "bottom", "left", "right"):
        print(f"    FR {k}: ({fr_ell[k][0]:.1f}, {fr_ell[k][1]:.1f})")
    print(f"  NR ellipse: center=({nr_ell['cx']:.1f},{nr_ell['cy']:.1f}) "
          f"w={nr_ell['w']:.1f} h={nr_ell['h']:.1f} angle={nr_ell['angle']:.0f}°  "
          f"contour={nr_ell['contour_area']:.0f}px²")
    for k in ("top", "bottom", "left", "right"):
        print(f"    NR {k}: ({nr_ell[k][0]:.1f}, {nr_ell[k][1]:.1f})")

    # Save debug overlays
    debug_dir = G3 / "calib"
    fr_dbg = fr_img.copy()
    cv2.ellipse(fr_dbg, ((fr_ell['cx'], fr_ell['cy']),
                          (fr_ell['w'], fr_ell['h']), fr_ell['angle']),
                (0, 255, 0), 2)
    for k, col in zip(("top","bottom","left","right"),
                       [(255,0,0),(255,128,0),(0,128,255),(0,0,255)]):
        cv2.circle(fr_dbg, (int(fr_ell[k][0]), int(fr_ell[k][1])), 6, col, 2)
    cv2.imwrite(str(debug_dir / "FR_sam3_ellipse.jpg"), fr_dbg)
    nr_dbg = nr_img.copy()
    cv2.ellipse(nr_dbg, ((nr_ell['cx'], nr_ell['cy']),
                          (nr_ell['w'], nr_ell['h']), nr_ell['angle']),
                (0, 255, 0), 2)
    for k, col in zip(("top","bottom","left","right"),
                       [(255,0,0),(255,128,0),(0,128,255),(0,0,255)]):
        cv2.circle(nr_dbg, (int(nr_ell[k][0]), int(nr_ell[k][1])), 6, col, 2)
    cv2.imwrite(str(debug_dir / "NR_sam3_ellipse.jpg"), nr_dbg)
    print(f"  debug overlays: FR_sam3_ellipse.jpg, NR_sam3_ellipse.jpg")

    # Step 4: assemble world + pixel dicts WITH SAM3 cardinal landmarks
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    # add bbox-derived rim_landmarks too (back compat)
    rim_fr = fr_rim_landmarks_from_bbox(bb_fr)
    rim_nr = nr_rim_landmarks_from_bbox(bb_nr)
    for k, (_, w) in rim_fr.items(): world_all[k] = w
    for k, (_, w) in rim_nr.items(): world_all[k] = w
    # Use ONLY the SAM3 ellipse CENTER (landmark 34 = rim ring 3D center).
    # Cardinal direction landmarks (top/bottom/left/right of image-space
    # ellipse) don't cleanly map to 3D world cardinals — especially for NR's
    # nearly-overhead view where the ellipse is nearly circular and image
    # directions don't correspond to world XY axes. The center, however,
    # has a single unambiguous 3D location (RIM_X, RIM_Y, RIM_Z).
    px_fr_all = {**refined_fr, **refined_fr_rim,
                 **{k: px for k,(px,_) in rim_fr.items()}}
    px_nr_all = {**refined_nr, **refined_nr_rim,
                 **{k: px for k,(px,_) in rim_nr.items()}}
    world_all[34] = (RIM_X, RIM_Y, RIM_Z)
    px_fr_all[34] = (fr_ell['cx'], fr_ell['cy'])
    px_nr_all[34] = (nr_ell['cx'], nr_ell['cy'])

    print(f"\n[step 4] PnP with {len(px_fr_all)} FR + {len(px_nr_all)} NR landmarks "
          f"(+1 SAM3 rim center each)")
    fr = solve_pnp(world_all, px_fr_all, fov=73)
    nr = solve_pnp(world_all, px_nr_all, fov=92)
    print(f"  FR: reproj mean={fr['mean']:.1f}px max={fr['max']:.1f}px")
    print(f"  NR: reproj mean={nr['mean']:.1f}px max={nr['max']:.1f}px")
    print(f"  FR cam: X={fr['cam'][0]:+7.0f} Y={fr['cam'][1]:+7.0f} "
          f"|Z|={abs(fr['cam'][2]):.0f}cm  (h={abs(fr['cam'][2])/30.48:.1f}ft)")
    print(f"  NR cam: X={nr['cam'][0]:+7.0f} Y={nr['cam'][1]:+7.0f} "
          f"|Z|={abs(nr['cam'][2]):.0f}cm  (h={abs(nr['cam'][2])/30.48:.1f}ft)")

    # Per-landmark reproj
    print("\n  per-FR-landmark reproj errors:")
    for k, e in sorted(fr['per'].items()):
        kind = ("FLOOR" if k <= 10 else
                "RIM-USER" if k <= 14 else
                "RIM-BBOX" if k <= 21 else
                "RIM-SAM3" if k <= 38 else "?")
        print(f"    #{k:>2} [{kind:9s}]: err={e:5.1f}px")
    print("  per-NR-landmark reproj errors:")
    for k, e in sorted(nr['per'].items()):
        kind = ("FLOOR" if k <= 10 else
                "RIM-USER" if k <= 14 else
                "RIM-BBOX" if k <= 21 else
                "RIM-SAM3" if k <= 38 else "?")
        print(f"    #{k:>2} [{kind:9s}]: err={e:5.1f}px")

    # Step 5: cross-check
    common = sorted(set(px_fr_all) & set(px_nr_all) & set(WORLD_FLOOR))
    errs: list[float] = []
    print(f"\n[step 5] triangulation cross-check on {len(common)} floor landmarks:")
    for k in common:
        X = triangulate(fr['P'], nr['P'], np.array(px_fr_all[k], float),
                                          np.array(px_nr_all[k], float))
        wt = np.array(WORLD_FLOOR[k])
        e = float(np.linalg.norm(X - wt))
        errs.append(e)
        print(f"    #{k:>2}  truth ({wt[0]:.0f},{wt[1]:.0f},{wt[2]:.0f}) -> "
              f"({X[0]:+7.0f},{X[1]:+7.0f},{X[2]:+6.0f})  err {e:5.1f}cm")
    # also cross-check rim center via SAM3 landmark #34
    X34 = triangulate(fr['P'], nr['P'], np.array(px_fr_all[34], float),
                                         np.array(px_nr_all[34], float))
    rim_world = np.array((RIM_X, RIM_Y, RIM_Z))
    rim_err = float(np.linalg.norm(X34 - rim_world))
    print(f"    #34 RIM-CENTER  truth (2009,713,305) -> "
          f"({X34[0]:+7.0f},{X34[1]:+7.0f},{X34[2]:+6.0f})  err {rim_err:5.1f}cm")
    if errs:
        m = float(np.mean(errs))
        print(f"\n  MEAN 3D cross-check (floor): {m:.1f}cm  ({m*0.394:.1f}in)")
        print(f"  RIM-CENTER cross-check     : {rim_err:.1f}cm")

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
        "sam3_ellipse": {"FR": fr_ell, "NR": nr_ell},
        "cross_check_mean_cm": float(np.mean(errs)) if errs else None,
        "rim_center_cross_check_cm": rim_err,
        "source_game": "3398befc",
        "method": "v4 + SAM3 rim ellipse",
    }
    out_path = ROOT / "data/client_report/triangulation_test/calibration_v4_sam3_g3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
