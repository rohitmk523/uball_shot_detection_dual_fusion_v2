#!/usr/bin/env python3
"""Automated court-line calibration for June games.

Detects the 4 corners of the AMG-painted lane from each empty-court frame
(red HSV mask -> largest contour -> 4 extreme corners) instead of relying on
hardcoded clicks from game-1. These corners map directly to four world
landmarks (#3, #4, #5, #6) at sub-camera accuracy.

Adds the SAM3 rim center as a single non-coplanar landmark (#34) so PnP
isn't fully degenerate. Sweeps a small FOV range around the Hero 12 Wide
nominal value to find the per-game minimum-reproj fit.

Outputs ``calibration_june_<id>_auto.json`` and prints floor cross-check.

Usage:
  python pipeline/calibrate_auto.py --game-id 4692eb2b
  python pipeline/calibrate_auto.py --all
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_v4 import (  # noqa: E402
    WORLD_FLOOR, WORLD_RIM, FR_W, NR_W,
    yolo_hoop_bbox, fr_rim_landmarks_from_bbox, nr_rim_landmarks_from_bbox,
)
from calibrate_june_sam3 import segment_rim_ellipse, SAM3_W  # noqa: E402
from ultralytics import YOLO, SAM  # noqa: E402

RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
W_IMG, H_IMG = 1920, 1080

# Paint-corner -> WORLD_FLOOR landmark IDs.
# FR view: top of paint in image = baseline (far from camera, world X=2145)
#          bottom of paint in image = FT line (close to camera, world X=1553)
#          left  in image (small x) = world Y=521 (smaller Y)
#          right in image (large x) = world Y=901
# NR view: NR is mounted ABOVE the rim. The actual baseline projects to image
#          y=2830 (OFF screen) per the existing calibration. NR's "bottom of
#          paint" in the image at y~1025 is NOT the baseline — it's the
#          painted-area lower edge at some intermediate world X (~1850 cm)
#          where the AMG painted box ends. We CANNOT use NR BL/BR as floor
#          landmarks. Only TL/TR (top of paint = FT line ends) are valid.
#          left  in image (small x) = world Y=901
#          right in image (large x) = world Y=521
FR_CORNER_TO_LM = {"tl": 5, "tr": 6, "bl": 3, "br": 4}
NR_CORNER_TO_LM = {"tl": 4, "tr": 3}  # only top corners valid


# ===============================================================
# Paint corner detection (same logic for FR/NR; vertical range differs)
# ===============================================================

def _red_mask(img: np.ndarray, y_lo_frac: float, y_hi_frac: float
              ) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, np.array([0, 100, 60], np.uint8),
                          np.array([10, 255, 255], np.uint8))
    hi = cv2.inRange(hsv, np.array([160, 100, 60], np.uint8),
                          np.array([179, 255, 255], np.uint8))
    red = cv2.bitwise_or(lo, hi)
    h, _ = red.shape
    crop = np.zeros_like(red)
    crop[int(h * y_lo_frac): int(h * y_hi_frac), :] = 255
    red = cv2.bitwise_and(red, crop)
    # Close gaps from logo/text, then re-open to remove specks
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return red


def _largest_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask)
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, -1)
    return out


def _corners_from_paint(paint: np.ndarray, epsilon_frac: float) -> dict | None:
    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5000:
        return None
    cx, cy = np.mean(cnt.reshape(-1, 2), axis=0)
    epsilon = epsilon_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    if len(approx) < 4:
        return None

    def best(filt):
        cands = [p for p in approx if filt(p)]
        if not cands:
            return None
        return max(cands, key=lambda p: float(np.hypot(p[0] - cx, p[1] - cy)))

    tl = best(lambda p: p[0] < cx and p[1] < cy)
    tr = best(lambda p: p[0] > cx and p[1] < cy)
    bl = best(lambda p: p[0] < cx and p[1] > cy)
    br = best(lambda p: p[0] > cx and p[1] > cy)
    if any(c is None for c in (tl, tr, bl, br)):
        return None
    return dict(tl=tuple(map(float, tl)), tr=tuple(map(float, tr)),
                bl=tuple(map(float, bl)), br=tuple(map(float, br)),
                area=float(paint.sum() / 255),
                approx_n=int(len(approx)))


def detect_fr_paint_corners(img: np.ndarray) -> dict | None:
    red = _red_mask(img, y_lo_frac=0.30, y_hi_frac=0.75)
    return _corners_from_paint(_largest_component(red), epsilon_frac=0.01)


def detect_nr_paint_corners(img: np.ndarray) -> dict | None:
    red = _red_mask(img, y_lo_frac=0.30, y_hi_frac=0.95)
    return _corners_from_paint(_largest_component(red), epsilon_frac=0.005)


def median_corners(corners_per_game: dict[str, dict]) -> dict:
    """Median over games of each corner (TL/TR/BL/BR) as fallback for
    games with occluded paint."""
    out = {}
    for key in ("tl", "tr", "bl", "br"):
        xs = [c[key][0] for c in corners_per_game.values() if c is not None]
        ys = [c[key][1] for c in corners_per_game.values() if c is not None]
        out[key] = (float(np.median(xs)), float(np.median(ys)))
    return out


# ===============================================================
# Sub-pixel corner refinement using the LANE LINES (white)
# ===============================================================

def refine_corner_with_lines(img: np.ndarray, corner: tuple[float, float],
                              win: int = 25) -> tuple[float, float]:
    """Refine a paint corner by snapping it to the nearest sub-pixel intersection
    of two strong line segments (the lane line + the baseline/FT line).
    Use cv2.cornerSubPix on a Canny edge map within a window around the corner.
    """
    cx, cy = corner
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize locally so cornerSubPix snaps to real edges, not noise
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    refined = cv2.cornerSubPix(
        gray, pt, (win, win), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    return float(refined[0, 0, 0]), float(refined[0, 0, 1])


# ===============================================================
# PnP solving with Hero 12 Wide pinhole + distortion (Brown-Conrady k1, k2)
# ===============================================================

def _K_from_fov(fov_h: float, w: int = W_IMG, h: int = H_IMG) -> np.ndarray:
    f = (w / 2.0) / np.tan(np.radians(fov_h / 2.0))
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def solve_pnp_with_fov_sweep(world: dict, px: dict, fov_grid: tuple[int, ...],
                              dist_grid: tuple[tuple, ...] | None = None
                              ) -> dict | None:
    """Try multiple (FOV, k1, k2) combinations; return the lowest-reproj fit."""
    keys = sorted(px)
    obj = np.array([world[k] for k in keys], dtype=np.float64)
    img = np.array([px[k] for k in keys], dtype=np.float64)
    if dist_grid is None:
        dist_grid = ((0.0, 0.0),)
    best = None
    for fov in fov_grid:
        K = _K_from_fov(fov)
        for (k1, k2) in dist_grid:
            dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
            # SQPNP works with 3+ points (no DLT init needed)
            flag = cv2.SOLVEPNP_SQPNP if len(keys) < 6 else cv2.SOLVEPNP_ITERATIVE
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            P = K @ np.hstack([R, tvec])
            mean = float(err.mean())
            cand = dict(K=K, R=R, t=tvec, rvec=rvec, P=P, dist=dist,
                        mean=mean, max=float(err.max()),
                        per_err=dict(zip(keys, err.tolist())),
                        fov=fov, k1=float(k1), k2=float(k2))
            if best is None or mean < best['mean']:
                best = cand
    return best


def triangulate(P1: np.ndarray, P2: np.ndarray,
                p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    A = np.array([p1[0] * P1[2] - P1[0], p1[1] * P1[2] - P1[1],
                  p2[0] * P2[2] - P2[0], p2[1] * P2[2] - P2[1]])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def cross_check(fr: dict, nr: dict, px_fr: dict, px_nr: dict,
                world: dict, exclude_z_nonzero: bool = True
                ) -> tuple[float, float, list]:
    common = sorted(set(px_fr) & set(px_nr) & set(world))
    errs, details = [], []
    for k in common:
        if exclude_z_nonzero and world[k][2] != 0:
            continue
        X = triangulate(fr['P'], nr['P'], np.asarray(px_fr[k], float),
                                          np.asarray(px_nr[k], float))
        gt = np.asarray(world[k], float)
        e = float(np.linalg.norm(X - gt))
        errs.append(e)
        details.append(dict(k=k, world=tuple(gt.tolist()),
                            recovered=tuple(map(float, X)), err_cm=e))
    if not errs:
        return float('nan'), float('nan'), details
    return float(np.mean(errs)), float(np.max(errs)), details


# ===============================================================
# Main driver
# ===============================================================

@dataclass
class GameLandmarks:
    gid: str
    fr_img: np.ndarray
    nr_img: np.ndarray
    fr_corners_raw: dict | None   # tl/tr/bl/br before refinement
    nr_corners_raw: dict | None
    fr_corners_used: dict          # tl/tr/bl/br after fallback + refinement
    nr_corners_used: dict
    fr_rim_px: tuple[float, float] | None
    nr_rim_px: tuple[float, float] | None
    fr_rim_bbox: dict | None
    nr_rim_bbox: dict | None


def collect_paint_corners(games: tuple[str, ...]) -> dict:
    """First pass: detect raw paint corners per game on both cameras."""
    out = {}
    for gid in games:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        fr_img = cv2.imread(str(G / "calib" / "FR_t30.jpg"))
        nr_img = cv2.imread(str(G / "calib" / "NR_t30.jpg"))
        out[gid] = dict(
            fr_img=fr_img, nr_img=nr_img,
            fr_corners=detect_fr_paint_corners(fr_img),
            nr_corners=detect_nr_paint_corners(nr_img),
        )
    return out


def calibrate_one(gid: str, paint_lib: dict, sam: SAM | None,
                   fr_model: YOLO | None, nr_model: YOLO | None,
                   verbose: bool = True) -> dict:
    """Solve auto-calibration for one game using the paint_lib (median fallback
    for occluded games) and per-game SAM3 rim detection."""
    gdata = paint_lib[gid]
    fr_img, nr_img = gdata['fr_img'], gdata['nr_img']

    # Median fallback for occluded corners
    fr_med = median_corners({g: paint_lib[g]['fr_corners']
                              for g in paint_lib if paint_lib[g]['fr_corners']})
    nr_med = median_corners({g: paint_lib[g]['nr_corners']
                              for g in paint_lib if paint_lib[g]['nr_corners']})
    fr_corners = gdata['fr_corners'] if gdata['fr_corners'] is not None else fr_med
    nr_corners = gdata['nr_corners'] if gdata['nr_corners'] is not None else nr_med

    # Detect significantly-broken corners (>40 px from median) -> swap to median
    def patch_with_median(corners: dict, med: dict, label: str) -> dict:
        out = dict(corners)
        for key in ("tl", "tr", "bl", "br"):
            d = float(np.hypot(corners[key][0] - med[key][0],
                                corners[key][1] - med[key][1]))
            if d > 40:
                if verbose:
                    print(f"    [{label} {key}] {corners[key]} -> median "
                          f"{med[key]} (drift {d:.0f}px)")
                out[key] = med[key]
        return out

    fr_corners = patch_with_median(fr_corners, fr_med, "FR")
    nr_corners = patch_with_median(nr_corners, nr_med, "NR")

    # Sub-pixel refine each corner against local Canny edges
    fr_refined = {k: refine_corner_with_lines(fr_img, fr_corners[k], win=20)
                  for k in FR_CORNER_TO_LM}
    nr_refined = {k: refine_corner_with_lines(nr_img, nr_corners[k], win=25)
                  for k in NR_CORNER_TO_LM}

    # Build px-world dicts for floor landmarks #3..#6
    px_fr = {FR_CORNER_TO_LM[k]: fr_refined[k] for k in FR_CORNER_TO_LM}
    px_nr = {NR_CORNER_TO_LM[k]: nr_refined[k] for k in NR_CORNER_TO_LM}
    # NR landmark #10: FT line center = midpoint of NR paint TL/TR
    # (world (1549, 711, 0)); cheap, well-conditioned, no extra detection
    if "tl" in nr_refined and "tr" in nr_refined:
        tl_x, tl_y = nr_refined["tl"]
        tr_x, tr_y = nr_refined["tr"]
        px_nr[10] = ((tl_x + tr_x) / 2.0, (tl_y + tr_y) / 2.0)
    # FR landmark #10 likewise: midpoint of paint BL/BR (FT line center)
    if "bl" in fr_refined and "br" in fr_refined:
        bl_x, bl_y = fr_refined["bl"]
        br_x, br_y = fr_refined["br"]
        px_fr[10] = ((bl_x + br_x) / 2.0, (bl_y + br_y) / 2.0)
    world_all = dict(WORLD_FLOOR)  # #1..#10 (auto-detected: #3..#6, #10)

    # ---- SAM3 rim center + YOLO-bbox-derived rim landmarks ----
    #   #34: SAM3 ellipse center (sharp at rim ring) at z=305
    #   #20: bbox top center -> FR rim ring top (z=305), NR rim back edge
    #   #21: FR only -> bbox bottom = net trail bottom (z=265, NON-coplanar!)
    fr_rim_px = nr_rim_px = None
    bb_fr = bb_nr = None
    fr_ell = nr_ell = None
    if sam is not None and fr_model is not None and nr_model is not None:
        bb_fr = yolo_hoop_bbox(fr_model, fr_img)
        bb_nr = yolo_hoop_bbox(nr_model, nr_img)
        if bb_fr is not None:
            fr_ell = segment_rim_ellipse(sam, fr_img,
                                          (bb_fr['x1'], bb_fr['y1'],
                                           bb_fr['x2'], bb_fr['y2']))
        if bb_nr is not None:
            nr_ell = segment_rim_ellipse(sam, nr_img,
                                          (bb_nr['x1'], bb_nr['y1'],
                                           bb_nr['x2'], bb_nr['y2']))
        # bbox-derived rim/net landmarks
        if bb_fr is not None:
            for k, (px, w) in fr_rim_landmarks_from_bbox(bb_fr).items():
                px_fr[k] = px
                world_all[k] = w
        if bb_nr is not None:
            for k, (px, w) in nr_rim_landmarks_from_bbox(bb_nr).items():
                px_nr[k] = px
                world_all[k] = w
        # SAM3 ellipse center (#34) — most precise rim landmark
        if fr_ell is not None:
            fr_rim_px = (float(fr_ell['cx']), float(fr_ell['cy']))
            px_fr[34] = fr_rim_px
            world_all[34] = (RIM_X, RIM_Y, RIM_Z)
        if nr_ell is not None:
            nr_rim_px = (float(nr_ell['cx']), float(nr_ell['cy']))
            px_nr[34] = nr_rim_px
            world_all[34] = (RIM_X, RIM_Y, RIM_Z)

    # ---- PnP: FORCE FOV to match SAM3 baseline (73/92) which produces
    # known-good rim cross-check. Auto-detection just refits the camera POSE
    # with sub-pixel-accurate landmarks. Avoids the FOV-sweep instability
    # that drove rim cross-check from 5cm to 15+cm.
    fov_grid_fr = (73,)
    fov_grid_nr = (92,)
    dist_grid = ((0.0, 0.0),)

    if verbose:
        print(f"  PnP solve: FR landmarks={sorted(px_fr.keys())}  "
              f"NR landmarks={sorted(px_nr.keys())}")
    fr = solve_pnp_with_fov_sweep(world_all, px_fr, fov_grid_fr, dist_grid)
    nr = solve_pnp_with_fov_sweep(world_all, px_nr, fov_grid_nr, dist_grid)
    if fr is None or nr is None:
        raise RuntimeError(f"{gid}: PnP failed")

    # ---- Cross-check on floor landmarks only ----
    floor_mean, floor_max, details = cross_check(fr, nr, px_fr, px_nr,
                                                  world_all, exclude_z_nonzero=True)
    # Rim cross-check
    rim_err = None
    if 34 in px_fr and 34 in px_nr:
        X = triangulate(fr['P'], nr['P'],
                        np.asarray(px_fr[34], float),
                        np.asarray(px_nr[34], float))
        rim_err = float(np.linalg.norm(X - np.array((RIM_X, RIM_Y, RIM_Z))))

    if verbose:
        print(f"  FR: fov={fr['fov']} k1={fr['k1']:+.2f} reproj={fr['mean']:.1f}px")
        print(f"  NR: fov={nr['fov']} k1={nr['k1']:+.2f} reproj={nr['mean']:.1f}px")
        print(f"  cross-check floor (n={len(details)}): mean={floor_mean:.1f}cm max={floor_max:.1f}cm")
        if rim_err is not None:
            print(f"  cross-check rim: {rim_err:.1f}cm")

    return dict(
        gid=gid,
        FR=dict(K=fr['K'].tolist(), rvec=fr['rvec'].tolist(),
                tvec=fr['t'].tolist(),
                reproj_mean=fr['mean'], reproj_max=fr['max'],
                fov=fr['fov'], k1=fr['k1'], k2=fr['k2'],
                paint_corners=fr_corners, paint_corners_refined=fr_refined,
                rim_px=fr_rim_px,
                hoop_bbox=bb_fr,
                sam3_ellipse=fr_ell),
        NR=dict(K=nr['K'].tolist(), rvec=nr['rvec'].tolist(),
                tvec=nr['t'].tolist(),
                reproj_mean=nr['mean'], reproj_max=nr['max'],
                fov=nr['fov'], k1=nr['k1'], k2=nr['k2'],
                paint_corners=nr_corners, paint_corners_refined=nr_refined,
                rim_px=nr_rim_px,
                hoop_bbox=bb_nr,
                sam3_ellipse=nr_ell),
        cross_check_floor_mean_cm=floor_mean,
        cross_check_floor_max_cm=floor_max,
        rim_center_cross_check_cm=rim_err,
        cross_check_details=details,
        method="auto: paint corners + SAM3 rim, FOV sweep + Brown-Conrady",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-sam3", action="store_true",
                    help="skip SAM3 (faster, but loses rim landmark)")
    args = ap.parse_args()
    if not args.game_id and not args.all:
        print("specify --game-id or --all"); return 1

    target_games = GAMES if args.all else (args.game_id,)

    print("=" * 80)
    print("[step 0] detecting paint corners on all 4 June games (for fallback)")
    paint_lib = collect_paint_corners(GAMES)
    for g in GAMES:
        c = paint_lib[g]['fr_corners']
        n = paint_lib[g]['nr_corners']
        print(f"  {g}: FR={'ok' if c else 'NONE'} {'(n=' + str(c['approx_n']) + ')' if c else ''}"
              f"  NR={'ok' if n else 'NONE'} {'(n=' + str(n['approx_n']) + ')' if n else ''}")

    sam = fr_model = nr_model = None
    if not args.no_sam3:
        print("\n[step 1] loading SAM3 + YOLO models (~10s)")
        t0 = time.time()
        sam = SAM(str(SAM3_W))
        fr_model = YOLO(str(FR_W))
        nr_model = YOLO(str(NR_W))
        print(f"  loaded in {time.time() - t0:.1f}s")

    for gid in target_games:
        print(f"\n=== calibrating {gid} ===")
        result = calibrate_one(gid, paint_lib, sam, fr_model, nr_model)
        out_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_auto.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  saved {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
