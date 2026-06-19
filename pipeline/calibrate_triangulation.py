#!/usr/bin/env python3
"""Calibration pre-check for triangulation on 5a5f1aae (FR + NR, Linear mode).

Assumptions:
  - HERO12 Linear FOV ≈ 73° horizontal (FR); HERO12 Wide ≈ 92° (NR).
  - Court is regulation NBA approximation (feet).
  - Manual landmark pixel coords estimated from FR_t006 and NR_t006 (both 1920x1080).

This is a *minimum-viable* calibration: 5-6 landmarks per camera, no lens
distortion modelled (Linear mode is close to pinhole). Reports per-camera
PnP reprojection error AND triangulates the rim center as a sanity check
(if both cameras' back-projected rays converge near the assumed rim position
in 3D, the calibration is sound enough for trajectory triangulation).

reprojection < 15px  -> calibration clean, triangulation viable
       15-40px        -> usable but landmark refinement recommended
       > 40px         -> something fundamental is off (court dims? FOV?)
"""
from __future__ import annotations
import cv2
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = ROOT / "data" / "client_report" / "triangulation_test" / "calib"
W, H = 1920, 1080

# --- Court 3D world coordinates ---
# Conventions match DEMO_UBALL/demo/lib/court.py exactly so calibrations
# are interchangeable. Origin at the AMG-baseline / scoreboard-side corner.
# X = court length (cm, 0 = AMG baseline, 2143.7 = far baseline)
# Y = court width  (cm, 0 = scoreboard side, 1426.4 = Iverson side)
# Z = up (cm, 0 = floor)
# Backboard / rim landmarks ADDED for full 3D PnP (demo's tool only does floor).
COURT_LENGTH_CM = 2143.7
COURT_WIDTH_CM  = 1426.4
CENTER_X = COURT_LENGTH_CM / 2.0      # 1071.85
CENTER_Y = COURT_WIDTH_CM / 2.0       # 713.20
LANE_HALF = 194.3                     # 12.75 ft lane half-width
FT_DISTANCE = 594.3                   # baseline -> FT line (cm)
# Backboard hangs ~120 cm (~4 ft) into court from baseline; rim ~15 cm in front
BB_X = 120.0
RIM_X = BB_X + 15.0                   # ~135 cm = 4'5" from baseline
RIM_Z = 304.8                         # 10 ft
BB_TOP_Z, BB_BOT_Z = 396.0, 287.0     # 13 ft top, ~9.4 ft bot
BB_HALF_W = 91.5                      # 6 ft wide / 2

WORLD = {
    # ---- floor landmarks (z = 0) ----
    "L_lane_base_top":   (0.0,         CENTER_Y - LANE_HALF, 0.0),
    "L_lane_base_bot":   (0.0,         CENTER_Y + LANE_HALF, 0.0),
    "L_ft_top":          (FT_DISTANCE, CENTER_Y - LANE_HALF, 0.0),
    "L_ft_bot":          (FT_DISTANCE, CENTER_Y + LANE_HALF, 0.0),
    "L_ft_center":       (FT_DISTANCE, CENTER_Y,             0.0),
    "center":            (CENTER_X,    CENTER_Y,             0.0),
    # ---- above-floor landmarks (needed to break planar PnP degeneracy) ----
    "L_rim_center":      (RIM_X,       CENTER_Y,             RIM_Z),
    "L_bb_top_left":     (BB_X,        CENTER_Y - BB_HALF_W, BB_TOP_Z),
    "L_bb_top_right":    (BB_X,        CENTER_Y + BB_HALF_W, BB_TOP_Z),
    "L_bb_bot_left":     (BB_X,        CENTER_Y - BB_HALF_W, BB_BOT_Z),
    "L_bb_bot_right":    (BB_X,        CENTER_Y + BB_HALF_W, BB_BOT_Z),
}

# --- Pixel coords (manually identified from 1920x1080 calibration frames) ---
# FR sees the AMG hoop from across the court (small, in upper-middle of frame).
# NR sees it close-up (large, in lower-middle).
# These are best-effort estimates; PnP tolerates ~5-10 px noise on each.
PX_FR = {
    # FR_t006.0.jpg (AMG hoop in upper-middle of frame)
    "rim_center":   ( 960,  290),  # hoop center
    "bb_top_left":  ( 870,  185),  # backboard top-left corner
    "bb_top_right": (1055,  185),  # backboard top-right corner
    "bb_bot_left":  ( 870,  255),  # backboard bottom-left
    "bb_bot_right": (1055,  255),  # backboard bottom-right
    "key_top_left": ( 800,  500),  # painted key top-left (FT line)
    "key_top_right":(1115,  500),  # painted key top-right
    "key_bot_left": ( 700,  350),  # painted key bot-left (baseline corner)
    "key_bot_right":(1235,  350),  # painted key bot-right
    "ft_line_center":(960,  500),
}
PX_NR = {
    # NR_t006.0.jpg (AMG hoop in lower-middle/bottom of frame)
    "rim_center":   ( 960, 1020),  # hoop center (near bottom edge)
    "bb_top_left":  ( 770,  775),
    "bb_top_right": (1150,  775),
    "bb_bot_left":  ( 770,  900),
    "bb_bot_right": (1150,  900),
    "key_top_left": ( 580,  640),  # painted key top-left
    "key_top_right":(1340,  640),
    "key_bot_left": ( 360, 1060),  # painted key bot-left (near baseline)
    "key_bot_right":(1560, 1060),
    "ft_line_center":(960,  640),
    "arc_top":      ( 960,  420),  # top of 3pt arc (above center logo)
}


def K_from_fov(fov_h_deg: float) -> np.ndarray:
    f = (W / 2) / np.tan(np.radians(fov_h_deg / 2))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)


def solve(label: str, px: dict, fov_h: float):
    keys = sorted(set(px.keys()) & set(WORLD.keys()))
    obj_pts = np.array([WORLD[k] for k in keys], dtype=np.float64)
    img_pts = np.array([px[k]   for k in keys], dtype=np.float64)
    K = K_from_fov(fov_h)
    dist = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE,
                                  useExtrinsicGuess=False)
    if not ok:
        # try EPNP
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                      flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        print(f"\n=== {label}: PnP FAILED ==="); return None

    # iterative refinement
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, rvec, tvec,
                                  useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)

    # Camera position in world coords
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).ravel()

    print(f"\n=== {label} (FOV {fov_h}°, f={K[0,0]:.0f}px) ===")
    print(f"  used {len(keys)} landmarks, reprojection error:")
    print(f"    mean = {err.mean():5.1f} px")
    print(f"    max  = {err.max():5.1f} px")
    print(f"  camera position (ft): X={cam_pos[0]:+6.1f}  Y={cam_pos[1]:+6.1f}  Z={cam_pos[2]:+6.1f}")
    print(f"  per-landmark errors:")
    for k, e, ip, pp in zip(keys, err, img_pts, proj):
        print(f"    {k:<18} px=({ip[0]:>4.0f},{ip[1]:>4.0f}) -> proj=({pp[0]:>4.0f},{pp[1]:>4.0f}) err={e:5.1f}")
    return dict(K=K.tolist(), dist=dist.tolist(),
                rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(),
                cam_pos_ft=cam_pos.tolist(),
                reproj_err_mean=float(err.mean()),
                reproj_err_max=float(err.max()))


def triangulate_point(K1, R1, t1, p1, K2, R2, t2, p2):
    """Standard linear DLT triangulation. Returns 3D point in world coords."""
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1],
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def main():
    fr = solve("FR (Linear)", PX_FR, fov_h=73)
    nr = solve("NR (Wide)",   PX_NR, fov_h=92)
    if not (fr and nr):
        return

    # Sanity check: triangulate the rim center using both cameras
    # and compare to the assumed world position (0, 0, 10)
    K1 = np.array(fr["K"]); rvec1 = np.array(fr["rvec"]); tvec1 = np.array(fr["tvec"])
    K2 = np.array(nr["K"]); rvec2 = np.array(nr["rvec"]); tvec2 = np.array(nr["tvec"])
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    p1 = np.array(PX_FR["rim_center"], dtype=np.float64)
    p2 = np.array(PX_NR["rim_center"], dtype=np.float64)
    Xrim = triangulate_point(K1, R1, tvec1, p1, K2, R2, tvec2, p2)
    expect = np.array(WORLD["rim_center"])
    err3d = np.linalg.norm(Xrim - expect)
    print(f"\n=== TRIANGULATION SANITY CHECK (rim center) ===")
    print(f"  expected: (0, 0, 10) ft")
    print(f"  recovered: ({Xrim[0]:+.2f}, {Xrim[1]:+.2f}, {Xrim[2]:+.2f}) ft")
    print(f"  3D error: {err3d:.2f} ft  ({err3d*12:.1f} inches)")

    out = {"FR": fr, "NR": nr,
           "triangulated_rim_ft": Xrim.tolist(),
           "rim_3d_err_ft": float(err3d)}
    (CAL_DIR / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {CAL_DIR / 'calibration.json'}")


if __name__ == "__main__":
    main()
