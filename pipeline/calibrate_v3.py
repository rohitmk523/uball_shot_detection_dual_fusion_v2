#!/usr/bin/env python3
"""Calibration v3 — adds 4 rim landmarks per camera (the user's
right/left/top/center of the rim) to the 10 floor landmarks. With non-floor
points, PnP becomes well-conditioned and should solve FOV, position, and
orientation correctly.

Per-user geometry: FR mounted on one backboard, NR on the other; both look at
the SAME (NR's) hoop. They face each other -> NR-right = FR-left physically.

We try 4 interpretations and pick the best:
  * Top of rim = BACK edge (touching backboard, X smaller) vs FRONT edge
  * NR-right = +Y (Iverson side) vs -Y (scoreboard side)
"""
from __future__ import annotations
import cv2, json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080

# ---- 10 floor landmarks (cm), shared with calibrate_v2 ----
WORLD_FLOOR = {
    1:  (2142.4,    0.9, 0.0),
    2:  (2142.4, 1425.3, 0.0),
    3:  (1553.0,  521.6, 0.0),
    4:  (1553.0,  901.4, 0.0),
    5:  (2145.7,  518.3, 0.0),
    6:  (2145.7,  907.9, 0.0),
    7:  (1369.6,  708.2, 0.0),
    8:  (1074.9,  711.5, 0.0),
    9:  (1733.1,  718.0, 0.0),
    10: (1549.7,  711.5, 0.0),
}
# ---- 4 rim landmarks (cm) for the LETS HOOP hoop at X≈2008 (R_hoop end).
# Floor-landmarks-only v2 calibration put NR camera at X=2046 (~ R baseline),
# meaning NR is mounted on the LETS HOOP backboard, not AMG. FR is at the
# opposite (AMG) backboard and looks at LETS HOOP from far away.
RIM_RADIUS = 22.86  # cm
COURT_LEN  = 2143.7
BB_INSET   = 120.0   # backboard hangs 4 ft into court from baseline
RIM_INSET  = 15.0    # rim 15 cm in front of backboard, toward center
RIM_X      = COURT_LEN - BB_INSET - RIM_INSET  # = 2008.7
RIM_Y      = 713.2
RIM_Z      = 304.8

# 4 candidate world-coord interpretations:
#   - "top" of rim   = BACK edge (touching backboard, x=112.1) or FRONT (x=157.9)
#   - NR-right side  = +Y (Iverson) or -Y (scoreboard)
def rim_worlds(top_back: bool, nr_right_plus_y: bool):
    top_x = RIM_X - RIM_RADIUS if top_back else RIM_X + RIM_RADIUS
    nr_right_y = RIM_Y + RIM_RADIUS if nr_right_plus_y else RIM_Y - RIM_RADIUS
    nr_left_y  = RIM_Y - RIM_RADIUS if nr_right_plus_y else RIM_Y + RIM_RADIUS
    # NR labels:                       FR labels (flipped left/right):
    return {
        11: (RIM_X,   nr_right_y, RIM_Z),  # NR-right == FR-left (physically same point)
        12: (RIM_X,   nr_left_y,  RIM_Z),  # NR-left  == FR-right
        13: (top_x,   RIM_Y,      RIM_Z),  # Top (both cameras)
        14: (RIM_X,   RIM_Y,      RIM_Z),  # Center (both cameras)
    }

# ---- Pixel coords ----
PX_FR_FLOOR = {
    1:(331,576), 2:(1550,577), 3:(709,728), 4:(1178,723),
    5:(770,579), 6:(1113,579), 7:(947,799), 8:(949,970),
    9:(942,672), 10:(944,728),
}
PX_NR_FLOOR_RAW = {
    1:(0,809), 2:(1917,826), 3:(1211,601), 4:(691,595),
    5:(540,1075), 6:(1346,1075), 7:(952,458), 8:(952,317),
    9:(949,823), 10:(951,601),
}
NR_INVISIBLE_FLOOR = {1, 2, 5, 6}
PX_NR_FLOOR = {k:v for k,v in PX_NR_FLOOR_RAW.items() if k not in NR_INVISIBLE_FLOOR}

# Rim clicks (FR clicks are in FR's perspective; NR clicks in NR's perspective)
# FR labels: 1=FR-right, 2=FR-left, 3=top, 4=center
# NR labels: 1=NR-right, 2=NR-left, 3=top, 4=center
# Mapping into our 11..14 world keys:
#   - FR-right (1) ↔ NR-left (2) physically  -> world key 12 (NR-left)
#   - FR-left  (2) ↔ NR-right (1) physically -> world key 11 (NR-right)
#   - top      (3) -> world key 13
#   - center   (4) -> world key 14
PX_FR_RIM_RAW = {1:(913,307), 2:(964,306), 3:(940,306), 4:(940,307)}
PX_NR_RIM_RAW = {1:(1153,1033), 2:(719,1044), 3:(948,839), 4:(944,1034)}
# remap FR's 1/2 to 12/11 (cross-side); NR's 1/2 to 11/12 (same labelling)
PX_FR_RIM = {12: PX_FR_RIM_RAW[1], 11: PX_FR_RIM_RAW[2],
             13: PX_FR_RIM_RAW[3], 14: PX_FR_RIM_RAW[4]}
PX_NR_RIM = {11: PX_NR_RIM_RAW[1], 12: PX_NR_RIM_RAW[2],
             13: PX_NR_RIM_RAW[3], 14: PX_NR_RIM_RAW[4]}


def K_from_fov(fov_h_deg):
    f = (W/2) / np.tan(np.radians(fov_h_deg/2))
    return np.array([[f,0,W/2],[0,f,H/2],[0,0,1]], dtype=np.float64)


def solve(world_dict, px_dict, fov_h):
    keys = sorted(px_dict.keys())
    obj = np.array([world_dict[k] for k in keys], dtype=np.float64)
    img = np.array([px_dict[k]    for k in keys], dtype=np.float64)
    K = K_from_fov(fov_h)
    dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1,2) - img, axis=1)
    return dict(K=K, R=R, rvec=rvec, tvec=tvec, cam_pos=cam,
                err_mean=float(err.mean()), err_max=float(err.max()),
                per=err, keys=keys, world_pts=obj, img_pts=img)


def triangulate(P1, P2, p1, p2):
    A = np.array([
        p1[0]*P1[2] - P1[0],
        p1[1]*P1[2] - P1[1],
        p2[0]*P2[2] - P2[0],
        p2[1]*P2[2] - P2[1],
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def evaluate(top_back, nr_right_plus_y, fr_fov, nr_fov):
    rim_world = rim_worlds(top_back, nr_right_plus_y)
    world_all = {**WORLD_FLOOR, **rim_world}
    px_fr_all = {**PX_FR_FLOOR, **PX_FR_RIM}
    px_nr_all = {**PX_NR_FLOOR, **PX_NR_RIM}
    fr = solve(world_all, px_fr_all, fr_fov)
    nr = solve(world_all, px_nr_all, nr_fov)
    if fr is None or nr is None:
        return None, None, None
    # triangulation cross-check on common landmarks
    common = sorted(set(px_fr_all) & set(px_nr_all))
    P1 = fr['K'] @ np.hstack([fr['R'], fr['tvec']])
    P2 = nr['K'] @ np.hstack([nr['R'], nr['tvec']])
    errs = []
    for k in common:
        X = triangulate(P1, P2, np.array(px_fr_all[k], float),
                                np.array(px_nr_all[k], float))
        Wt = np.array(world_all[k])
        errs.append(np.linalg.norm(X - Wt))
    return fr, nr, np.mean(errs)


def main():
    print("=== sweeping interpretations (top edge + Y polarity) ===")
    best = None
    for top_back in [True, False]:
        for ny in [True, False]:
            for fr_fov in [73, 92, 122]:
                fr, nr, tri_err = evaluate(top_back, ny, fr_fov, 92)
                if fr is None: continue
                tag = (f"top={'BACK' if top_back else 'FRONT'}  "
                       f"NR-right={'+Y(Iverson)' if ny else '-Y(scoreboard)'}  "
                       f"FR_FOV={fr_fov}°")
                summary = (f"  FR reproj={fr['err_mean']:5.1f}px  "
                           f"NR reproj={nr['err_mean']:5.1f}px  "
                           f"3D cross-check={tri_err:.1f}cm")
                print(f" {tag}\n{summary}")
                score = fr['err_mean'] + nr['err_mean'] + tri_err
                if best is None or score < best['score']:
                    best = dict(fr=fr, nr=nr, tri=tri_err, score=score,
                                top_back=top_back, ny=ny, fr_fov=fr_fov)
    print(f"\n=== BEST interpretation ===")
    b = best
    print(f"  top edge: {'BACK (touching backboard)' if b['top_back'] else 'FRONT (away from backboard)'}")
    print(f"  NR-right side: {'+Y (Iverson)' if b['ny'] else '-Y (scoreboard)'}")
    print(f"  FR FOV: {b['fr_fov']}°")
    print(f"  FR reproj={b['fr']['err_mean']:.1f}px (max {b['fr']['err_max']:.1f})")
    print(f"  NR reproj={b['nr']['err_mean']:.1f}px (max {b['nr']['err_max']:.1f})")
    print(f"  cross-camera 3D error: {b['tri']:.1f} cm ({b['tri']*0.394:.1f} in)")
    print()
    print(f"  FR cam pos (cm): X={b['fr']['cam_pos'][0]:+7.0f}  Y={b['fr']['cam_pos'][1]:+7.0f}  height|Z|={abs(b['fr']['cam_pos'][2]):.0f}")
    print(f"  FR cam pos (ft): X={b['fr']['cam_pos'][0]/30.48:+5.1f}  Y={b['fr']['cam_pos'][1]/30.48:+5.1f}  height={abs(b['fr']['cam_pos'][2])/30.48:.1f}")
    print(f"  NR cam pos (cm): X={b['nr']['cam_pos'][0]:+7.0f}  Y={b['nr']['cam_pos'][1]:+7.0f}  height|Z|={abs(b['nr']['cam_pos'][2]):.0f}")
    print(f"  NR cam pos (ft): X={b['nr']['cam_pos'][0]/30.48:+5.1f}  Y={b['nr']['cam_pos'][1]/30.48:+5.1f}  height={abs(b['nr']['cam_pos'][2])/30.48:.1f}")
    print()
    # rim sanity-check projection (project the rim CENTER into each camera, see where it lands)
    rim_world = rim_worlds(b['top_back'], b['ny'])
    dist = np.zeros(5)
    rc = np.array([rim_world[14]], dtype=np.float64)
    for nm, c in [('FR', b['fr']), ('NR', b['nr'])]:
        p, _ = cv2.projectPoints(rc, c['rvec'], c['tvec'], c['K'], dist)
        px = p.ravel()
        ok = (0 <= px[0] < W) and (0 <= px[1] < H)
        print(f"  {nm} projects rim CENTER -> pixel ({px[0]:6.0f},{px[1]:6.0f})  {'IN-FRAME' if ok else 'OUT'}")


if __name__ == "__main__":
    main()
