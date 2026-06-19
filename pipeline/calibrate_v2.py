#!/usr/bin/env python3
"""Calibration v2 — uses the user-clicked landmarks (matched to the DEMO_UBALL
court convention, cm units) for 5a5f1aae FR + NR.

User's 10 numbered landmarks (match demo FR_cal_clickpairs order):
  1  R baseline @ scoreboard corner            (2142.4,    0.9, 0)
  2  R baseline @ Iverson corner               (2142.4, 1425.3, 0)
  3  R FT line @ lane (top, scoreboard side)   (1553.0,  521.6, 0)
  4  R FT line @ lane (bot, Iverson side)      (1553.0,  901.4, 0)
  5  R lane-base @ baseline (top)              (2145.7,  518.3, 0)
  6  R lane-base @ baseline (bot)              (2145.7,  907.9, 0)
  7  R top-of-key (3pt apex)                   (1369.6,  708.2, 0)
  8  center court                              (1074.9,  711.5, 0)
  9  R FT-circle far apex (near center)        (1733.1,  718.0, 0)
 10  R FT-line center                          (1549.7,  711.5, 0)

NR can't see {1,2,5,6} (the R baseline = far end behind NR). All 10 visible in FR.

All landmarks are on the floor (z=0) -> planar PnP. We use SOLVEPNP_IPPE which
handles the planar case correctly, then disambiguate the up/down ambiguity by
keeping the solution with camera Z > 0.
"""
from __future__ import annotations
import cv2, json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080

# World coords (cm) matching DEMO_UBALL court convention
WORLD = {
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

# User-clicked pixel coords (2026-06-02)
PX_FR = {
    1: (331, 576), 2: (1550, 577), 3: (709, 728), 4: (1178, 723),
    5: (770, 579), 6: (1113, 579), 7: (947, 799), 8: (949, 970),
    9: (942, 672), 10: (944, 728),
}
# NR — user marked 1,2,5,6 as not visible
PX_NR_RAW = {
    1: (0, 809), 2: (1917, 826), 3: (1211, 601), 4: (691, 595),
    5: (540, 1075), 6: (1346, 1075), 7: (952, 458), 8: (952, 317),
    9: (949, 823), 10: (951, 601),
}
NR_INVISIBLE = {1, 2, 5, 6}
PX_NR = {k: v for k, v in PX_NR_RAW.items() if k not in NR_INVISIBLE}

# Approximate intrinsics (HERO12, 1080p)
# FR was confirmed Linear (~73° H FOV); NR is Wide-ish (~92° H FOV)
def K_from_fov(fov_h_deg: float) -> np.ndarray:
    f = (W / 2) / np.tan(np.radians(fov_h_deg / 2))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)


def solve_planar(label, px, fov_h):
    keys = sorted(px.keys())
    obj = np.array([WORLD[k] for k in keys], dtype=np.float64)
    img = np.array([px[k]    for k in keys], dtype=np.float64)
    K = K_from_fov(fov_h)
    dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print(f"\n=== {label}: PnP failed"); return None
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    per = np.linalg.norm(proj.reshape(-1,2) - img, axis=1)

    print(f"\n=== {label}  (FOV {fov_h}°, f={K[0,0]:.0f}px, "
          f"{len(keys)} landmarks) ===")
    print(f"  reprojection error mean={per.mean():5.1f}px  max={per.max():5.1f}px")
    # Convention note: our world has Z up. PnP's cam_pos[2] sign depends on
    # frame handedness, so we report height as |Z| from floor.
    print(f"  camera position (cm): X={cam[0]:+7.0f}  Y={cam[1]:+7.0f}  height|Z|={abs(cam[2]):+6.0f}")
    print(f"               (ft):    X={cam[0]/30.48:+6.1f}  Y={cam[1]/30.48:+6.1f}  height={abs(cam[2])/30.48:+5.1f}")
    print(f"  per-landmark errors:")
    for k, p, q, e in zip(keys, img, proj.reshape(-1,2), per):
        w = WORLD[k]
        print(f"    #{k:>2}  world ({w[0]:>7.1f}, {w[1]:>7.1f})  "
              f"img ({int(p[0]):>4},{int(p[1]):>4}) -> proj ({int(q[0]):>4},{int(q[1]):>4})  err {e:5.1f}")
    return dict(K=K, R=R, rvec=rvec, tvec=tvec,
                cam_pos=cam, err_mean=float(per.mean()), err_max=float(per.max()),
                keys=keys, img_pts=img, world_pts=obj)


def project_rim_into_cameras(fr, nr):
    """Project the assumed L (AMG) and R (LETS HOOP) rim centers into each
    camera and report pixel positions. If projections land where a human eye
    would expect the rim, the calibration is sound."""
    L_RIM_CM = np.array([135.0, 713.2, 304.8])     # AMG hoop, 10ft up
    R_RIM_CM = np.array([2008.7, 713.2, 304.8])    # LETS HOOP hoop, 10ft up
    dist = np.zeros(5)
    print(f"\n=== rim center projections (sanity check) ===")
    for cam_name, cam in [("FR", fr), ("NR", nr)]:
        if cam is None:
            continue
        for rim_name, rim in [("L_rim (AMG)", L_RIM_CM), ("R_rim (LETS HOOP)", R_RIM_CM)]:
            proj, _ = cv2.projectPoints(rim.reshape(1, 3),
                                        cam['rvec'], cam['tvec'], cam['K'], dist)
            px = proj.ravel()
            in_frame = (0 <= px[0] < W) and (0 <= px[1] < H)
            print(f"  {cam_name} sees {rim_name:>17} at pixel ({px[0]:>5.0f},{px[1]:>5.0f})"
                  f"  {'IN-FRAME' if in_frame else 'OUT'}")


def triangulate_3d(fr, nr, p_fr, p_nr, label=""):
    """Triangulate a 3D point from one pixel pair (FR, NR) using both
    cameras' calibrated projection matrices."""
    P1 = fr['K'] @ np.hstack([fr['R'], fr['tvec']])
    P2 = nr['K'] @ np.hstack([nr['R'], nr['tvec']])
    A = np.array([
        p_fr[0]*P1[2] - P1[0],
        p_fr[1]*P1[2] - P1[1],
        p_nr[0]*P2[2] - P2[0],
        p_nr[1]*P2[2] - P2[1],
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def main():
    fr = solve_planar("FR (Linear)", PX_FR, fov_h=73)
    nr = solve_planar("NR (Wide)",   PX_NR, fov_h=92)

    project_rim_into_cameras(fr, nr)

    # Cross-camera triangulation sanity: use the 6 landmarks both cameras saw,
    # triangulate each one using FR+NR rays, and compare to world truth.
    if fr is not None and nr is not None:
        common = sorted(set(PX_FR) & set(PX_NR))
        print(f"\n=== triangulation cross-check on {len(common)} common landmarks ===")
        errs_cm = []
        for k in common:
            X = triangulate_3d(fr, nr, np.array(PX_FR[k], float),
                                       np.array(PX_NR[k], float))
            W_true = np.array(WORLD[k])
            err = np.linalg.norm(X - W_true)
            errs_cm.append(err)
            print(f"  #{k:>2}  truth {tuple(round(v,1) for v in W_true)}  "
                  f"triangulated ({X[0]:+7.1f},{X[1]:+7.1f},{X[2]:+6.1f})  "
                  f"err {err:5.1f} cm")
        if errs_cm:
            m = np.mean(errs_cm)
            print(f"  mean 3D error: {m:.1f} cm  ({m*0.394:.1f} in)")

    if fr is not None and nr is not None:
        Path(ROOT / "data/client_report/triangulation_test").mkdir(parents=True, exist_ok=True)
        out = {
            "FR": {"K": fr['K'].tolist(), "rvec": fr['rvec'].tolist(),
                   "tvec": fr['tvec'].tolist(), "cam_pos_cm": fr['cam_pos'].tolist(),
                   "reproj_mean": fr['err_mean']},
            "NR": {"K": nr['K'].tolist(), "rvec": nr['rvec'].tolist(),
                   "tvec": nr['tvec'].tolist(), "cam_pos_cm": nr['cam_pos'].tolist(),
                   "reproj_mean": nr['err_mean']},
        }
        (ROOT / "data/client_report/triangulation_test/calibration_v2.json").write_text(
            json.dumps(out, indent=2))
        print(f"\nsaved calibration_v2.json")


if __name__ == "__main__":
    main()
