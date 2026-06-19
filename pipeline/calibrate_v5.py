#!/usr/bin/env python3
"""Calibration v5 — jointly optimizes intrinsics (focal length), extrinsics
(rotation + translation), AND radial lens distortion (k1, k2) using
scipy.optimize.least_squares.

The v4 calibration assumed a pinhole model and we observed ~50-120 cm X-bias
on far-court shot trajectories (4PT MISSES called MAKES, FT MAKES called
MISSES). That's the distortion signature: landmarks near the rim (the
calibration anchors) reproject well, but the lens distortion at the image
edges shifts ball positions on far-court trajectories.

Inputs: same 14 landmarks per camera as v4 (10 floor + 4 rim).
Outputs: calibration_v5.json with K, R, t, k1, k2 per camera.
"""
from __future__ import annotations
import cv2, json, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "data/client_report/triangulation_test/calib"
FR_FRAME = CALIB_DIR / "FR_t006.0.jpg"
NR_FRAME = CALIB_DIR / "NR_t006.0.jpg"
FR_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt")
NR_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt")
W, H = 1920, 1080

RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8
RIM_RADIUS = 22.86
NET_LEN = 40.0

WORLD_FLOOR = {
    1:(2142.4,0.9,0.0), 2:(2142.4,1425.3,0.0), 3:(1553.0,521.6,0.0),
    4:(1553.0,901.4,0.0), 5:(2145.7,518.3,0.0), 6:(2145.7,907.9,0.0),
    7:(1369.6,708.2,0.0), 8:(1074.9,711.5,0.0), 9:(1733.1,718.0,0.0),
    10:(1549.7,711.5,0.0),
}
WORLD_RIM = {
    11: (RIM_X, RIM_Y - RIM_RADIUS, RIM_Z),
    12: (RIM_X, RIM_Y + RIM_RADIUS, RIM_Z),
    13: (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z),
    14: (RIM_X, RIM_Y, RIM_Z),
}

CLICKS_FR_FLOOR = {1:(331,576),2:(1550,577),3:(709,728),4:(1178,723),
                   5:(770,579),6:(1113,579),7:(947,799),8:(949,970),
                   9:(942,672),10:(944,728)}
CLICKS_NR_FLOOR = {3:(1211,601),4:(691,595),7:(952,458),8:(952,317),
                   9:(949,823),10:(951,601)}
USER_FR_RIM = {12:(913,307), 11:(964,306), 13:(940,306), 14:(940,307)}
USER_NR_RIM = {11:(1153,1033), 12:(719,1044), 13:(948,839), 14:(944,1034)}


def refine_clicks(img: np.ndarray, clicks: dict, win: int = 15) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keys = sorted(clicks)
    pts = np.array([clicks[k] for k in keys], dtype=np.float32).reshape(-1,1,2)
    refined = cv2.cornerSubPix(gray, pts, (win, win), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return {k: (float(refined[i,0,0]), float(refined[i,0,1]))
            for i, k in enumerate(keys)}


def yolo_hoop_bbox(model: YOLO, img: np.ndarray) -> dict:
    res = model.predict(img, conf=0.15, verbose=False, device="cpu")[0]
    best = None
    for b in res.boxes:
        if int(b.cls[0]) != 1: continue
        c = float(b.conf[0])
        x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
        if best is None or c > best["conf"]:
            best = dict(conf=c, x1=x1,y1=y1,x2=x2,y2=y2)
    return best


def K_from_fx(fx: float) -> np.ndarray:
    return np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)


def fx_from_fov(fov_deg: float) -> float:
    return (W/2) / np.tan(np.radians(fov_deg/2))


def initial_pnp(world_pts: np.ndarray, img_pts: np.ndarray, fov_init: float):
    """Pinhole-model PnP for an initial guess."""
    K = K_from_fx(fx_from_fov(fov_init))
    dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(world_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec, K[0,0]


def joint_calibrate(world_pts: np.ndarray, img_pts: np.ndarray,
                    fov_init: float, fx_bounds=(200.0, 3000.0),
                    k1_bounds=(-0.4, 0.4), fit_k2: bool = False) -> dict:
    """Fit fx + (rvec, tvec) + (k1, k2) jointly by minimizing reprojection
    error. Returns dict with K, R, t, dist, reproj stats."""
    rvec0, tvec0, fx0 = initial_pnp(world_pts, img_pts, fov_init)
    n_k = 2 if fit_k2 else 1
    theta0 = np.concatenate([
        np.array([fx0] + [0.0]*n_k),
        rvec0.ravel(), tvec0.ravel()
    ])

    def unpack(theta):
        fx = theta[0]
        if fit_k2:
            k1, k2 = theta[1], theta[2]; off = 3
        else:
            k1, k2 = theta[1], 0.0; off = 2
        rvec = theta[off:off+3].reshape(3,1)
        tvec = theta[off+3:off+6].reshape(3,1)
        return fx, k1, k2, rvec, tvec

    def residuals(theta):
        fx, k1, k2, rvec, tvec = unpack(theta)
        K = K_from_fx(fx)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0])
        proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
        return (proj.reshape(-1,2) - img_pts).ravel()

    lo = [fx_bounds[0], k1_bounds[0]]
    hi = [fx_bounds[1], k1_bounds[1]]
    if fit_k2:
        lo.append(-0.2); hi.append(0.2)
    lo.extend([-np.inf]*6); hi.extend([np.inf]*6)
    lo = np.array(lo); hi = np.array(hi)

    result = least_squares(residuals, theta0, bounds=(lo, hi),
                           method="trf", loss="linear",
                           xtol=1e-10, ftol=1e-10, max_nfev=2000)
    fx, k1, k2, rvec, tvec = unpack(result.x)
    fx = float(fx)
    K = K_from_fx(fx)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()
    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
    per = np.linalg.norm(proj.reshape(-1,2) - img_pts, axis=1)
    return dict(K=K, R=R, t=tvec, rvec=rvec, dist=dist,
                cam=cam, fx=fx, k1=k1, k2=k2,
                fov=float(np.degrees(2*np.arctan((W/2)/fx))),
                reproj_mean=float(per.mean()), reproj_max=float(per.max()),
                per=per.tolist())


def main():
    print("[load] images + YOLO models")
    fr_img = cv2.imread(str(FR_FRAME))
    nr_img = cv2.imread(str(NR_FRAME))
    fr_model = YOLO(str(FR_W)); nr_model = YOLO(str(NR_W))

    print("[step1] subpixel-refining clicks")
    fr_floor = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    nr_floor = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    print("[step2] YOLO hoop bbox -> net-bottom landmark (#21)")
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)
    px_fr_top  = ((bb_fr['x1']+bb_fr['x2'])/2, bb_fr['y1'])
    px_fr_bot  = ((bb_fr['x1']+bb_fr['x2'])/2, bb_fr['y2'])
    px_nr_top  = ((bb_nr['x1']+bb_nr['x2'])/2, bb_nr['y1'])
    extra_fr = {20: (px_fr_top, (RIM_X, RIM_Y, RIM_Z)),
                21: (px_fr_bot, (RIM_X, RIM_Y, RIM_Z - NET_LEN))}
    extra_nr = {20: (px_nr_top, (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z))}

    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    for k,(_,w) in extra_fr.items(): world_all[k] = w
    for k,(_,w) in extra_nr.items(): world_all[k] = w
    px_fr_all = {**fr_floor, **fr_rim, **{k:px for k,(px,_) in extra_fr.items()}}
    px_nr_all = {**nr_floor, **nr_rim, **{k:px for k,(px,_) in extra_nr.items()}}

    # Stack into arrays
    fr_keys = sorted(px_fr_all)
    nr_keys = sorted(px_nr_all)
    fr_world = np.array([world_all[k] for k in fr_keys], dtype=np.float64)
    fr_img_pts = np.array([px_fr_all[k] for k in fr_keys], dtype=np.float64)
    nr_world = np.array([world_all[k] for k in nr_keys], dtype=np.float64)
    nr_img_pts = np.array([px_nr_all[k] for k in nr_keys], dtype=np.float64)

    print("\n[step3] joint optimization (fx + k1 + k2 + rvec + tvec)")
    fr = joint_calibrate(fr_world, fr_img_pts, fov_init=54)
    nr = joint_calibrate(nr_world, nr_img_pts, fov_init=89)

    print(f"\n  FR: fov={fr['fov']:.1f}°, fx={fr['fx']:.0f}px, "
          f"k1={fr['k1']:+.4f}, k2={fr['k2']:+.4f}")
    print(f"      reproj mean={fr['reproj_mean']:.2f}px (max {fr['reproj_max']:.1f})")
    print(f"      cam pos (cm): X={fr['cam'][0]:+.0f} Y={fr['cam'][1]:+.0f} "
          f"|Z|={abs(fr['cam'][2]):.0f}  (h={abs(fr['cam'][2])/30.48:.1f}ft)")
    print(f"\n  NR: fov={nr['fov']:.1f}°, fx={nr['fx']:.0f}px, "
          f"k1={nr['k1']:+.4f}, k2={nr['k2']:+.4f}")
    print(f"      reproj mean={nr['reproj_mean']:.2f}px (max {nr['reproj_max']:.1f})")
    print(f"      cam pos (cm): X={nr['cam'][0]:+.0f} Y={nr['cam'][1]:+.0f} "
          f"|Z|={abs(nr['cam'][2]):.0f}  (h={abs(nr['cam'][2])/30.48:.1f}ft)")

    # 3D triangulation cross-check
    P_fr = fr['K'] @ np.hstack([fr['R'], fr['t']])
    P_nr = nr['K'] @ np.hstack([nr['R'], nr['t']])
    common = sorted(set(fr_keys) & set(nr_keys) & set(WORLD_FLOOR.keys()))
    errs = []
    print(f"\n[step4] 3D cross-check on {len(common)} common floor landmarks "
          f"(uses undistorted pixels):")
    for k in common:
        # undistort the pixel from each camera first
        pt_fr = np.array([[px_fr_all[k]]], dtype=np.float32)
        pt_nr = np.array([[px_nr_all[k]]], dtype=np.float32)
        u_fr = cv2.undistortPoints(pt_fr, fr['K'], fr['dist'], P=fr['K'])[0,0]
        u_nr = cv2.undistortPoints(pt_nr, nr['K'], nr['dist'], P=nr['K'])[0,0]
        A = np.array([
            u_fr[0]*P_fr[2] - P_fr[0], u_fr[1]*P_fr[2] - P_fr[1],
            u_nr[0]*P_nr[2] - P_nr[0], u_nr[1]*P_nr[2] - P_nr[1],
        ])
        _,_,V = np.linalg.svd(A); X = V[-1]/V[-1,3]
        wt = np.array(WORLD_FLOOR[k])
        e = float(np.linalg.norm(X[:3] - wt))
        errs.append(e)
        print(f"    #{k:>2}  truth ({wt[0]:.0f},{wt[1]:.0f},{wt[2]:.0f}) -> "
              f"({X[0]:+.0f},{X[1]:+.0f},{X[2]:+.0f})  err {e:.1f} cm")
    print(f"  MEAN 3D cross-check error: {np.mean(errs):.1f} cm "
          f"({np.mean(errs)*0.394:.1f}in)")

    out = {
        "version": "v5_distortion",
        "FR": {"K": fr['K'].tolist(), "rvec": fr['rvec'].tolist(),
               "tvec": fr['t'].tolist(), "dist": fr['dist'].tolist(),
               "cam_cm": fr['cam'].tolist(), "reproj_mean": fr['reproj_mean'],
               "fov": fr['fov'], "k1": fr['k1'], "k2": fr['k2']},
        "NR": {"K": nr['K'].tolist(), "rvec": nr['rvec'].tolist(),
               "tvec": nr['t'].tolist(), "dist": nr['dist'].tolist(),
               "cam_cm": nr['cam'].tolist(), "reproj_mean": nr['reproj_mean'],
               "fov": nr['fov'], "k1": nr['k1'], "k2": nr['k2']},
        "cross_check_mean_cm": float(np.mean(errs)),
    }
    out_path = ROOT / "data/client_report/triangulation_test/calibration_v5.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
