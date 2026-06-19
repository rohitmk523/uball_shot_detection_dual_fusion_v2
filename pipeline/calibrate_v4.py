#!/usr/bin/env python3
"""Calibration v4 — CV-assisted refinement of user clicks + YOLO-derived
rim/net landmarks.

Two CV tricks:
  1. cv2.cornerSubPix refines each user click to subpixel accuracy using
     local image gradients. Cuts pixel noise where the click is on a
     real corner (painted line intersection).
  2. YOLO Basketball Hoop bbox in each camera supplies 4 spread-out
     landmarks (top of rim ring + left/right edges + net bottom) instead
     of the user's 4 nearly-coincident rim clicks. The net bottom lives
     ~40cm below the rim → different Z than the rim ring → breaks more
     planar degeneracy.

Compared to v3 (24.8 / 10.1 px reproj, 23.6 cm landmark cross-check) we
expect FR reproj to drop and X-axis bias to shrink.
"""
from __future__ import annotations
import cv2, json, numpy as np
from pathlib import Path
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
NET_LEN = 40.0   # net hangs ~40cm below rim

# Court convention floor landmarks (cm, demo court units)
WORLD_FLOOR = {
    1:(2142.4,0.9,0.0), 2:(2142.4,1425.3,0.0), 3:(1553.0,521.6,0.0),
    4:(1553.0,901.4,0.0), 5:(2145.7,518.3,0.0), 6:(2145.7,907.9,0.0),
    7:(1369.6,708.2,0.0), 8:(1074.9,711.5,0.0), 9:(1733.1,718.0,0.0),
    10:(1549.7,711.5,0.0),
}

# User's original floor clicks
CLICKS_FR_FLOOR = {1:(331,576),2:(1550,577),3:(709,728),4:(1178,723),
                   5:(770,579),6:(1113,579),7:(947,799),8:(949,970),
                   9:(942,672),10:(944,728)}
CLICKS_NR_FLOOR = {3:(1211,601),4:(691,595),7:(952,458),8:(952,317),
                   9:(949,823),10:(951,601)}


def K_from_fov(fov_h):
    f = (W/2) / np.tan(np.radians(fov_h/2))
    return np.array([[f,0,W/2],[0,f,H/2],[0,0,1]], dtype=np.float64)


def refine_clicks(img, clicks, win=15):
    """cv2.cornerSubPix: drag each pixel onto the nearest sub-pixel corner.
    No-op for points not on a real corner — and cv2 returns the input pixel
    back, so it never hurts. win=15 means search ±15 px around each click."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keys = sorted(clicks)
    pts = np.array([clicks[k] for k in keys], dtype=np.float32).reshape(-1,1,2)
    refined = cv2.cornerSubPix(gray, pts, (win, win), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return {k: (float(refined[i,0,0]), float(refined[i,0,1]))
            for i, k in enumerate(keys)}


def yolo_hoop_bbox(model, img):
    res = model.predict(img, conf=0.15, verbose=False, device='cpu')[0]
    best = None
    for b in res.boxes:
        if int(b.cls[0]) != 1: continue
        c = float(b.conf[0])
        x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
        if best is None or c > best['conf']:
            best = dict(conf=c, x1=x1,y1=y1,x2=x2,y2=y2)
    return best


# User's original rim clicks (kept; well-positioned at the rim ring level)
USER_FR_RIM = {12:(913,307), 11:(964,306), 13:(940,306), 14:(940,307)}
USER_NR_RIM = {11:(1153,1033), 12:(719,1044), 13:(948,839), 14:(944,1034)}
WORLD_RIM = {
    11: (RIM_X, RIM_Y - RIM_RADIUS, RIM_Z),
    12: (RIM_X, RIM_Y + RIM_RADIUS, RIM_Z),
    13: (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z),
    14: (RIM_X, RIM_Y, RIM_Z),
}


def fr_rim_landmarks_from_bbox(bb):
    """Use ONLY the two bbox points that map to well-defined 3D positions:
      top center = rim ring center top    -> (RIM_X, RIM_Y, RIM_Z)        [z=305]
      bot center = bottom of trailing net -> (RIM_X, RIM_Y, RIM_Z-NET_LEN)[z=265]
    These two share XY but differ by 40cm in Z — gives the key non-planar
    landmark FR has been missing. Skip bbox left/right midpoints (their 3D
    position depends on net silhouette, which is hard to model exactly).
    """
    px_top = ((bb['x1']+bb['x2'])/2, bb['y1'])
    px_bot = ((bb['x1']+bb['x2'])/2, bb['y2'])
    return {
        20: (px_top, (RIM_X, RIM_Y, RIM_Z)),
        21: (px_bot, (RIM_X, RIM_Y, RIM_Z - NET_LEN)),
    }


def nr_rim_landmarks_from_bbox(bb):
    """NR sees the rim from above. bbox top center = the rim's BACK edge
    (touching backboard, x = RIM_X - RIM_RADIUS) at z=RIM_Z. bbox bottom is
    typically at the image edge for NR (frame cut-off), so skip it.
    """
    px_top = ((bb['x1']+bb['x2'])/2, bb['y1'])
    return {
        20: (px_top, (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z)),
    }


def solve_pnp(world, px, fov):
    keys = sorted(px)
    obj = np.array([world[k] for k in keys], dtype=np.float64)
    img = np.array([px[k]    for k in keys], dtype=np.float64)
    K = K_from_fov(fov)
    dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise RuntimeError("PnP failed")
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1,2) - img, axis=1)
    return dict(K=K, R=R, t=tvec, rvec=rvec,
                P=K @ np.hstack([R, tvec]),
                cam=cam, mean=float(err.mean()), max=float(err.max()),
                per=dict(zip(keys, err.tolist())))


def triangulate(P1, P2, p1, p2):
    A = np.array([p1[0]*P1[2] - P1[0], p1[1]*P1[2] - P1[1],
                  p2[0]*P2[2] - P2[0], p2[1]*P2[2] - P2[1]])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def main():
    print("[load] images + YOLO models")
    fr_img = cv2.imread(str(FR_FRAME))
    nr_img = cv2.imread(str(NR_FRAME))
    fr_model = YOLO(str(FR_W))
    nr_model = YOLO(str(NR_W))

    # --- Step 1: subpixel-refine the user's floor clicks ---
    print("\n[step 1] subpixel-refining user floor clicks...")
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    for k in sorted(CLICKS_FR_FLOOR):
        ox, oy = CLICKS_FR_FLOOR[k]; nx, ny = refined_fr[k]
        d = np.hypot(nx-ox, ny-oy)
        print(f"  FR #{k}: ({ox},{oy}) -> ({nx:.1f},{ny:.1f}) shift={d:.1f}px")
    for k in sorted(CLICKS_NR_FLOOR):
        ox, oy = CLICKS_NR_FLOOR[k]; nx, ny = refined_nr[k]
        d = np.hypot(nx-ox, ny-oy)
        print(f"  NR #{k}: ({ox},{oy}) -> ({nx:.1f},{ny:.1f}) shift={d:.1f}px")

    # --- Step 2: YOLO hoop bbox -> rim/net landmarks ---
    print("\n[step 2] YOLO hoop -> 3D-spread rim landmarks...")
    bb_fr = yolo_hoop_bbox(fr_model, fr_img)
    bb_nr = yolo_hoop_bbox(nr_model, nr_img)
    print(f"  FR hoop bbox: ({bb_fr['x1']:.0f},{bb_fr['y1']:.0f})-({bb_fr['x2']:.0f},{bb_fr['y2']:.0f}) conf={bb_fr['conf']:.2f}")
    print(f"  NR hoop bbox: ({bb_nr['x1']:.0f},{bb_nr['y1']:.0f})-({bb_nr['x2']:.0f},{bb_nr['y2']:.0f}) conf={bb_nr['conf']:.2f}")
    rim_fr = fr_rim_landmarks_from_bbox(bb_fr)
    rim_nr = nr_rim_landmarks_from_bbox(bb_nr)

    # also subpixel-refine user's rim clicks (sit on the orange rim ring edge)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    # --- Step 3: assemble world + pixel dicts ---
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    for k, (_, w) in rim_fr.items(): world_all[k] = w
    for k, (_, w) in rim_nr.items(): world_all[k] = w

    px_fr_all = {**refined_fr, **refined_fr_rim,
                 **{k: px for k,(px,_) in rim_fr.items()}}
    px_nr_all = {**refined_nr, **refined_nr_rim,
                 **{k: px for k,(px,_) in rim_nr.items()}}

    print(f"\n[step 3] solving PnP with {len(px_fr_all)} FR + {len(px_nr_all)} NR landmarks...")
    fr = solve_pnp(world_all, px_fr_all, fov=73)
    nr = solve_pnp(world_all, px_nr_all, fov=92)
    print(f"  FR: reproj mean={fr['mean']:.1f}px max={fr['max']:.1f}px")
    print(f"  NR: reproj mean={nr['mean']:.1f}px max={nr['max']:.1f}px")
    print(f"  FR cam: X={fr['cam'][0]:+7.0f}cm Y={fr['cam'][1]:+7.0f}cm |Z|={abs(fr['cam'][2]):.0f}cm  "
          f"(h={abs(fr['cam'][2])/30.48:.1f}ft)")
    print(f"  NR cam: X={nr['cam'][0]:+7.0f}cm Y={nr['cam'][1]:+7.0f}cm |Z|={abs(nr['cam'][2]):.0f}cm  "
          f"(h={abs(nr['cam'][2])/30.48:.1f}ft)")
    print("  per-landmark FR errors:")
    for k, e in fr['per'].items():
        kind = "FLOOR" if k <= 10 else "RIM"
        print(f"    #{k:>2} [{kind:5s}]: err={e:5.1f}px")
    print("  per-landmark NR errors:")
    for k, e in nr['per'].items():
        kind = "FLOOR" if k <= 10 else "RIM"
        print(f"    #{k:>2} [{kind:5s}]: err={e:5.1f}px")

    # --- Step 4: triangulation cross-check on common floor landmarks ---
    common = sorted(set(px_fr_all) & set(px_nr_all) & set(WORLD_FLOOR))
    errs = []
    print(f"\n[step 4] triangulation cross-check on {len(common)} common floor landmarks:")
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

    # save calibration_v4.json
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
        "rim_landmarks": {
            "FR": {k: {"px": px, "world": w} for k,(px,w) in rim_fr.items()},
            "NR": {k: {"px": px, "world": w} for k,(px,w) in rim_nr.items()},
        },
        "cross_check_mean_cm": float(np.mean(errs)) if errs else None,
    }
    (ROOT / "data/client_report/triangulation_test/calibration_v4.json").write_text(
        json.dumps(out, indent=2))
    print("\nsaved calibration_v4.json")


if __name__ == "__main__":
    main()
