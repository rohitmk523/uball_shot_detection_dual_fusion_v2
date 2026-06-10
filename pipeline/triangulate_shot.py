#!/usr/bin/env python3
"""End-to-end triangulation pipeline for one (or more) shot windows on
5a5f1aae (FR + NR, Linear+Wide). Uses the repo's own YOLO weights:

  FR (far angle):  Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt
  NR (near angle): Uball_dual_angle_shot_detection/.../basketball_yolo11n3/weights/best.pt

Both produce classes {0: Basketball, 1: Basketball Hoop}.

Per shot window:
  1. Decode FR + NR frames (NR_frame = FR_frame + 1 sync offset).
  2. YOLO detect ball + hoop in each.
  3. For frames where ball detected in BOTH, triangulate 3D position via DLT
     using v3 PnP calibration (FR FOV=73°, NR FOV=92°, RIM_X=2008.7).
  4. Fit gravity-constrained parabola (least squares on x,y linear + z =
     z0 + vz*t - g/2*t^2).
  5. Check if trajectory crosses the rim plane (z=304.8 cm) within the
     18-inch rim circle centered at (2008.7, 713.2).
  6. Emit MAKE/MISS verdict + write trajectory JSON.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/client_report/triangulation_test/shots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FR_VIDEO = ROOT / "data/client_report/triangulation_test/5a5f1aae_FR_5min.mp4"
NR_VIDEO = ROOT / "data/client_report/triangulation_test/5a5f1aae_NR_5min.mp4"

FR_WEIGHTS = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt")
NR_WEIGHTS = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt")

# NR hoop bbox (from v4 calibration on this game's calibration frame). NR sees
# the rim near the bottom of the frame. We use this to detect "ball bounced up
# out of the rim" — a strong rim-out MISS signal that triangulation alone
# misses when the bounce happens during a detection gap.
NR_HOOP_X1, NR_HOOP_Y1, NR_HOOP_X2, NR_HOOP_Y2 = 714.0, 826.0, 1138.0, 1080.0

W, H = 1920, 1080
FPS = 29.97
SYNC_OFFSET_NR = 1   # NR_frame = FR_frame + 1

# ---- world geometry (cm, demo court convention) ----
RIM_X, RIM_Y, RIM_Z = 2008.7, 713.2, 304.8
RIM_RADIUS = 22.86  # 18-in / 2
BALL_RADIUS = 12.0  # ~24-cm men's basketball / 2
G_CM = 980.665      # gravity cm/s²

# ---- calibration v3 BEST result baked in (top=BACK, NR-right=-Y, FR FOV=73°) ----
def _K(fov_h):
    f = (W/2) / np.tan(np.radians(fov_h/2))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)

WORLD_FLOOR = {
    1:(2142.4,0.9,0.0), 2:(2142.4,1425.3,0.0), 3:(1553.0,521.6,0.0),
    4:(1553.0,901.4,0.0), 5:(2145.7,518.3,0.0), 6:(2145.7,907.9,0.0),
    7:(1369.6,708.2,0.0), 8:(1074.9,711.5,0.0), 9:(1733.1,718.0,0.0),
    10:(1549.7,711.5,0.0),
}
# rim landmarks under best interp: top=BACK (x=RIM_X-r), NR-right=-Y
WORLD_RIM = {
    11: (RIM_X, RIM_Y - RIM_RADIUS, RIM_Z),  # NR-right == FR-left
    12: (RIM_X, RIM_Y + RIM_RADIUS, RIM_Z),  # NR-left  == FR-right
    13: (RIM_X - RIM_RADIUS, RIM_Y, RIM_Z),  # top (back edge)
    14: (RIM_X, RIM_Y, RIM_Z),
}
WORLD_ALL = {**WORLD_FLOOR, **WORLD_RIM}

PX_FR = {1:(331,576),2:(1550,577),3:(709,728),4:(1178,723),5:(770,579),
         6:(1113,579),7:(947,799),8:(949,970),9:(942,672),10:(944,728),
         12:(913,307),11:(964,306),13:(940,306),14:(940,307)}
PX_NR_RAW = {1:(0,809),2:(1917,826),3:(1211,601),4:(691,595),5:(540,1075),
             6:(1346,1075),7:(952,458),8:(952,317),9:(949,823),10:(951,601),
             11:(1153,1033),12:(719,1044),13:(948,839),14:(944,1034)}
NR_SKIP = {1,2,5,6}
PX_NR = {k:v for k,v in PX_NR_RAW.items() if k not in NR_SKIP}


def solve_pnp(world, px, fov):
    keys = sorted(px)
    obj = np.array([world[k] for k in keys], dtype=np.float64)
    img = np.array([px[k]    for k in keys], dtype=np.float64)
    K = _K(fov)
    dist = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("PnP failed")
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - img, axis=1)))
    P = K @ np.hstack([R, tvec])
    cam_pos = (-R.T @ tvec).ravel()
    return dict(K=K, R=R, t=tvec, rvec=rvec, P=P, reproj=err, cam_pos=cam_pos)


def calibrate():
    """v4 CV-assisted calibration: subpixel-refined floor clicks + YOLO
    hoop-bbox-derived net-bottom landmark (#21 at z=265, breaks planar
    degeneracy). FR FOV=54° (narrow Linear), NR FOV=89° — these snap
    the cameras to physically correct mount positions.

    If env var CALIB_V5=1, load v5 (joint fx + k1 distortion fit) instead.
    """
    import os as _os
    import json as _json
    use_june_hybrid = _os.environ.get("CALIB_JUNE_HYBRID") == "1"
    if use_june_hybrid:
        # Hybrid: AUTO FR pose (auto-detected paint corners, ~11px reproj)
        # + SAM3 NR pose (existing, ~18px reproj). JSON shape matches SAM3.
        sp = _os.environ.get("CALIB_JUNE_JSON","")
        if not sp or not Path(sp).exists():
            raise RuntimeError(f"CALIB_JUNE_JSON not set or missing: {sp}")
        cal = _json.loads(Path(sp).read_text())
        floor = cal.get('cross_check_mean_cm', float('nan'))
        rim = cal.get('rim_center_cross_check_cm', float('nan'))
        print(f"[calib june-hybrid] {Path(sp).name}: SAM3-floor={floor:.1f}cm "
              f"SAM3-rim={rim:.1f}cm  AUTO-FR-reproj={cal.get('auto_FR_reproj','?')}")
        def _pack(c):
            K = np.array(c['K']); rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3,1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack(cal['FR']); nr = _pack(cal['NR'])
        print(f"[calib june-hybrid] FR reproj={fr['reproj']:.1f}px (auto)  "
              f"NR reproj={nr['reproj']:.1f}px (sam3)")
        return fr, nr
    use_june_auto = _os.environ.get("CALIB_JUNE_AUTO") == "1"
    if use_june_auto:
        # Auto-detected paint-corner calibration (no game-1 clicks).
        # Path comes from CALIB_JUNE_JSON; structure matches calibrate_auto.py
        # output: K/rvec/tvec/etc under FR and NR keys.
        sp = _os.environ.get("CALIB_JUNE_JSON","")
        if not sp or not Path(sp).exists():
            raise RuntimeError(f"CALIB_JUNE_JSON not set or missing: {sp}")
        cal = _json.loads(Path(sp).read_text())
        floor = cal.get('cross_check_floor_mean_cm', float('nan'))
        rim = cal.get('rim_center_cross_check_cm', float('nan')) or float('nan')
        print(f"[calib june-auto] {Path(sp).name}: floor={floor:.1f}cm rim={rim:.1f}cm")
        def _pack(c):
            K = np.array(c['K']); rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3,1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack(cal['FR']); nr = _pack(cal['NR'])
        print(f"[calib june-auto] FR reproj={fr['reproj']:.1f}px NR reproj={nr['reproj']:.1f}px")
        return fr, nr
    use_june_sam3 = _os.environ.get("CALIB_JUNE_SAM3") == "1"
    if use_june_sam3:
        # SAM3-augmented per-game June calibration.
        sp = _os.environ.get("CALIB_JUNE_JSON","")
        if not sp or not Path(sp).exists():
            raise RuntimeError(f"CALIB_JUNE_JSON not set or missing: {sp}")
        cal = _json.loads(Path(sp).read_text())
        print(f"[calib june-sam3] {Path(sp).name}: floor={cal['cross_check_mean_cm']:.1f}cm "
              f"rim={cal['rim_center_cross_check_cm']:.1f}cm")
        def _pack(c):
            K = np.array(c['K']); rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3,1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack(cal['FR']); nr = _pack(cal['NR'])
        print(f"[calib june-sam3] FR reproj={fr['reproj']:.1f}px NR reproj={nr['reproj']:.1f}px")
        return fr, nr
    use_june = _os.environ.get("CALIB_JUNE") == "1"
    if use_june:
        # Per-game June calibration. Path comes from CALIB_JUNE_JSON env var.
        # Same loader structure as CALIB_G3.
        june_p = _os.environ.get("CALIB_JUNE_JSON","")
        if not june_p or not Path(june_p).exists():
            raise RuntimeError(f"CALIB_JUNE_JSON not set or missing: {june_p}")
        cal = _json.loads(Path(june_p).read_text())
        print(f"[calib june] {Path(june_p).name}: cross-check={cal['cross_check_mean_cm']:.1f}cm")
        def _pack_direct(c):
            K = np.array(c['K']); rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack_direct(cal['FR']); nr = _pack_direct(cal['NR'])
        print(f"[calib june] FR reproj={fr['reproj']:.1f}px NR reproj={nr['reproj']:.1f}px")
        return fr, nr
    use_sam3 = _os.environ.get("CALIB_SAM3") == "1"
    if use_sam3:
        # SAM3-assisted game-3 calibration. Same loader as CALIB_G3 but
        # with an extra SAM3-derived precise rim-center landmark in the
        # PnP solve, dropping rim-center cross-check error from ~20cm to
        # ~5cm. Best fit for ball trajectories near the rim.
        sp = ROOT / "data/client_report/triangulation_test/calibration_v4_sam3_g3.json"
        cal = _json.loads(sp.read_text())
        print(f"[calib v4-sam3-g3] cross-check floor={cal['cross_check_mean_cm']:.1f}cm "
              f"rim={cal['rim_center_cross_check_cm']:.1f}cm")
        def _pack_direct(c):
            K = np.array(c['K'])
            rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack_direct(cal['FR'])
        nr = _pack_direct(cal['NR'])
        print(f"[calib v4-sam3-g3] FR reproj={fr['reproj']:.1f}px  "
              f"NR reproj={nr['reproj']:.1f}px")
        return fr, nr
    use_g3 = _os.environ.get("CALIB_G3") == "1"
    if use_g3:
        # Game-3-specific calibration. Pre-solved PnP saved to json; load
        # K/rvec/tvec directly without re-solving. Built from game-3 frames
        # to compensate for the 3-11 px camera drift observed between
        # game-1's calibration day and game-3's recording day.
        g3p = ROOT / "data/client_report/triangulation_test/calibration_v4_g3.json"
        cal = _json.loads(g3p.read_text())
        print(f"[calib v4-g3] cross-check mean: {cal['cross_check_mean_cm']:.1f} cm")
        def _pack_direct(c):
            K = np.array(c['K'])
            rvec = np.array(c['rvec'])
            t = np.array(c['tvec']).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            P = K @ np.hstack([R, t])
            cam_pos = (-R.T @ t).ravel()
            return dict(K=K, R=R, t=t, rvec=rvec, P=P,
                        reproj=c['reproj_mean'], cam_pos=cam_pos)
        fr = _pack_direct(cal['FR'])
        nr = _pack_direct(cal['NR'])
        print(f"[calib v4-g3] FR reproj={fr['reproj']:.1f}px "
              f"(cam X={fr['cam_pos'][0]:.0f}, h={abs(fr['cam_pos'][2])/30.48:.1f}ft)")
        print(f"[calib v4-g3] NR reproj={nr['reproj']:.1f}px "
              f"(cam X={nr['cam_pos'][0]:.0f}, h={abs(nr['cam_pos'][2])/30.48:.1f}ft)")
        return fr, nr
    use_v5 = _os.environ.get("CALIB_V5") == "1"
    if use_v5:
        v5p = ROOT / "data/client_report/triangulation_test/calibration_v5.json"
        if v5p.exists():
            v5 = _json.loads(v5p.read_text())
            print(f"[calib v5] FR FOV={v5['FR']['fov']:.1f}° k1={v5['FR']['k1']:+.3f}  "
                  f"NR FOV={v5['NR']['fov']:.1f}° k1={v5['NR']['k1']:+.3f}")
            print(f"[calib v5] cross-check landmarks: {v5['cross_check_mean_cm']:.1f} cm")
            def _pack(c):
                K = np.array(c['K']); R, _ = cv2.Rodrigues(np.array(c['rvec']))
                t = np.array(c['tvec']).reshape(3,1)
                dist = np.array(c['dist'])
                P = K @ np.hstack([R, t])
                cam_pos = (-R.T @ t).ravel()
                return dict(K=K, R=R, t=t, rvec=np.array(c['rvec']),
                            P=P, dist=dist, reproj=c['reproj_mean'],
                            cam_pos=cam_pos)
            return _pack(v5['FR']), _pack(v5['NR'])

    cal = _json.loads((ROOT / "data/client_report/triangulation_test/calibration_v4.json").read_text())
    px_fr = {int(k): tuple(v) for k, v in cal['refined_px']['FR'].items()}
    px_nr = {int(k): tuple(v) for k, v in cal['refined_px']['NR'].items()}
    # add net-bottom landmark (#21) and rim-top (#20)
    for k, info in cal['rim_landmarks']['FR'].items():
        px_fr[int(k)] = tuple(info['px']);  WORLD_ALL[int(k)] = tuple(info['world'])
    for k, info in cal['rim_landmarks']['NR'].items():
        px_nr[int(k)] = tuple(info['px']);  WORLD_ALL[int(k)] = tuple(info['world'])
    # user-clicked rim points are in WORLD_RIM already; add their refined pixels
    # (we still want them as landmarks even though the floor calibration was refined)
    for k in (11, 12, 13, 14):
        px_fr.setdefault(k, PX_FR[k])
        if k in PX_NR: px_nr.setdefault(k, PX_NR[k])
    fr = solve_pnp(WORLD_ALL, px_fr, fov=54)
    nr = solve_pnp(WORLD_ALL, px_nr, fov=89)
    print(f"[calib v4] FR reproj={fr['reproj']:.1f}px (cam X={fr['cam_pos'][0]:.0f}, h={abs(fr['cam_pos'][2])/30.48:.1f}ft)")
    print(f"[calib v4] NR reproj={nr['reproj']:.1f}px (cam X={nr['cam_pos'][0]:.0f}, h={abs(nr['cam_pos'][2])/30.48:.1f}ft)")
    return fr, nr


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


def detect_ball_hoop(model, frame, conf=0.20, imgsz=640):
    """Run YOLO; return best Basketball center+box and best Hoop box (or None)."""
    res = model.predict(frame, conf=conf, iou=0.5, imgsz=imgsz,
                        verbose=False, device='cpu')[0]
    best_ball = None
    best_hoop = None
    for box in res.boxes:
        cls = int(box.cls[0])
        c = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if cls == 0 and (best_ball is None or c > best_ball['conf']):
            best_ball = dict(conf=c, cx=cx, cy=cy, w=x2-x1, h=y2-y1)
        elif cls == 1 and (best_hoop is None or c > best_hoop['conf']):
            best_hoop = dict(conf=c, cx=cx, cy=cy, w=x2-x1, h=y2-y1)
    return best_ball, best_hoop


def fit_parabola(ts, xs, ys, zs):
    """Linear x,y vs t. Gravity-constrained z = z0 + vz*t - g/2*t²
    so fit (z + g/2*t²) = z0 + vz*t. Returns dict with v_x, v_y, v_z, x0, y0, z0,
    and a function eval(t) -> (x,y,z)."""
    ts = np.asarray(ts); xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
    Ax = np.vstack([ts, np.ones_like(ts)]).T
    vx, x0 = np.linalg.lstsq(Ax, xs, rcond=None)[0]
    vy, y0 = np.linalg.lstsq(Ax, ys, rcond=None)[0]
    z_corr = zs + 0.5*G_CM*ts**2
    vz, z0 = np.linalg.lstsq(Ax, z_corr, rcond=None)[0]

    def at(t):
        return (x0 + vx*t, y0 + vy*t, z0 + vz*t - 0.5*G_CM*t*t)

    # residuals
    xr = (x0 + vx*ts) - xs
    yr = (y0 + vy*ts) - ys
    zr = (z0 + vz*ts - 0.5*G_CM*ts**2) - zs
    rms = float(np.sqrt(np.mean(xr*xr + yr*yr + zr*zr)))
    return dict(x0=x0, y0=y0, z0=z0, vx=vx, vy=vy, vz=vz,
                rms_cm=rms, at=at)


def extract_arc(ts, zs, xs, min_apex_cm=240, t_before=0.8, t_after=0.15):
    """Locate the shot arc: keep samples in [apex-t_before, apex+t_after].
    apex = highest-z sample that's also near the rim in X (1500≤X≤2200).
    Asymmetric window: we want the full rise to apex but only a sliver after,
    because once the ball enters the net the tracker often jumps to a stray
    object — and we'd rather rely on parabola extrapolation than dirty data."""
    if len(zs) < 6:
        return np.arange(len(zs))
    order = np.argsort(zs)[::-1]
    apex = None
    for cand in order:
        if zs[cand] >= min_apex_cm and 1500 <= xs[cand] <= 2200:
            apex = int(cand); break
    if apex is None:
        apex = int(np.argmax(zs))
        if zs[apex] < min_apex_cm:
            return np.arange(len(zs))
    t_apex = ts[apex]
    mask = (ts >= t_apex - t_before) & (ts <= t_apex + t_after)
    return np.where(mask)[0]


def ransac_parabola(ts, xs, ys, zs, n_iter=200, inlier_cm=80, seed=42):
    """Sample 4-point subsets, fit gravity-constrained parabola, keep the
    subset with the most inliers, refit on inliers. Robust to detection
    noise and stray frames."""
    rng = np.random.default_rng(seed)
    n = len(ts)
    if n < 4:
        return fit_parabola(ts, xs, ys, zs), np.ones(n, bool)
    best_inl = None
    best_fit = None
    for _ in range(n_iter):
        idx = rng.choice(n, size=min(4, n), replace=False)
        try:
            f = fit_parabola(ts[idx], xs[idx], ys[idx], zs[idx])
        except np.linalg.LinAlgError:
            continue
        xp = f['x0'] + f['vx']*ts
        yp = f['y0'] + f['vy']*ts
        zp = f['z0'] + f['vz']*ts - 0.5*G_CM*ts**2
        r = np.sqrt((xp-xs)**2 + (yp-ys)**2 + (zp-zs)**2)
        inl = r <= inlier_cm
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_fit = inl, f
    if best_inl.sum() >= 4:
        best_fit = fit_parabola(ts[best_inl], xs[best_inl], ys[best_inl], zs[best_inl])
    return best_fit, best_inl


def find_rim_crossings(parab, ts):
    """Return list of (t, x, y) where parabola descends through z=RIM_Z within
    the rim radius."""
    t_min, t_max = ts.min(), ts.max() + 1.0
    candidates = np.arange(t_min, t_max, 0.005)
    z = parab['z0'] + parab['vz']*candidates - 0.5*G_CM*candidates**2
    # sign-changes near rim plane
    diff = z - RIM_Z
    crosses = []
    for i in range(1, len(diff)):
        if diff[i-1]*diff[i] < 0:
            # linear interpolate
            frac = diff[i-1] / (diff[i-1] - diff[i])
            t_x = candidates[i-1] + frac*(candidates[i]-candidates[i-1])
            x, y, _ = parab['at'](t_x)
            r = np.hypot(x - RIM_X, y - RIM_Y)
            crosses.append(dict(t=float(t_x), x=float(x), y=float(y), r=float(r),
                                inside_rim=bool(r <= RIM_RADIUS + BALL_RADIUS)))
    return crosses


# ---- Court-bounds sample sanitizer (opt-in, June depth-degeneracy fix) ----
# In the FT/3PT-arc region (ball high, far from rim) the FR/NR rays are
# near-parallel, so a few samples triangulate to physically impossible points
# (X observed up to 647 m). Those hijack apex selection (argmax z) and break
# the post-apex walker. Bounds are half-court world frame, generous margins.
_COURT_X = (800.0, 2600.0)
_COURT_Y = (0.0, 1430.0)
_COURT_Z = (-50.0, 1100.0)


def sample_in_court(s) -> bool:
    x, y, z = s['X_cm']
    return (_COURT_X[0] < x < _COURT_X[1]
            and _COURT_Y[0] < y < _COURT_Y[1]
            and _COURT_Z[0] < z < _COURT_Z[1])


def sanitize_samples(samples):
    """Drop physically-impossible triangulations (outside court bounds)."""
    return [s for s in samples if sample_in_court(s)]


def descent_verdict(samples, apex_idx, apex_dxy, apex_z,
                    rim_x=RIM_X, rim_y=RIM_Y, rim_z=RIM_Z):
    """Discriminate MAKE / MISS from post-apex trajectory shape.

    Empirically (from 10 GT-labelled shots on 5a5f1aae) the cleanest signal is
    z-bounce. After a real MAKE the ball descends monotonically until the net
    occludes it (bounce = 0). After a rim-out MISS the ball hits the rim and
    z reverses upward by 5-15 cm. A clean MISS leaves apex_dxy >> 200 cm.

    Rules:
      1. apex_dxy > 200       -> MISS (clean miss)
      2. apex_z   <  250       -> UNDECIDED (likely ball-in-hands data)
      3. clean post-apex descent + bounce >= 5 cm in rim region -> MISS (rim-out)
      4. z descended below rim plane within rim+ball tol of rim XY -> MAKE
      5. detection ended above rim plane but trajectory headed in   -> MAKE
      6. anything else -> MISS
    """
    if apex_z < 250:
        return None, f"UNDECIDED (apex z={apex_z:.0f}cm < 250cm: likely bad detections)"
    # Apex-dxy can be 400-600 cm for legitimate 3PT/4PT shots (release point is
    # far from rim and apex sits midway). Only reject if the shot really never
    # came near the rim region (XY-wise) — let descent + rim-plane analysis
    # decide for everything else.
    if apex_dxy > 1000:
        return None, f"MISS (CLEAN: apex r={apex_dxy:.0f}cm — never near rim)"
    if apex_idx >= len(samples) - 3:
        return None, "UNDECIDED (no descent data after apex)"

    # Walk post-apex for ~1s, stopping only on XY teleport (>100 cm jump = ball
    # lost / detector grabbed wrong object). z oscillation is ALLOWED — that's
    # the bounce signal we want to measure. After the walk, max(z after z_min)
    # captures the full rim-bounce arc, not just the first frame of the rise.
    # XY jump cap scales with time since last sample (detection gaps mean larger
    # apparent jumps even at realistic ball speeds). 25 m/s is faster than any
    # real basketball shot, so this caps at the physical limit.
    MAX_BALL_VEL_CMS = 2500.0
    # DESCENT_SKIP_NOISY=1: a hop/velocity violation skips the offending
    # sample instead of terminating the walk. Depth (X) noise in the
    # ill-conditioned high-arc zone is transient (single-sample spikes that
    # snap back); a real wrong-ball grab persists, so we still terminate
    # after MAX_CONSEC_SKIPS consecutive violations.
    _env = __import__("os").environ
    _skip_noisy = _env.get("DESCENT_SKIP_NOISY") == "1"
    # Sweep on June games (2026-06-10): monotonic up to 8, plateau after.
    MAX_CONSEC_SKIPS = int(_env.get("DESCENT_MAX_SKIPS", "8"))
    consec_skips = 0
    post = [samples[apex_idx]]
    pass_through_seen = False
    gap_stop_near_rim = False
    for s in samples[apex_idx+1:]:
        prev = post[-1]
        px, py, pz = prev['X_cm']
        pt = prev['t']
        x, y, z = s['X_cm']
        dt = max(s['t'] - pt, 1.0/29.97)

        # Detection gap > 150 ms while the prior frame had the ball descending
        # cleanly toward rim center (z < 360 cm, r < 45 cm) = ball just entered
        # the net and detection was lost. Anything past the gap is a different
        # ball or post-make noise. Stop the walker BEFORE the bad sample lands.
        # Convergence guard: the trailing 4 samples must show r truly
        # decreasing (median dr <= +1.5 cm/sample). FP gap-stops in game-2 all
        # had ball drifting AWAY from rim (median dr +5 to +20) right before
        # the gap, while real makes converged (median dr -2 to -8). Median is
        # used (not mean) so a single noisy detection spike doesn't flip a
        # converging trajectory to "diverging".
        prev_r = np.hypot(px - rim_x, py - rim_y)
        if dt > 0.15 and pz < 360 and prev_r < 45:
            if len(post) >= 4:
                tail = post[-4:]
                tail_r = [float(np.hypot(p['X_cm'][0]-rim_x,
                                         p['X_cm'][1]-rim_y)) for p in tail]
                tail_drs = [tail_r[k] - tail_r[k-1] for k in range(1, len(tail_r))]
                med_dr = float(np.median(tail_drs)) if tail_drs else 0.0
            else:
                med_dr = -999.0   # too few samples to judge, allow gap-stop
            if med_dr <= 1.5:
                gap_stop_near_rim = True
                break

        # absolute hop cap = 250 cm — protects against detector grabbing a
        # different basketball on the floor right after a make.
        hop = np.hypot(x-px, y-py)
        if hop > 250 or hop / dt > MAX_BALL_VEL_CMS:
            if _skip_noisy:
                consec_skips += 1
                if consec_skips > MAX_CONSEC_SKIPS:
                    break
                continue
            break
        consec_skips = 0
        post.append(s)
        # Stop AFTER recording a clean pass-through (z below rim, near rim XY).
        r_now = np.hypot(x - rim_x, y - rim_y)
        if z < rim_z - 20 and r_now < 60:
            pass_through_seen = True
            break
        if len(post) >= 30:
            break
    if len(post) < 4:
        return None, f"UNDECIDED (only {len(post)} clean post-apex samples)"

    xs = np.array([s['X_cm'][0] for s in post])
    ys = np.array([s['X_cm'][1] for s in post])
    zs = np.array([s['X_cm'][2] for s in post])

    # Rim-contact bounce detection: find the FIRST point in the rim region
    # (z in [290, 360]) where z reverses upward by >= 5 cm within the next 5
    # frames. This is the rim hit. Global z_min would be confused by ball
    # falling to floor AFTER the bounce.
    contact_idx = None
    bounce_size = 0.0
    r_at_contact = 0.0
    z_at_contact = 0.0
    for i in range(len(zs) - 1):
        if 290 <= zs[i] <= 360:
            ahead = zs[i+1:i+6]
            if len(ahead) > 0:
                rise = float(ahead.max() - zs[i])
                if rise >= 5:
                    contact_idx = i
                    bounce_size = rise
                    z_at_contact = float(zs[i])
                    r_at_contact = float(np.hypot(xs[i]-rim_x, ys[i]-rim_y))
                    break

    z_min = float(zs.min())
    z_min_idx = int(np.argmin(zs))
    r_at_zmin = float(np.hypot(xs[z_min_idx]-rim_x, ys[z_min_idx]-rim_y))
    info = dict(z_min=z_min, z_at_contact=z_at_contact,
                bounce_cm=bounce_size, r_at_contact=r_at_contact,
                r_at_zmin=r_at_zmin, n_post=len(post),
                gap_stop_near_rim=gap_stop_near_rim,
                # walk extents for downstream guards/diagnostics
                apex_t=float(samples[apex_idx]['t']),
                stop_t=float(post[-1]['t']),
                stop_xyz=tuple(map(float, post[-1]['X_cm'])))

    # DESCENT_Y_GUARD=1: lateral (Y) sanity check on MAKE claims. Y is the
    # well-conditioned triangulation axis (depth X is noisy far from the rim),
    # so a MAKE whose final approach sits laterally off rim center is depth
    # noise masquerading as convergence. TP/FP separation on June games:
    # TP y_off med 5.5cm p90 21.9cm, FP med 24.7cm p90 59.3cm.
    _y_guard = __import__("os").environ.get("DESCENT_Y_GUARD") == "1"
    y_off_stop = float(abs(post[-1]['X_cm'][1] - rim_y))
    info['y_off_stop'] = y_off_stop

    # Pre-empt all other rules: if the walker terminated because a detection gap
    # opened up while the ball was clearly converging on the rim (z < 360, r <
    # 45), that's a make whose net-occluded final approach we never saw. Any
    # earlier rim contact in the data was a rattle that ultimately dropped through.
    # Y-guard: gap-stop MAKE additionally requires the last accepted sample to
    # be laterally centered (|y - rim_y| <= 22cm). June TP gap-stop y_off max
    # is 20.1cm; FPs extend to 24.2 — threshold 22 kills 3 extra FP, 0 TP.
    # NOTE (2026-06-10 FP audit): a "post-gap reappearance contradiction"
    # guard (ball reappearing at rim height r>60, below-plane r>45, or on the
    # floor shortly after the gap => rim-out) was implemented and TESTED — it
    # is a NET REGRESSION (6 FP killed, 9+ TP killed). After a true make the
    # ball exits the net, hits the floor and bounces away, producing the SAME
    # reappearance signatures on sparse tracks. Do not re-try; see
    # fp_audit_results.json + CONTEXT.md §7.
    if gap_stop_near_rim and not (_y_guard and y_off_stop > 22):
        last_x, last_y, last_z = post[-1]['X_cm']
        last_r = float(np.hypot(last_x - rim_x, last_y - rim_y))
        return info, (f"MAKE (gap-stop: ball at z={last_z:.0f}cm r={last_r:.0f}cm "
                      f"converging on rim, detection lost in net)")

    # Rule 3 (PRECEDENCE): rim-plane crossing FIRST. If the ball actually passed
    # through the rim plane near the rim center, that's a MAKE regardless of any
    # prior rim contact (rattle-in makes are common). Interpolate (x,y) at
    # z=rim_z between the last "above rim" sample and first "below rim" sample.
    cross_r = None
    cross_xy = None
    for i in range(len(zs) - 1):
        if zs[i] >= rim_z and zs[i+1] < rim_z:
            frac = (zs[i] - rim_z) / max(zs[i] - zs[i+1], 1e-3)
            cx = xs[i] + frac*(xs[i+1] - xs[i])
            cy = ys[i] + frac*(ys[i+1] - ys[i])
            cross_r = float(np.hypot(cx - rim_x, cy - rim_y))
            cross_xy = (float(cx), float(cy))
            break
    _make_r = float(__import__("os").environ.get("MAKE_R_CM", "40"))
    if cross_r is not None and cross_r < _make_r:
        # Post-crossing continuation guard: a true make continues DOWNWARD
        # through the net (ball ends up at z << rim_z). A rim-bounce that
        # briefly clipped below rim plane will quickly come back UP to rim
        # level. Find the first below-rim sample after the cross and look
        # ahead 8 frames — if z bounces back above rim_z and stays there,
        # treat it as a rim-out rather than a make.
        cross_i = None
        for k in range(len(zs) - 1):
            if zs[k] >= rim_z and zs[k+1] < rim_z:
                cross_i = k + 1
                break
        bounce_back = False
        if cross_i is not None:
            tail = zs[cross_i:cross_i+8]
            if len(tail) >= 3:
                tail_min = float(tail.min())
                tail_end = float(tail[-1])
                # Bounce-back: ball never reaches z < rim_z - 30 (didn't
                # truly drop through net) AND ends up back at/above rim_z.
                if tail_min > rim_z - 30 and tail_end >= rim_z - 5:
                    bounce_back = True
        if bounce_back:
            info['cross_r_cm'] = cross_r
            info['cross_xy'] = cross_xy
            info['bounce_back'] = True
            # Y-guard exemption: a dead-center crossing (r<25, laterally
            # centered) is physically inside the rim mouth — the "bounce
            # back" tail is the net holding the ball at rim level, not a
            # rim-out. June: recovers 3 FN, costs 0 TN (bounce_back never
            # fires on true misses here).
            _cy = float(abs(cross_xy[1] - rim_y))
            if _y_guard and cross_r < 25 and _cy < 20:
                return info, (f"MAKE (centered crossing r={cross_r:.0f}cm "
                              f"cy={_cy:.0f}cm; net-hover tail overridden)")
            return info, (f"MISS (rim-bounce: ball clipped below rim at "
                          f"r={cross_r:.0f}cm then bounced back above rim plane)")
        info['cross_r_cm'] = cross_r
        info['cross_xy'] = cross_xy
        cross_y_off = float(abs(cross_xy[1] - rim_y))
        if contact_idx is not None:
            # Y-guard: rattled-in crossings are depth-contaminated. June data
            # separates perfectly on (crossing lateral offset, final lateral
            # offset): TP pairs all have cross_y<15 OR y_stop<20; FP pairs
            # (6/6) have both >= those. Kills 6/6 FP, 0/7 TP.
            if _y_guard and cross_y_off >= 15 and y_off_stop >= 20:
                return info, (f"MISS (rattle rejected: bounce={bounce_size:.0f}cm then "
                              f"cross at r={cross_r:.0f}cm but lateral off-center "
                              f"cross_y={cross_y_off:.0f}cm & y_off={y_off_stop:.0f}cm)")
            return info, (f"MAKE (rattled in: bounce={bounce_size:.0f}cm at z={z_at_contact:.0f}cm "
                          f"then through rim at r={cross_r:.0f}cm)")
        # Y-guard: clean pass-through claims with final sample >40cm lateral
        # off rim center are depth noise — kills 3/4 FP, 1/27 TP.
        if _y_guard and y_off_stop > 40:
            return info, (f"MISS (cross rejected: r={cross_r:.0f}cm but lateral "
                          f"off-center y_off={y_off_stop:.0f}cm > 40cm)")
        return info, (f"MAKE (passed through rim plane at r={cross_r:.0f}cm, "
                      f"z_min={z_min:.0f}cm)")

    # Rule 4: rim-contact bounce in the rim region (no clean crossing observed)
    if contact_idx is not None and r_at_contact < 150:
        return info, (f"MISS (RIM-OUT: bounce={bounce_size:.0f}cm at z={z_at_contact:.0f}cm, "
                      f"r={r_at_contact:.0f}cm)")
    if cross_r is not None:
        info['cross_r_cm'] = cross_r
        info['cross_xy'] = cross_xy
        # Physical limit: rim_radius + ball_radius = 22.86 + 12 = 34.86 cm.
        # Allow ~5 cm calibration noise -> 40 cm tolerance for MAKE.
        # Above 40, the ball center was outside the rim opening.
        if cross_r < _make_r:
            return info, (f"MAKE (passed through rim plane at r={cross_r:.0f}cm, "
                          f"z_min={z_min:.0f}cm)")
        else:
            # Y-guard lateral-crossing override: cross_r is depth(X)-driven
            # and X carries a per-game systematic bias in the descent region
            # (454da9cf: ALL crossings sit ~140cm short in X). A laterally
            # centered crossing (|cy - rim_y| <= 25cm) whose tail does NOT
            # bounce back above the rim plane is a make — trust Y over r.
            # June: recovers 17/18 cross FN, flips 4/23 TN (FT bricks that
            # fall straight down are kinematically inseparable — accepted).
            if _y_guard:
                _cy = float(abs(cross_xy[1] - rim_y))
                if _cy <= 25:
                    # front-rim-brick check: tail bouncing back above rim
                    # plane after the crossing = rim hit, not a make.
                    _ci = next((k + 1 for k in range(len(zs) - 1)
                                if zs[k] >= rim_z and zs[k+1] < rim_z), None)
                    _bb = False
                    if _ci is not None:
                        _tail = zs[_ci:_ci+8]
                        if len(_tail) >= 3 and float(_tail.min()) > rim_z - 30 \
                                and float(_tail[-1]) >= rim_z - 5:
                            _bb = True
                    if not _bb:
                        return info, (f"MAKE (lateral-centered crossing: "
                                      f"cy={_cy:.0f}cm, r={cross_r:.0f}cm is "
                                      f"depth-biased, z_min={z_min:.0f}cm)")
            return info, (f"MISS (crossed rim plane at r={cross_r:.0f}cm > "
                          f"{_make_r:.0f}cm: ball outside rim opening)")

    # Rule 5: detection ended above rim plane while trajectory headed in.
    # Trust the descent endpoint XY more than apex XY — the ball can be
    # released from far back but still come down on rim center (3PT etc).
    # Convergence guard: trailing 4 samples must show r truly closing on the
    # rim (median dr <= +1.5). FPs in game-2 had ball drifting away in the
    # last frames even though z_min looked low. Median is robust to single
    # noisy detection spikes.
    # Proximity guard: for shots released far from the rim (apex_dxy > 80 cm
    # — i.e. 3PT/long FG, not free throws), z_min must be within 50 cm of
    # rim plane. The walker for these shots can stop early at z >> rim_z
    # with the ball still drifting laterally — without proximity we'd call
    # them MAKE on stale convergence data. Free throws (apex_dxy <= 80 cm)
    # are released close enough that the walker reliably loses the ball at
    # higher z while still on a true make trajectory, so skip the guard.
    if z_min >= rim_z and r_at_zmin < 80 and apex_dxy < 600 \
            and (apex_dxy <= 80 or z_min - rim_z <= 50):
        if len(post) >= 4:
            tail_r = [float(np.hypot(p['X_cm'][0]-rim_x,
                                     p['X_cm'][1]-rim_y)) for p in post[-4:]]
            tail_drs = [tail_r[k] - tail_r[k-1] for k in range(1, len(tail_r))]
            med_dr = float(np.median(tail_drs)) if tail_drs else 0.0
        else:
            med_dr = -999.0
        if med_dr <= 1.5:
            # Y-guard: smooth-descent MAKE requires lateral centering. June
            # TP y_off max 10.3cm; FPs (incl. demoted gap-stops cascading
            # here) sit at 25-33cm. Kills 5/7 FP, 0/7 TP.
            if _y_guard and y_off_stop > 20:
                return info, (f"MISS (smooth descent rejected: z={z_min:.0f}cm "
                              f"r={r_at_zmin:.0f}cm but lateral off-center "
                              f"y_off={y_off_stop:.0f}cm > 20cm)")
            return info, (f"MAKE (smooth descent to z={z_min:.0f}cm at r={r_at_zmin:.0f}cm, "
                          f"detection lost in net)")

    # Rule 6: default to MISS
    return info, (f"MISS (z_min={z_min:.0f}cm, r={r_at_zmin:.0f}cm — no clear make signal)")


def nr_rebound_check(samples, apex_idx,
                     hx1=NR_HOOP_X1, hy1=NR_HOOP_Y1, hx2=NR_HOOP_X2, hy2=NR_HOOP_Y2):
    """Detect a rim-bounce rebound in NR by tracking ball cy after it enters the
    hoop bbox. For a MAKE, ball drops cleanly out of frame through the net —
    follow-up samples (if any) are either still near the hoop bottom or jump
    to an unrelated detection elsewhere in the frame (we filter those out).

    For a rim-out MISS, ball reaches max_cy near the hoop bottom then cy drops
    back UP within a few frames while the ball is STILL spatially near the hoop.

    Returns (rebound, max_cy, min_cy_after, t_window) for diagnostics. We only
    flag rebound if:
      - max_cy was near the hoop bottom (cy > hy2 - 100)
      - The "after" sample is within 0.4 s of max_cy moment
      - That sample is still NEAR the hoop (cy > hy1 - 50 and X within hoop ± 200)
      - The upward delta is ≥ 80 px
    """
    max_cy = -1.0; max_cy_idx = -1
    t_max = -1.0
    cy_floor = (hy1 + hy2) / 2.0
    t_apex = float(samples[apex_idx]['t'])
    for i in range(apex_idx, len(samples)):
        s = samples[i]
        t_i = float(s['t'])
        if t_i - t_apex > 1.0:
            break
        nx, ny = float(s['nr_px'][0]), float(s['nr_px'][1])
        if hx1 <= nx <= hx2 and ny >= cy_floor and ny <= hy2 + 30:
            if ny > max_cy:
                max_cy = ny; max_cy_idx = i; t_max = t_i
    if max_cy_idx < 0:
        return False, -1.0, -1.0, 0.0
    min_cy_after = max_cy
    t_min_after = t_max
    min_cy_after_idx = max_cy_idx
    for j in range(max_cy_idx + 1, len(samples)):
        s = samples[j]
        nx, ny = float(s['nr_px'][0]), float(s['nr_px'][1])
        tj = float(s['t'])
        if tj - t_max > 0.4:
            break
        if not (hx1 - 200 <= nx <= hx2 + 200):
            continue
        if ny < hy1 - 50:
            continue
        if ny < min_cy_after:
            min_cy_after = ny; t_min_after = tj; min_cy_after_idx = j
    rebound_px = max_cy - min_cy_after

    # 3D-Z guard: if the world Z keeps descending across the cy-drop window,
    # the apparent NR rebound is just the YOLO detector grabbing a different
    # object (player, basket of next play) after the real ball entered the net.
    # A TRUE rim-out has z RISING (or flat) during the bounce.
    z_at_max  = float(samples[max_cy_idx]['X_cm'][2])
    z_at_min  = float(samples[min_cy_after_idx]['X_cm'][2])
    z_descending_through_rebound = (z_at_min < z_at_max - 5.0)

    return (rebound_px >= 100.0 and not z_descending_through_rebound,
            max_cy, min_cy_after, t_min_after - t_max)


def run_shot(name, t_start_clip, t_end_clip, fr, nr, fr_model, nr_model,
             gt_label, conf=0.20, buffer_after_s=2.0,
             fr_video=None, nr_video=None, sync_offset=None, imgsz=640):
    # Empirically the plays-table annotation window ends RIGHT when the shot
    # completes (player release ≈ window end). Extending the search by ~2 s
    # lets us see the full descent + rim contact / pass-through.
    t_end_search = t_end_clip + buffer_after_s
    print(f"\n=== shot {name}  clip t={t_start_clip:.1f}–{t_end_clip:.1f}s  "
          f"(searching to {t_end_search:.1f}s)  GT={gt_label} ===")
    f_start = int(t_start_clip   * FPS)
    f_end   = int(t_end_search   * FPS)

    fr_path = Path(fr_video) if fr_video else FR_VIDEO
    nr_path = Path(nr_video) if nr_video else NR_VIDEO
    sync_off = SYNC_OFFSET_NR if sync_offset is None else sync_offset
    capFR = cv2.VideoCapture(str(fr_path))
    capNR = cv2.VideoCapture(str(nr_path))

    samples = []
    fr_frame = f_start
    capFR.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    capNR.set(cv2.CAP_PROP_POS_FRAMES, f_start + sync_off)

    while fr_frame <= f_end:
        ok1, img_fr = capFR.read()
        ok2, img_nr = capNR.read()
        if not (ok1 and ok2): break
        ball_fr, hoop_fr = detect_ball_hoop(fr_model, img_fr, conf=conf, imgsz=imgsz)
        ball_nr, hoop_nr = detect_ball_hoop(nr_model, img_nr, conf=conf, imgsz=imgsz)
        if ball_fr and ball_nr:
            # If calibration carries distortion, undistort pixels first.
            fr_px = np.array([ball_fr['cx'], ball_fr['cy']], dtype=np.float64)
            nr_px = np.array([ball_nr['cx'], ball_nr['cy']], dtype=np.float64)
            if 'dist' in fr:
                u = cv2.undistortPoints(fr_px.reshape(1,1,2).astype(np.float32),
                                         fr['K'], fr['dist'], P=fr['K'])
                fr_px = u[0,0].astype(np.float64)
            if 'dist' in nr:
                u = cv2.undistortPoints(nr_px.reshape(1,1,2).astype(np.float32),
                                         nr['K'], nr['dist'], P=nr['K'])
                nr_px = u[0,0].astype(np.float64)
            X = triangulate(fr['P'], nr['P'], fr_px, nr_px)
            samples.append(dict(
                frame=fr_frame, t=(fr_frame - f_start)/FPS,
                fr_px=(ball_fr['cx'], ball_fr['cy']), fr_conf=ball_fr['conf'],
                nr_px=(ball_nr['cx'], ball_nr['cy']), nr_conf=ball_nr['conf'],
                X_cm=X.tolist()))
        fr_frame += 1
    capFR.release(); capNR.release()

    print(f"  triangulated samples: {len(samples)} / {f_end-f_start+1} candidate frames")
    if len(samples) < 6:
        print(f"  TOO FEW samples to fit a trajectory -> VERDICT: UNDECIDED")
        r = dict(name=name, gt=gt_label, verdict="UNDECIDED",
                 n_samples=len(samples), samples=samples,
                 clip_t=[t_start_clip, t_end_clip])
        (OUT_DIR / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        return r

    ts_all = np.array([s['t']    for s in samples])
    xs_all = np.array([s['X_cm'][0] for s in samples])
    ys_all = np.array([s['X_cm'][1] for s in samples])
    zs_all = np.array([s['X_cm'][2] for s in samples])

    # bound to plausible court volume
    bounds = (xs_all > 0) & (xs_all < 2400) & (ys_all > -200) & (ys_all < 1700) \
           & (zs_all > -50) & (zs_all < 800)
    if bounds.sum() < 6:
        print(f"  too few in-bound samples ({bounds.sum()}) -> UNDECIDED")
        return dict(name=name, gt=gt_label, verdict="UNDECIDED",
                    n_samples=len(samples), samples=samples)
    ts = ts_all[bounds]; xs = xs_all[bounds]
    ys = ys_all[bounds]; zs = zs_all[bounds]

    # auto-segment the shot arc (drops the pre-shot ball-in-hands)
    arc = extract_arc(ts, zs, xs, min_apex_cm=240, t_before=0.8, t_after=0.15)
    print(f"  arc segment: idx {arc[0]}–{arc[-1]} of {len(ts)}, t={ts[arc[0]]:.2f}–{ts[arc[-1]]:.2f}s, "
          f"z_peak={zs[arc].max():.0f}cm")
    if len(arc) < 3:
        print(f"  arc too short ({len(arc)}) -> UNDECIDED");
        r = dict(name=name, gt=gt_label, verdict="UNDECIDED (arc too short)",
                 n_samples=len(samples), n_arc=int(len(arc)),
                 samples=samples, clip_t=[t_start_clip, t_end_clip])
        (OUT_DIR / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        return r

    # RANSAC parabola over the arc to absorb stray triangulations
    parab, inl = ransac_parabola(ts[arc], xs[arc], ys[arc], zs[arc],
                                  n_iter=300, inlier_cm=70)
    print(f"  RANSAC: {inl.sum()}/{len(arc)} inliers, RMS residual {parab['rms_cm']:.1f}cm ({parab['rms_cm']/2.54:.1f}in)")
    print(f"  release: x0={parab['x0']:.0f} y0={parab['y0']:.0f} z0={parab['z0']:.0f} cm   "
          f"v=({parab['vx']:.0f},{parab['vy']:.0f},{parab['vz']:.0f}) cm/s")

    # apex of the triangulated arc (robust signal)
    apex_idx = int(np.argmax(zs[arc]))
    apex = dict(x=float(xs[arc][apex_idx]), y=float(ys[arc][apex_idx]),
                z=float(zs[arc][apex_idx]),
                t=float(ts[arc][apex_idx]))
    apex['dxy_to_rim'] = float(np.hypot(apex['x']-RIM_X, apex['y']-RIM_Y))
    print(f"  apex: X={apex['x']:.0f}cm Y={apex['y']:.0f}cm Z={apex['z']:.0f}cm   "
          f"distance to rim XY = {apex['dxy_to_rim']:.0f}cm "
          f"(Y_err={apex['y']-RIM_Y:+.0f}cm, X_err={apex['x']-RIM_X:+.0f}cm)")

    crosses = find_rim_crossings(parab, ts[arc])
    # Verdict tolerance bands: tight = ball must fit through rim (r ≤ 10.86 cm
    # of rim center), permissive = include the 23.6 cm landmark calibration
    # bias measured in v3. A clean MAKE per physics is r<11; we report MAKE up
    # to r<50 with confidence levels, since 25-30 cm of systematic X-bias from
    # FR's 22 px reprojection error is real.
    # Apex-based verdict (most robust given v4's 6 cm calibration cross-check):
    # if the apex sits within 60 cm of rim center in XY AND parabola crosses
    # z=305 within 100 cm of rim center → MAKE. The 100 cm tolerance reflects
    # vx noise near apex × free-fall time from apex to rim plane.
    apex_dxy = apex['dxy_to_rim']
    apex_t_clip = apex['t']
    apex_sample_idx = min(range(len(samples)),
                          key=lambda i: abs(samples[i]['t'] - apex_t_clip))
    descent_info, descent_v = descent_verdict(samples, apex_sample_idx,
                                              apex_dxy, apex['z'])
    verdict = f"{descent_v}  [apex r={apex_dxy:.0f}cm, z_peak={apex['z']:.0f}]"

    # NR-camera rebound override: if triangulation said MAKE but NR shows the
    # ball bounced back up out of the hoop region, flip to MISS.
    # Guard: skip the override if any post-apex 3D sample has z < 150 cm —
    # the ball clearly hit the floor BELOW the rim, which can only happen if
    # it went through the net. Rim-rattle bounces stay above z=230.
    if descent_v.startswith("MAKE"):
        post_apex_z_min = min((s['X_cm'][2] for s in samples[apex_sample_idx:apex_sample_idx+30]),
                              default=9999)
        rebound, max_cy, min_cy_after, dt_reb = nr_rebound_check(samples, apex_sample_idx)
        if rebound and post_apex_z_min > 150:
            verdict = (f"MISS (NR-rebound override: cy {max_cy:.0f}→{min_cy_after:.0f} "
                       f"in {dt_reb:.2f}s, Δ={max_cy-min_cy_after:.0f}px)  "
                       f"[was: {descent_v[:50]}]")

    print(f"  VERDICT: {verdict}  |  GT: {gt_label}")
    out = dict(name=name, gt=gt_label, verdict=verdict,
               n_samples=len(samples), n_arc=int(len(arc)), n_inliers=int(inl.sum()),
               parabola={k:(v if not callable(v) else None) for k,v in parab.items() if k!='at'},
               rim_crossings=crosses, samples=samples,
               clip_t=[t_start_clip, t_end_clip])
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {out_path.name}")
    return out


SHOTS = [
    # name,           clip_t_start, clip_t_end, ground truth
    # ---- MAKES (right-angle, t=120-420s in 5a5f1aae) ----
    ("c57649e2_FT",   27.4,  33.6, "FREE_THROW_MAKE"),
    ("27002a29_FT",   38.2,  44.2, "FREE_THROW_MAKE"),
    ("eb5c3b8a_FG",   92.2,  97.8, "FG_MAKE"),
    ("22a5e7ab_FT",  146.6, 152.2, "FREE_THROW_MAKE"),
    ("b478cde3_FG",  257.8, 263.2, "FG_MAKE"),
    # ---- MISSES (right-angle, t=120-420s in 5a5f1aae) ----
    ("aa64082d_3P",   73.0,  80.0, "3PT_MISS"),
    ("f2242319_FT", 135.0, 140.8, "FREE_THROW_MISS"),
    ("c861f7a9_3P", 198.2, 205.0, "3PT_MISS"),
    ("2d060942_3P", 216.4, 222.8, "3PT_MISS"),
    ("fd830ab0_FG", 234.8, 240.4, "FG_MISS"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-sep shot names to run (default: all)")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--fr-video", help="path to FR video (default: 5-min clip)")
    ap.add_argument("--nr-video", help="path to NR video (default: 5-min clip)")
    ap.add_argument("--shots-json", help="path to JSON manifest of shots "
                    "(default: hardcoded 5-min SHOTS)")
    ap.add_argument("--clips-dir", help="dir containing per-shot {name}_FR.mp4 + "
                    "{name}_NR.mp4 clips. If set, shot t_start/t_end are interpreted "
                    "as offsets WITHIN each clip. Sync offset is set to 0 (already "
                    "baked in by the extractor).")
    ap.add_argument("--out-dir", help="output dir for per-shot JSON")
    args = ap.parse_args()

    global FR_VIDEO, NR_VIDEO, OUT_DIR
    if args.fr_video: FR_VIDEO = Path(args.fr_video)
    if args.nr_video: NR_VIDEO = Path(args.nr_video)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

    shots = SHOTS
    if args.shots_json:
        manifest = json.loads(Path(args.shots_json).read_text())
        shots = [(s['name'], s['t_start'], s['t_end'], s['gt']) for s in manifest]
        print(f"[load] {len(shots)} shots from {args.shots_json}")

    print(f"[setup] loading YOLO models ...")
    fr_model = YOLO(str(FR_WEIGHTS))
    nr_model = YOLO(str(NR_WEIGHTS))
    print(f"  FR weights: {FR_WEIGHTS.name}  classes={fr_model.names}")
    print(f"  NR weights: {NR_WEIGHTS.name}  classes={nr_model.names}")

    fr_cal, nr_cal = calibrate()

    clips_dir = Path(args.clips_dir) if args.clips_dir else None
    sel = set(args.only.split(",")) if args.only else None
    summary = []
    for name, t0, t1, gt in shots:
        if sel and name not in sel:
            continue
        fr_v = nr_v = sync = None
        if clips_dir is not None:
            fr_v = clips_dir / f"{name}_FR.mp4"
            nr_v = clips_dir / f"{name}_NR.mp4"
            if not fr_v.exists() or not nr_v.exists():
                print(f"[skip] {name}: missing clip ({fr_v.exists()=}, {nr_v.exists()=})")
                continue
            sync = 0   # NR shift already baked in by extractor
        r = run_shot(name, t0, t1, fr_cal, nr_cal, fr_model, nr_model, gt,
                     conf=args.conf, imgsz=args.imgsz,
                     fr_video=fr_v, nr_video=nr_v, sync_offset=sync)
        summary.append(dict(name=name, gt=gt,
                            verdict=r.get('verdict'),
                            n_samples=r.get('n_samples')))

    MAKE_LABELS = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
    MISS_LABELS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}
    print("\n=== SUMMARY ===")
    tp = tn = fp = fn = und = 0
    for s in summary:
        v = s['verdict'] or "?"
        gt = s['gt']
        if v.startswith("UNDECIDED"):
            mark = "—UNDECIDED"; und += 1
        elif gt in MAKE_LABELS and v.startswith("MAKE"):
            mark = "✓ TP";  tp += 1
        elif gt in MISS_LABELS and v.startswith("MISS"):
            mark = "✓ TN";  tn += 1
        elif gt in MISS_LABELS and v.startswith("MAKE"):
            mark = "✗ FP";  fp += 1
        elif gt in MAKE_LABELS and v.startswith("MISS"):
            mark = "✗ FN";  fn += 1
        else:
            mark = "?"
        print(f"  {s['name']:20s}  GT={gt:18s}  -> {v}  [{mark}]")
    decided = tp + tn + fp + fn
    print(f"\n  Decided: {decided}/{len(summary)}  Undecided: {und}")
    if decided:
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}  "
              f"Accuracy={100*(tp+tn)/decided:.1f}%")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary.json")


if __name__ == "__main__":
    main()
