#!/usr/bin/env python3
"""Diagnostic: do we have a lens-model mismatch?

Per CONTEXT.md sec 3.1: cameras are GoPro Hero 12 in Wide mode (~122 deg FOV)
but pipeline calibrated as pinhole 73 deg / 92 deg. This script keeps the SAME
landmark pixels used by current calibration and re-solves PnP with multiple
intrinsic hypotheses:

  (a) Pinhole, sweep FOV from 60..125 in 5 deg steps  (find optimal FOV)
  (b) Pinhole + Brown-Conrady k1,k2 distortion (cv2.solvePnP with full dist)
  (c) Fisheye Kannala-Brandt (cv2.fisheye.solvePnP) at the predicted ~122 deg

Reports per-game floor cross-check error for each model. Strongest signal
of lens-model mismatch = optimal FOV in (a) is far from baseline 73/92 OR
fisheye (c) cuts cross-check substantially.

Run:
  python pipeline/calibrate_intrinsics_probe.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_v4 import (  # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    WORLD_FLOOR, WORLD_RIM,
    refine_clicks,
)

W, H = 1920, 1080
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def K_from_fov(fov_deg: float, w: int = W, h: int = H) -> np.ndarray:
    f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def solve_pinhole(world: dict, px: dict, fov: float,
                  dist: np.ndarray | None = None) -> dict | None:
    """Standard cv2.solvePnP with pinhole K from FOV + optional Brown-Conrady dist."""
    keys = sorted(px)
    obj = np.array([world[k] for k in keys], dtype=np.float64)
    img = np.array([px[k] for k in keys], dtype=np.float64)
    K = K_from_fov(fov)
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    P = K @ np.hstack([R, tvec])
    return dict(K=K, R=R, t=tvec, rvec=rvec, P=P, dist=dist,
                mean=float(err.mean()), max=float(err.max()))


def solve_fisheye(world: dict, px: dict, fov: float,
                  D: np.ndarray | None = None) -> dict | None:
    """cv2.fisheye.solvePnP — Kannala-Brandt model. fov sets fx via equidistant
    approximation; D is 4-vector of fisheye distortion coefficients.
    """
    keys = sorted(px)
    obj = np.array([world[k] for k in keys], dtype=np.float64).reshape(-1, 1, 3)
    img = np.array([px[k] for k in keys], dtype=np.float64).reshape(-1, 1, 2)
    K = K_from_fov(fov)
    if D is None:
        D = np.zeros((4, 1), dtype=np.float64)
    else:
        D = np.asarray(D, dtype=np.float64).reshape(4, 1)
    try:
        ok, rvec, tvec = cv2.fisheye.solvePnP(obj, img, K, D,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error as e:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    P = K @ np.hstack([R, tvec])
    return dict(K=K, R=R, t=tvec, rvec=rvec, P=P, dist=D.ravel(),
                mean=float(err.mean()), max=float(err.max()),
                model="fisheye")


def triangulate(P1: np.ndarray, P2: np.ndarray,
                p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    A = np.array([p1[0] * P1[2] - P1[0], p1[1] * P1[2] - P1[1],
                  p2[0] * P2[2] - P2[0], p2[1] * P2[2] - P2[1]])
    _, _, V = np.linalg.svd(A)
    X = V[-1] / V[-1, 3]
    return X[:3]


def cross_check(fr_sol: dict, nr_sol: dict, px_fr: dict, px_nr: dict,
                common_floor: Sequence[int]) -> tuple[float, float]:
    errs = []
    for k in common_floor:
        X = triangulate(fr_sol['P'], nr_sol['P'],
                        np.asarray(px_fr[k], float),
                        np.asarray(px_nr[k], float))
        gt = np.asarray(WORLD_FLOOR[k], float)
        errs.append(float(np.linalg.norm(X - gt)))
    return float(np.mean(errs)), float(np.max(errs))


def probe_game(gid: str) -> dict:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    fr_path = G / "calib" / "FR_t30.jpg"
    nr_path = G / "calib" / "NR_t30.jpg"
    fr_img = cv2.imread(str(fr_path))
    nr_img = cv2.imread(str(nr_path))
    if fr_img is None or nr_img is None:
        raise FileNotFoundError(f"missing frames for {gid}")

    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    # rim ring clicks — kept for context but not used in floor cross-check
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    px_fr = {**refined_fr, **refined_fr_rim}
    px_nr = {**refined_nr, **refined_nr_rim}
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    common_floor = sorted(set(px_fr) & set(px_nr) & set(WORLD_FLOOR))

    rows = []

    # ---- (a) FOV sweep, pinhole no distortion ----
    for fov_fr in range(60, 126, 5):
        for fov_nr in (fov_fr - 5, fov_fr, fov_fr + 5, fov_fr + 10):
            if not 60 <= fov_nr <= 130:
                continue
            fr_s = solve_pinhole(world_all, px_fr, fov=fov_fr)
            nr_s = solve_pinhole(world_all, px_nr, fov=fov_nr)
            if fr_s is None or nr_s is None:
                continue
            mean_cc, max_cc = cross_check(fr_s, nr_s, px_fr, px_nr, common_floor)
            rows.append(dict(model="pinhole", fov_fr=fov_fr, fov_nr=fov_nr,
                             reproj_fr=fr_s['mean'], reproj_nr=nr_s['mean'],
                             cc_mean=mean_cc, cc_max=max_cc))

    # ---- (b) Pinhole + Brown-Conrady k1 sweep at 122 / 100 ----
    for fov_fr, fov_nr in [(122, 100), (105, 95), (95, 90)]:
        for k1 in (-0.3, -0.2, -0.1, 0.0, 0.1):
            dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
            fr_s = solve_pinhole(world_all, px_fr, fov=fov_fr, dist=dist)
            nr_s = solve_pinhole(world_all, px_nr, fov=fov_nr, dist=dist)
            if fr_s is None or nr_s is None:
                continue
            mean_cc, max_cc = cross_check(fr_s, nr_s, px_fr, px_nr, common_floor)
            rows.append(dict(model=f"pinhole+k1={k1}",
                             fov_fr=fov_fr, fov_nr=fov_nr,
                             reproj_fr=fr_s['mean'], reproj_nr=nr_s['mean'],
                             cc_mean=mean_cc, cc_max=max_cc))

    # ---- (c) Fisheye Kannala-Brandt at predicted Hero12 Wide FOV ----
    for fov_fr in (110, 115, 122, 125):
        for fov_nr in (90, 95, 100):
            fr_s = solve_fisheye(world_all, px_fr, fov=fov_fr)
            nr_s = solve_fisheye(world_all, px_nr, fov=fov_nr)
            if fr_s is None or nr_s is None:
                continue
            mean_cc, max_cc = cross_check(fr_s, nr_s, px_fr, px_nr, common_floor)
            rows.append(dict(model="fisheye",
                             fov_fr=fov_fr, fov_nr=fov_nr,
                             reproj_fr=fr_s['mean'], reproj_nr=nr_s['mean'],
                             cc_mean=mean_cc, cc_max=max_cc))

    rows.sort(key=lambda r: r['cc_mean'])
    return dict(game=gid, common_floor=common_floor, n_landmarks=len(common_floor),
                rows=rows)


def main() -> int:
    print("=" * 110)
    print(f"{'game':10s}  {'model':22s}  {'fov_FR':6s} {'fov_NR':6s}  "
          f"{'reproj_FR':>9s} {'reproj_NR':>9s}  {'cc_mean':>8s} {'cc_max':>8s}")
    print("=" * 110)
    summary = {}
    for gid in GAMES:
        result = probe_game(gid)
        baseline_pinhole_73_92 = next(
            (r for r in result['rows']
             if r['model'] == 'pinhole' and r['fov_fr'] == 75 and r['fov_nr'] == 90),
            None)
        # Top 5
        for r in result['rows'][:5]:
            tag = " <-- BEST" if r is result['rows'][0] else ""
            print(f"{gid:10s}  {r['model']:22s}  {r['fov_fr']:>6} {r['fov_nr']:>6}  "
                  f"{r['reproj_fr']:>9.2f} {r['reproj_nr']:>9.2f}  "
                  f"{r['cc_mean']:>8.2f} {r['cc_max']:>8.2f}{tag}")
        if baseline_pinhole_73_92 is not None:
            r = baseline_pinhole_73_92
            print(f"{gid:10s}  {'pinhole (baseline)':22s}  {r['fov_fr']:>6} {r['fov_nr']:>6}  "
                  f"{r['reproj_fr']:>9.2f} {r['reproj_nr']:>9.2f}  "
                  f"{r['cc_mean']:>8.2f} {r['cc_max']:>8.2f}")
        print("-" * 110)
        summary[gid] = dict(best=result['rows'][0], all_rows=result['rows'])

    # Save full results
    out_path = ROOT / "data/client_report/triangulation_test/intrinsics_probe.json"
    out_path.write_text(json.dumps(
        {g: dict(best=summary[g]['best'], top10=summary[g]['all_rows'][:10])
         for g in summary}, indent=2))
    print(f"\nfull results -> {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
