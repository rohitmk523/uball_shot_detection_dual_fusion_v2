#!/usr/bin/env python3
"""Quick inspect of the intrinsics_probe results — view fisheye and k1 rows."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_v4 import (  # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    WORLD_FLOOR, WORLD_RIM, refine_clicks,
)
import cv2
import numpy as np
from calibrate_intrinsics_probe import (  # noqa: E402
    solve_pinhole, solve_fisheye, cross_check, W, H,
)

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def probe(gid: str) -> None:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    fr_img = cv2.imread(str(G / "calib" / "FR_t30.jpg"))
    nr_img = cv2.imread(str(G / "calib" / "NR_t30.jpg"))
    refined_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    refined_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    refined_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    refined_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)
    px_fr = {**refined_fr, **refined_fr_rim}
    px_nr = {**refined_nr, **refined_nr_rim}
    world_all = {**WORLD_FLOOR, **WORLD_RIM}
    common_floor = sorted(set(px_fr) & set(px_nr) & set(WORLD_FLOOR))

    print(f"\n=== {gid}: fisheye sweep ===")
    for fov_fr in (105, 110, 115, 120, 122, 125):
        for fov_nr in (85, 90, 95, 100):
            fr_s = solve_fisheye(world_all, px_fr, fov=fov_fr)
            nr_s = solve_fisheye(world_all, px_nr, fov=fov_nr)
            if fr_s is None or nr_s is None:
                print(f"  fisheye {fov_fr}/{fov_nr}: FAILED to solve")
                continue
            mean_cc, max_cc = cross_check(fr_s, nr_s, px_fr, px_nr, common_floor)
            print(f"  fisheye {fov_fr}/{fov_nr}: reproj=({fr_s['mean']:.1f},{nr_s['mean']:.1f}) "
                  f"cc=({mean_cc:.1f},{max_cc:.1f})")


for g in GAMES:
    probe(g)
