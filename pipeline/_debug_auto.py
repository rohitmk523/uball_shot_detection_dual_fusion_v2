#!/usr/bin/env python3
"""Debug calibrate_auto by:
  1. Loading the BEST existing calibration (SAM3)
  2. Re-projecting each of my detected paint corners
  3. Comparing actual paint-corner pixel position to projected expected position
This tells me if my world<->image mapping is correct, or if the corner
detection itself is off-position relative to ground truth.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_auto import (  # noqa: E402
    detect_fr_paint_corners, detect_nr_paint_corners,
    FR_CORNER_TO_LM, NR_CORNER_TO_LM,
)
from calibrate_v4 import WORLD_FLOOR  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def project(world: tuple, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
            ) -> tuple[float, float]:
    obj = np.array([world], dtype=np.float64).reshape(-1, 1, 3)
    pr, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    return float(pr[0, 0, 0]), float(pr[0, 0, 1])


for gid in GAMES:
    cal = json.loads((ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_sam3.json").read_text())
    fr_img = cv2.imread(str(ROOT / f"data/client_report/triangulation_test/june_{gid}/calib/FR_t30.jpg"))
    nr_img = cv2.imread(str(ROOT / f"data/client_report/triangulation_test/june_{gid}/calib/NR_t30.jpg"))

    fr_corners = detect_fr_paint_corners(fr_img)
    nr_corners = detect_nr_paint_corners(nr_img)
    if fr_corners is None or nr_corners is None:
        print(f"{gid}: SKIP (no corners)"); continue

    K_fr = np.array(cal['FR']['K'])
    rvec_fr = np.array(cal['FR']['rvec'])
    tvec_fr = np.array(cal['FR']['tvec'])
    K_nr = np.array(cal['NR']['K'])
    rvec_nr = np.array(cal['NR']['rvec'])
    tvec_nr = np.array(cal['NR']['tvec'])

    print(f"\n=== {gid} ===")
    print(f"  FR corners detected:")
    for corner_key, lm_id in FR_CORNER_TO_LM.items():
        x_px, y_px = fr_corners[corner_key]
        wx, wy, wz = WORLD_FLOOR[lm_id]
        ex_px, ey_px = project((wx, wy, wz), K_fr, rvec_fr, tvec_fr)
        d = float(np.hypot(x_px - ex_px, y_px - ey_px))
        print(f"    {corner_key.upper()} #{lm_id} (world={wx:.0f},{wy:.0f},{wz:.0f}): "
              f"detected=({x_px:.0f},{y_px:.0f}) "
              f"expected=({ex_px:.0f},{ey_px:.0f}) "
              f"off={d:.1f}px")
    print(f"  NR corners detected:")
    for corner_key, lm_id in NR_CORNER_TO_LM.items():
        x_px, y_px = nr_corners[corner_key]
        wx, wy, wz = WORLD_FLOOR[lm_id]
        ex_px, ey_px = project((wx, wy, wz), K_nr, rvec_nr, tvec_nr)
        d = float(np.hypot(x_px - ex_px, y_px - ey_px))
        print(f"    {corner_key.upper()} #{lm_id} (world={wx:.0f},{wy:.0f},{wz:.0f}): "
              f"detected=({x_px:.0f},{y_px:.0f}) "
              f"expected=({ex_px:.0f},{ey_px:.0f}) "
              f"off={d:.1f}px")
