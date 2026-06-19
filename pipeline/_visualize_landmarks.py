#!/usr/bin/env python3
"""Overlay refined-click landmark positions onto each June empty-court frame
so we can see if the clicks (which came from game-1) actually land on the
correct court-line corners for the June cameras (which may have moved).
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from calibrate_v4 import (  # noqa: E402
    CLICKS_FR_FLOOR, CLICKS_NR_FLOOR, USER_FR_RIM, USER_NR_RIM,
    refine_clicks,
)

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
OUT_DIR = ROOT / "data/client_report/triangulation_test/_landmark_check"
OUT_DIR.mkdir(exist_ok=True)


def annotate(img: np.ndarray, raw: dict, refined: dict, color_raw, color_ref,
             title: str) -> np.ndarray:
    out = img.copy()
    for k in sorted(raw):
        ox, oy = int(raw[k][0]), int(raw[k][1])
        rx, ry = int(refined[k][0]), int(refined[k][1])
        cv2.circle(out, (ox, oy), 5, color_raw, 2)
        cv2.circle(out, (rx, ry), 8, color_ref, 2)
        cv2.line(out, (ox, oy), (rx, ry), (0, 255, 255), 1)
        cv2.putText(out, str(k), (rx + 10, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ref, 1)
    cv2.putText(out, title, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


for gid in GAMES:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    fr_img = cv2.imread(str(G / "calib" / "FR_t30.jpg"))
    nr_img = cv2.imread(str(G / "calib" / "NR_t30.jpg"))

    rf_fr = refine_clicks(fr_img, CLICKS_FR_FLOOR)
    rf_nr = refine_clicks(nr_img, CLICKS_NR_FLOOR)
    rf_fr_rim = refine_clicks(fr_img, USER_FR_RIM, win=7)
    rf_nr_rim = refine_clicks(nr_img, USER_NR_RIM, win=7)

    shifts_fr = [(k, np.hypot(rf_fr[k][0] - CLICKS_FR_FLOOR[k][0],
                              rf_fr[k][1] - CLICKS_FR_FLOOR[k][1]))
                 for k in CLICKS_FR_FLOOR]
    shifts_nr = [(k, np.hypot(rf_nr[k][0] - CLICKS_NR_FLOOR[k][0],
                              rf_nr[k][1] - CLICKS_NR_FLOOR[k][1]))
                 for k in CLICKS_NR_FLOOR]
    print(f"\n=== {gid} ===")
    print(f"  FR shifts: " + ", ".join(f"#{k}={d:.1f}" for k, d in shifts_fr))
    print(f"  NR shifts: " + ", ".join(f"#{k}={d:.1f}" for k, d in shifts_nr))

    title_fr = f"{gid} FR  (red=orig click, green=cornerSubPix refined)"
    title_nr = f"{gid} NR  (red=orig click, green=cornerSubPix refined)"
    fr_out = annotate(fr_img, {**CLICKS_FR_FLOOR, **USER_FR_RIM},
                       {**rf_fr, **rf_fr_rim},
                       (0, 0, 255), (0, 255, 0), title_fr)
    nr_out = annotate(nr_img, {**CLICKS_NR_FLOOR, **USER_NR_RIM},
                       {**rf_nr, **rf_nr_rim},
                       (0, 0, 255), (0, 255, 0), title_nr)
    cv2.imwrite(str(OUT_DIR / f"{gid}_FR_clicks.jpg"), fr_out,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT_DIR / f"{gid}_NR_clicks.jpg"), nr_out,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f"\nimages written to {OUT_DIR}")
