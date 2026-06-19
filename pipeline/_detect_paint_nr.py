#!/usr/bin/env python3
"""Detect the red AMG-painted key rectangle in NR (Near-Right) frames.

NR is mounted near the rim looking down at the court. The painted lane is
the bottom 2/3 of the image (with the rim and the AMG logo visible). Same
strategy as FR: HSV red mask -> largest component -> 4 extreme corners.

In NR view, the paint perspective is INVERTED relative to FR:
  - top of paint in NR image  = FT line (closer to camera, looks wider)
    Wait, actually NR is mounted NEAR the rim. So:
      - bottom of paint in NR image = closest to camera = baseline
      - top of paint in NR image    = farther from camera = FT line
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
OUT_DIR = ROOT / "data/client_report/triangulation_test/_paint_detect_nr"
OUT_DIR.mkdir(exist_ok=True)


def red_paint_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 100, 60], np.uint8),
                                np.array([10, 255, 255], np.uint8))
    high_red = cv2.inRange(hsv, np.array([160, 100, 60], np.uint8),
                                 np.array([179, 255, 255], np.uint8))
    red = cv2.bitwise_or(low_red, high_red)
    h, w = red.shape
    # NR: red paint is in lower half (between ~30% and ~95% of image height)
    crop = np.zeros_like(red)
    crop[int(h * 0.30): int(h * 0.95), :] = 255
    red = cv2.bitwise_and(red, crop)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return red


def largest_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask)
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, -1)
    return out


def corners_from_paint(paint_mask: np.ndarray) -> dict | None:
    ys, xs = np.where(paint_mask > 0)
    if len(xs) < 100:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())

    contours, _ = cv2.findContours(paint_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    if len(approx) < 4:
        return None

    def best_in_quadrant(filt):
        candidates = [p for p in approx if filt(p)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: np.hypot(p[0] - cx, p[1] - cy))

    tl = best_in_quadrant(lambda p: p[0] < cx and p[1] < cy)
    tr = best_in_quadrant(lambda p: p[0] > cx and p[1] < cy)
    bl = best_in_quadrant(lambda p: p[0] < cx and p[1] > cy)
    br = best_in_quadrant(lambda p: p[0] > cx and p[1] > cy)

    if tl is None or tr is None or bl is None or br is None:
        return None
    return dict(tl=tuple(map(float, tl)),
                tr=tuple(map(float, tr)),
                bl=tuple(map(float, bl)),
                br=tuple(map(float, br)),
                centroid=(cx, cy),
                approx_n=len(approx))


for gid in GAMES:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    img = cv2.imread(str(G / "calib" / "NR_t30.jpg"))
    red = red_paint_mask(img)
    paint = largest_component(red)
    corners = corners_from_paint(paint)
    if corners is None:
        print(f"{gid}: NO corners"); continue
    print(f"{gid}: paint area={paint.sum()/255:.0f}px  approx_n={corners['approx_n']}")
    print(f"    tl={corners['tl']} tr={corners['tr']} bl={corners['bl']} br={corners['br']}")

    vis = img.copy()
    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    for label, (cx, cy) in [("TL", corners['tl']), ("TR", corners['tr']),
                              ("BL", corners['bl']), ("BR", corners['br'])]:
        cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 3)
        cv2.putText(vis, label, (int(cx) + 12, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"{gid} NR | paint corners",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / f"{gid}_NR_paint.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT_DIR / f"{gid}_NR_paint_mask.jpg"), paint,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

print(f"\nimages -> {OUT_DIR}")
