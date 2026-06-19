#!/usr/bin/env python3
"""Detect the red AMG-painted key rectangle in FR frames. Its 4 corners are
exactly the 4 floor landmarks #3, #4, #5, #6 in WORLD_FLOOR.

Strategy:
  1. HSV mask for the red painted area (excluding red AMG logo lettering)
  2. Largest connected component = the key paint
  3. Find the 4 extreme corners (top-left, top-right, bot-left, bot-right)
     by clustering edge pixels by quadrant relative to centroid
  4. Refine each corner with cv2.cornerSubPix on the white-line edges around it
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
OUT_DIR = ROOT / "data/client_report/triangulation_test/_paint_detect_fr"
OUT_DIR.mkdir(exist_ok=True)


def red_paint_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 100, 60], np.uint8),
                                np.array([10, 255, 255], np.uint8))
    high_red = cv2.inRange(hsv, np.array([160, 100, 60], np.uint8),
                                 np.array([179, 255, 255], np.uint8))
    red = cv2.bitwise_or(low_red, high_red)
    # Crop: only middle area
    h, w = red.shape
    crop = np.zeros_like(red)
    crop[int(h * 0.30): int(h * 0.75), int(w * 0.20): int(w * 0.80)] = 255
    red = cv2.bitwise_and(red, crop)
    # Fill in small gaps but keep boundaries crisp
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
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
    """Find 4 corners (top-left, top-right, bot-left, bot-right) of the
    red paint rectangle. The paint is roughly trapezoidal in FR view
    (perspective distortion: bottom edges are LONGER, top edges shorter).
    """
    ys, xs = np.where(paint_mask > 0)
    if len(xs) < 100:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())

    # Boundary contour points
    contours, _ = cv2.findContours(paint_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    # Polynomial fit: approximate as a 4-vertex polygon
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    # If we get >4 vertices, pick the 4 that are extreme on diagonal directions
    if len(approx) < 4:
        return None

    # Score each vertex by quadrant: top-left = (x small, y small), etc.
    def best_in_quadrant(filt):
        candidates = [p for p in approx if filt(p)]
        if not candidates:
            return None
        return max(candidates,
                   key=lambda p: -(np.hypot(p[0] - cx, p[1] - cy)))

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
    img = cv2.imread(str(G / "calib" / "FR_t30.jpg"))
    red = red_paint_mask(img)
    paint = largest_component(red)
    corners = corners_from_paint(paint)
    if corners is None:
        print(f"{gid}: NO corners"); continue
    print(f"{gid}: paint area={paint.sum()/255:.0f}px  approx_n={corners['approx_n']}  "
          f"tl={corners['tl']} tr={corners['tr']} bl={corners['bl']} br={corners['br']}")

    vis = img.copy()
    # Draw paint mask outline
    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    for label, (cx, cy) in [("TL", corners['tl']), ("TR", corners['tr']),
                              ("BL", corners['bl']), ("BR", corners['br'])]:
        cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 3)
        cv2.putText(vis, label, (int(cx) + 12, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"{gid} | paint corners (TL/TR top of key = baseline, BL/BR bot = FT line)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / f"{gid}_FR_paint.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f"\nimages -> {OUT_DIR}")
