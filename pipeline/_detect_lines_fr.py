#!/usr/bin/env python3
"""Detect court lines on FR (Far-Right) empty-court frame.

Goal: from a clean June-game FR frame, identify the pixel locations of:
  - baseline (back-of-key line, X=2145 in world cm)
  - free-throw line (X=1553)
  - left lane line (Y=521)
  - right lane line (Y=901)
  - left sideline / scoreboard side
  - right sideline / opposite side

Then their intersections give 4-6 floor landmarks per frame WITHOUT any
hardcoded clicks. Combined with SAM3 rim center, this is enough for PnP.

This script visualizes intermediate steps; later it gets wrapped into the
calibrate_auto.py producer.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
OUT_DIR = ROOT / "data/client_report/triangulation_test/_line_detect_fr"
OUT_DIR.mkdir(exist_ok=True)


def detect_white_lines(img: np.ndarray) -> np.ndarray:
    """Mask of bright, low-saturation pixels (white court paint)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    white = cv2.inRange(hsv, np.array([0, 0, 170], np.uint8),
                              np.array([179, 50, 255], np.uint8))
    # Mask out the upper half (where backboard/walls are) and the bottom
    # 15% (the courtside logo). Keep only court area.
    h, w = white.shape
    mask = np.zeros_like(white)
    mask[int(h * 0.40): int(h * 0.85), :] = 255
    white = cv2.bitwise_and(white, mask)
    # Clean up
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return white


def hough(white: np.ndarray, min_len: int = 80) -> list[tuple]:
    """Probabilistic Hough on the white mask."""
    edges = cv2.Canny(white, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                           minLineLength=min_len, maxLineGap=20)
    if raw is None:
        return []
    return [tuple(int(v) for v in row[0]) for row in raw]


def line_angle(x1, y1, x2, y2) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def cluster_lines(lines: list[tuple],
                   angle_low: float, angle_high: float
                   ) -> list[tuple]:
    """Pick lines whose absolute angle (mod 180) lies in [low, high]."""
    out = []
    for (x1, y1, x2, y2) in lines:
        ang = abs(line_angle(x1, y1, x2, y2))
        if ang > 90:
            ang = 180 - ang
        if angle_low <= ang <= angle_high:
            out.append((x1, y1, x2, y2))
    return out


def visualize(gid: str, img: np.ndarray, white: np.ndarray,
              all_lines: list[tuple], h_lines: list[tuple],
              v_lines: list[tuple]) -> None:
    vis = img.copy()
    for (x1, y1, x2, y2) in h_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow horizontal
    for (x1, y1, x2, y2) in v_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)    # green vertical
    cv2.putText(vis, f"{gid} | white-line Hough  H={len(h_lines)} V={len(v_lines)}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / f"{gid}_FR_lines.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT_DIR / f"{gid}_FR_white_mask.jpg"), white,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


for gid in GAMES:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    img = cv2.imread(str(G / "calib" / "FR_t30.jpg"))
    white = detect_white_lines(img)
    lines = hough(white)
    h_lines = cluster_lines(lines, 0, 25)    # near-horizontal
    v_lines = cluster_lines(lines, 65, 90)   # near-vertical
    print(f"{gid}: white mask {white.sum()/255:.0f}px  "
          f"hough {len(lines)} -> H={len(h_lines)} V={len(v_lines)}")
    visualize(gid, img, white, lines, h_lines, v_lines)

print(f"\nimages -> {OUT_DIR}")
