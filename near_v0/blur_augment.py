#!/usr/bin/env python3
"""Motion-blur augmentation for detector v2. The ball at the rim is motion-
blurred at 30fps (the single biggest measured enemy of ball recall). For each
train frame that contains a ball, write a motion-blurred copy with the SAME
labels (linear blur doesn't move boxes). Keeps originals too, so the model sees
both sharp and blurred balls. Run in-container before training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def motion_kernel(length, angle_deg):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def main():
    split = Path(sys.argv[1])          # .../yolo_split
    imgs = split / "train/images"
    lbls = split / "train/labels"
    # deterministic per-index variety (no global RNG -> reproducible)
    lengths = [7, 9, 11, 13, 15]
    angles = [0, 30, 60, 90, 120, 150]
    n = 0
    for i, lp in enumerate(sorted(lbls.glob("*.txt"))):
        cls = [ln.split()[0] for ln in lp.read_text().splitlines() if ln.split()]
        if "0" not in cls:             # only blur ball-containing frames
            continue
        ip = imgs / f"{lp.stem}.jpg"
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        ker = motion_kernel(lengths[i % len(lengths)], angles[(i // 5) % len(angles)])
        blurred = cv2.filter2D(img, -1, ker)
        cv2.imwrite(str(imgs / f"{lp.stem}_mblur.jpg"), blurred)
        (lbls / f"{lp.stem}_mblur.txt").write_text(lp.read_text())
        n += 1
    print(f"blur-augmented {n} ball frames")


if __name__ == "__main__":
    main()
