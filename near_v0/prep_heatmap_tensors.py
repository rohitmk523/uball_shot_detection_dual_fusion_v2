#!/usr/bin/env python3
"""Pre-calculate and cache ball-detection heatmap clips for training.

For each usable rim-crop clip in dataset_index.json:
  - Load the 320x320 video frames
  - Run the fine-tuned YOLO detector to locate the ball
  - Create a 160x160 2D Gaussian heatmap centered on the ball center
  - Sample N_FRAMES (16) frames uniformly
  - Save as a consolidated numpy tensor [N, 16, 160, 160] uint8
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
RC = REPO / "data/near_rimcrop"
CACHE = RC / "cache"
DET_W = REPO / "near_v0/weights/near_det_v1_best.pt"
N_FRAMES = 16
RES = 160
BALL = 0


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_idx(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def make_gaussian_heatmap(cx, cy, res=160, sigma=10.0):
    x = np.arange(0, res, 1, float)
    y = np.arange(0, res, 1, float)
    x, y = np.meshgrid(x, y)
    h = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return (h * 255).astype(np.uint8)


def load_clip_heatmap(path: Path, model, dev) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None

    # Batch prediction for speed
    results = model.predict(frames, conf=0.15, imgsz=320, device=dev, verbose=False)
    heatmaps = []
    for r in results:
        balls = []
        for b in r.boxes:
            if int(b.cls[0]) == BALL:
                balls.append((float(b.conf[0]), [float(v) for v in b.xyxy[0]]))
        
        h_mask = np.zeros((RES, RES), dtype=np.uint8)
        if balls:
            best_ball = max(balls)[1]
            bx1, by1, bx2, by2 = best_ball
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            # Map 320x320 crop coordinates to RESxRES (160x160)
            scale = RES / 320.0
            cx = bcx * scale
            cy = bcy * scale
            h_mask = make_gaussian_heatmap(cx, cy, res=RES, sigma=10.0)
        heatmaps.append(h_mask)

    idx = sample_idx(len(heatmaps), N_FRAMES)
    return np.stack([heatmaps[i] for i in idx]).astype(np.uint8)


def main():
    if not DET_W.exists():
        print(f"Weights not found at {DET_W}", file=sys.stderr)
        return 1

    idx = json.loads((RC / "dataset_index.json").read_text())
    recs = [r for r in idx["records"] if r["usable"]]
    
    dev = device()
    print(f"Loading YOLO from {DET_W} on {dev}...")
    model = YOLO(str(DET_W))
    
    print(f"Processing {len(recs)} clips...")
    arr = np.zeros((len(recs), N_FRAMES, RES, RES), dtype=np.uint8)
    ok = 0
    for i, r in enumerate(recs):
        clip_path = REPO / r["path"]
        h_clip = load_clip_heatmap(clip_path, model, dev)
        if h_clip is None:
            continue
        arr[ok] = h_clip
        ok += 1
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(recs)} clips...", flush=True)
            
    arr = arr[:ok]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "heatmaps_u8.npy", arr)
    print(f"Saved {ok}/{len(recs)} heatmaps to {CACHE / 'heatmaps_u8.npy'} of shape {arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
