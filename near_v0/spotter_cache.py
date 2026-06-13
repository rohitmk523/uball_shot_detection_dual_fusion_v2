#!/usr/bin/env python3
"""Cache the detector's raw per-frame detections over a game window, so the
spotter config can be swept offline (no re-running the detector per config).

Output JSON: {"fps":.., "t0":.., "stride":1, "rim":[x1,y1,x2,y2],
              "frames":[{"f":idx, "balls":[[x1,y1,x2,y2],...]} ...]}
Only frames with >=1 ball are stored (sparse). Rim = running-median hoop box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
BALL, HOOP = 0, 1
CONF, IMGSZ = 0.25, 960   # slightly lower conf to maximise ball recall in cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--det", default=str(REPO / "near_v0/weights/near_det_v1_best.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    det = YOLO(args.det)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    start_f = int(args.t0 * fps)
    end_f = int(min(args.t1, 1e9) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    rim_acc = None
    frames = []
    fidx = start_f
    while fidx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_f) % args.stride == 0:
            r = det.predict(frame, conf=CONF, imgsz=IMGSZ, device=dev,
                            verbose=False)[0]
            balls = [[round(float(v), 1) for v in b.xyxy[0]]
                     for b in r.boxes if int(b.cls[0]) == BALL]
            hoops = [(float(b.conf[0]), [float(v) for v in b.xyxy[0]])
                     for b in r.boxes if int(b.cls[0]) == HOOP]
            if hoops:
                hb = np.array(max(hoops)[1])
                rim_acc = hb if rim_acc is None else 0.95 * rim_acc + 0.05 * hb
            if balls:
                frames.append({"f": fidx, "balls": balls})
        fidx += 1
    cap.release()

    out = args.out or str(REPO / f"data/near_detector/spotter_cache/{args.game}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "game": args.game, "fps": fps, "t0": args.t0, "t1": args.t1,
        "stride": args.stride,
        "rim": [round(float(v), 1) for v in rim_acc] if rim_acc is not None else None,
        "frames": frames}))
    print(f"{args.game}: cached {len(frames)} ball-frames over "
          f"{(end_f-start_f)/fps/60:.1f}min -> {out}")


if __name__ == "__main__":
    main()
