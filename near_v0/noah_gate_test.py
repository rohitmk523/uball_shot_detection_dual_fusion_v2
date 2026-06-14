#!/usr/bin/env python3
"""Does a Noah depth-cue GATE on top of the CNN fix the depth-illusion
false-makes (clip #04) without breaking correct makes? For each e74164e6 event,
re-run the detector at the rim crossing to get ball_w/rim_w, then test:
   if CNN says MAKE but ratio < THR  ->  flip to MISS
Gate threshold taken from the held-out tuning games (ratio<0.28), NOT tuned here.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_end_to_end import crop_rect, CONF, IMGSZ, BALL, HOOP, in_zone

DET = REPO / "near_v0/weights/near_det_v1_best.pt"
GAME = "e74164e6"
VIDEO = f"{REPO}/data/client_report/triangulation_test/june_{GAME}/{GAME}_NR_full.mp4"
THR = 0.28      # from tuning games: flags 58% misses / 15% makes


def g(e):
    x = e["gt"]
    return ast.literal_eval(x) if isinstance(x, str) else x


def rim_box(det, cap, dev, n=25):
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bs = []
    for f in np.linspace(N*0.2, N*0.8, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        r = det.predict(fr, conf=0.3, imgsz=IMGSZ, device=dev, verbose=False)[0]
        hs = [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls[0]) == HOOP]
        if hs:
            bs.append(max(hs, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
    return np.median(np.array(bs), 0)


def noah_ratio(det, cap, fps, dev, t, rim):
    """ball_w/rim_w at the rim crossing (ball closest to rim center) near t."""
    cxr, cyr = (rim[0]+rim[2])/2, (rim[1]+rim[3])/2
    rimw, rimh = rim[2]-rim[0], rim[3]-rim[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t-0.7)*fps))
    best = None
    for _ in range(int(1.4*fps)):
        ok, fr = cap.read()
        if not ok:
            break
        r = det.predict(fr, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) != BALL:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bx, by = (x1+x2)/2, (y1+y2)/2
            if abs(bx-cxr) < 0.6*rimw and abs(by-cyr) < 0.7*rimh:
                d = abs(by-cyr)
                if best is None or d < best[0]:
                    best = (d, (x2-x1)/(rimw+1e-6))
    return best[1] if best else None


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = YOLO(str(DET))
    d = json.load(open(REPO / f"data/near_detector/e2e_{GAME}.json"))
    matched = [e for e in d["events"] if e.get("gt") and e["gt"] not in (None, "None")]
    by = {}
    for e in matched:
        pid = g(e)["pid"]
        if pid not in by or abs(e["t"]-g(e)["t1"]) < abs(by[pid]["t"]-g(by[pid])["t1"]):
            by[pid] = e
    matched = list(by.values())

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    rim = rim_box(det, cap, dev)

    base_correct = gate_correct = 0
    flipped_good = flipped_bad = 0
    for e in matched:
        gg = g(e)
        cnn_make = e["pred_make"]
        ratio = noah_ratio(det, cap, fps, dev, e["t"], rim)
        # gate: CNN make + low ratio -> miss
        gated_make = cnn_make
        if cnn_make and ratio is not None and ratio < THR:
            gated_make = False
            if not gg["make"]:
                flipped_good += 1     # correctly rescued a false-make
            else:
                flipped_bad += 1      # broke a true make
        base_correct += (cnn_make == gg["make"])
        gate_correct += (gated_make == gg["make"])
    cap.release()
    n = len(matched)
    print(f"e74164e6 matched shots: {n}")
    print(f"  CNN alone:       {base_correct}/{n} = {base_correct/n:.3f}")
    print(f"  CNN + Noah gate: {gate_correct}/{n} = {gate_correct/n:.3f}")
    print(f"  gate flipped {flipped_good} false-makes->correct, "
          f"broke {flipped_bad} true-makes")


if __name__ == "__main__":
    main()
