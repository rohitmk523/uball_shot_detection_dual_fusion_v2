#!/usr/bin/env python3
"""Diagnostic: classify a FROZEN game's GT shots using TRAINING-STYLE crops
(GT window + detector hoop box + training motion-anchor + training tensor).

This decouples the CLASSIFIER+CROP from the SPOTTER. If accuracy here is high
(~0.95), the classifier generalizes fine and the end-to-end gap is purely the
serving window/spotter (cheap fix). If low, the skew is deeper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_train import build_model
from build_rimcrop_dataset import crop_rect, SEG_BEFORE, SEG_AFTER, \
    WIN_BEFORE, WIN_AFTER

DET_W = REPO / "near_v0/weights/near_det_v1_best.pt"
CLS_W = REPO / "near_v0/weights/classifier_all17.pt"
N_FRAMES = 16


def sample_idx(n, k):
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def det_hoop_box(det, video, dev, n=30):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes = []
    for f in np.linspace(N * 0.2, N * 0.8, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        r = det.predict(fr, conf=0.3, imgsz=960, device=dev, verbose=False)[0]
        hs = [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls[0]) == 1]
        if hs:
            x1, y1, x2, y2 = max(hs, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            boxes.append(((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))
    cap.release()
    return np.median(np.array(boxes), 0)   # cx,cy,w,h


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "e74164e6"
    video = f"{REPO}/data/client_report/triangulation_test/june_{game}/{game}_NR_full.mp4"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = YOLO(str(DET_W))
    ck = torch.load(CLS_W, map_location=dev)
    clf = build_model(ck["in_ch"]).to(dev).eval()
    clf.load_state_dict(ck["state_dict"])

    rim = det_hoop_box(det, video, dev)
    x1, y1, side = crop_rect(rim)
    print(f"{game} detector rim box cx={rim[0]:.0f} w={rim[2]:.0f}; crop=({x1},{y1},{side})")

    man = json.loads(Path(f"{REPO}/data/client_report/triangulation_test/"
                          f"june_{game}/shots_right.json").read_text())
    shots = [{"t0": float(s["t_start"]), "t1": float(s["t_end"]),
              "make": s["gt"].endswith("MAKE"), "gt": s["gt"]} for s in man]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    correct = 0
    mk_ok = mk = ms_ok = ms = 0
    for s in shots:
        seg0 = max(0, s["t1"] - SEG_BEFORE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seg0 * fps))
        nfr = int((SEG_BEFORE + SEG_AFTER) * fps)
        gframes = []
        for _ in range(nfr):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr[y1:y1+side, x1:x1+side], cv2.COLOR_BGR2GRAY)
            gframes.append(cv2.resize(g, (160, 160)))
        if len(gframes) < int(fps):
            continue
        en = np.zeros(len(gframes))
        for i in range(1, len(gframes)):
            en[i] = float(np.mean(cv2.absdiff(
                cv2.GaussianBlur(gframes[i], (5, 5), 0),
                cv2.GaussianBlur(gframes[i-1], (5, 5), 0))))
        peak = int(np.argmax(en))
        lo = max(0, peak - int(WIN_BEFORE * fps))
        hi = min(len(gframes), peak + int(WIN_AFTER * fps))
        seg = gframes[lo:hi]
        idx = sample_idx(len(seg), N_FRAMES)
        x = np.stack([seg[i] for i in idx]).astype(np.float32) / 255.0
        d = np.abs(np.diff(x, axis=0))
        xb = torch.from_numpy(np.concatenate([x, d], 0)).unsqueeze(0).to(dev)
        with torch.no_grad():
            p = float(F.softmax(clf(xb), 1)[0, 1])
        pred = p >= 0.5
        correct += (pred == s["make"])
        if s["make"]:
            mk += 1; mk_ok += pred
        else:
            ms += 1; ms_ok += (not pred)
    cap.release()
    n = mk + ms
    print(f"\n=== CLASSIFIER on FROZEN {game}, TRAINING-STYLE crops (GT window) ===")
    print(f"n={n}  acc={correct/max(1,n):.3f}")
    print(f"  make recall={mk_ok/max(1,mk):.3f} ({mk_ok}/{mk})")
    print(f"  miss recall={ms_ok/max(1,ms):.3f} ({ms_ok}/{ms})")
    print("\n-> if acc ~0.95: classifier+crop fine, end-to-end gap = SPOTTER/window (cheap fix)")
    print("-> if acc ~0.69: deeper skew or generalization gap")


if __name__ == "__main__":
    main()
