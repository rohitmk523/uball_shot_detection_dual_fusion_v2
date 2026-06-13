#!/usr/bin/env python3
"""END-TO-END shot-detection test on a FROZEN game (unseen by detector AND
classifier). The real product metric: from raw video, with NO ground-truth
windows, find shots and call make/miss.

Pipeline:
  1. detector over the video -> static rim + ball-in-rim events (spotter)
  2. per spotted event: build the rim-crop tensor (same recipe as training:
     crop around rim, motion-anchor, 16 gray frames @160 + diff channels)
  3. classifier_all17 -> make/miss
  4. match events to GT (no GT used in steps 1-3); report spotting recall and
     make/miss accuracy on matched events.

Usage: --game e74164e6 --video <NR.mp4> [--t0 S --t1 S] [--stride N]
"""
from __future__ import annotations

import argparse
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

DET_W = REPO / "near_v0/weights/near_det_v1_best.pt"
CLS_W = REPO / "near_v0/weights/classifier_all17.pt"
BALL, HOOP = 0, 1
CONF, IMGSZ = 0.30, 960
ZONE_ABOVE, EVENT_GAP_S, MIN_EVENT_FRAMES = 1.2, 1.2, 2
CROP_SCALE, OUT_SIZE, N_FRAMES = 1.6, 320, 16
WIN_BEFORE, WIN_AFTER = 0.7, 1.4


def in_zone(bb, rim):
    bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    x1, y1, x2, y2 = rim
    h, wdt = y2 - y1, x2 - x1
    return x1 - 0.2 * wdt <= bx <= x2 + 0.2 * wdt and (y1 - ZONE_ABOVE * h) <= by <= y2


def crop_rect(rim, W=1920, H=1080):
    cx = (rim[0] + rim[2]) / 2
    w = rim[2] - rim[0]
    side = int(CROP_SCALE * w)
    x1 = int(np.clip(cx - side / 2, 0, W - side))
    cy = (rim[1] + rim[3]) / 2
    y1 = int(np.clip(cy - side * 0.55, 0, max(0, H - side)))
    side = min(side, W - x1, H - y1)
    return x1, y1, side


def sample_idx(n, k):
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def build_clip_tensor(frames_gray, anchor, fps):
    lo = max(0, anchor - int(WIN_BEFORE * fps))
    hi = min(len(frames_gray), anchor + int(WIN_AFTER * fps))
    seg = frames_gray[lo:hi]
    idx = sample_idx(len(seg), N_FRAMES)
    x = np.stack([seg[i] for i in idx]).astype(np.float32) / 255.0
    d = np.abs(np.diff(x, axis=0))
    return torch.from_numpy(np.concatenate([x, d], 0)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    man = json.loads((REPO / f"data/client_report/triangulation_test/"
                      f"june_{args.game}/shots_right.json").read_text())
    gt = [{"t0": float(s["t_start"]), "t1": float(s["t_end"]), "gt": s["gt"],
           "make": s["gt"].endswith("MAKE"), "pid": s["play_id"][:8]}
          for s in man if args.t0 <= float(s["t_start"]) <= args.t1]

    det = YOLO(str(DET_W))
    ck = torch.load(CLS_W, map_location=dev)
    clf = build_model(ck["in_ch"]).to(dev)
    clf.load_state_dict(ck["state_dict"])
    clf.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    start_f = int(args.t0 * fps)
    end_f = int(min(args.t1, 1e9) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # pass 1: spot events + keep grayscale rim-region frames for cropping
    rim_acc = None
    hits = []                       # (t, fidx)
    gray_buf = {}                   # fidx -> gray rim-crop region (full frame gray)
    crop = None
    fidx = start_f
    while fidx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_f) % args.stride == 0:
            r = det.predict(frame, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)[0]
            balls = [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls[0]) == BALL]
            hoops = [(float(b.conf[0]), [float(v) for v in b.xyxy[0]])
                     for b in r.boxes if int(b.cls[0]) == HOOP]
            if hoops:
                hb = max(hoops)[1]
                rim_acc = np.array(hb) if rim_acc is None else 0.95 * rim_acc + 0.05 * np.array(hb)
                crop = crop_rect(rim_acc)
            if rim_acc is not None and any(in_zone(bb, rim_acc) for bb in balls):
                hits.append((fidx / fps, fidx))
        fidx += 1
    cap.release()

    # cluster into events
    events = []
    for t, fi in hits:
        if not events or t - events[-1][-1][0] > EVENT_GAP_S:
            events.append([(t, fi)])
        else:
            events[-1].append((t, fi))
    events = [e for e in events if len(e) >= MIN_EVENT_FRAMES]
    ev = [{"t": float(np.median([x[0] for x in e])),
           "f0": e[0][1], "f1": e[-1][1]} for e in events]

    # pass 2: for each event, decode its window in grayscale rim-crop & classify
    x1, y1, side = crop
    cap = cv2.VideoCapture(args.video)
    for e in ev:
        lo = max(start_f, e["f0"] - int((WIN_BEFORE + 0.6) * fps))
        hi = e["f1"] + int((WIN_AFTER + 0.6) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        gframes = []
        fi = lo
        while fi <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame[y1:y1 + side, x1:x1 + side], cv2.COLOR_BGR2GRAY)
            gframes.append(cv2.resize(g, (OUT_SIZE, OUT_SIZE)))
            fi += 1
        # motion anchor inside this window
        en = np.zeros(len(gframes))
        for i in range(1, len(gframes)):
            en[i] = float(np.mean(cv2.absdiff(
                cv2.GaussianBlur(gframes[i], (5, 5), 0),
                cv2.GaussianBlur(gframes[i - 1], (5, 5), 0))))
        anchor = int(np.argmax(en))
        gsmall = [cv2.resize(g, (160, 160)) for g in gframes]
        xb = build_clip_tensor(gsmall, anchor, fps).to(dev)
        with torch.no_grad():
            p = float(F.softmax(clf(xb), 1)[0, 1])
        e["p_make"] = round(p, 3)
        e["pred_make"] = p >= 0.5
    cap.release()

    # match events to GT (no GT used above)
    for e in ev:
        e["gt"] = None
        for g in gt:
            if g["t0"] - 2.0 <= e["t"] <= g["t1"] + 2.5:
                e["gt"] = g
                break
    matched = [e for e in ev if e["gt"]]
    matched_gt = {e["gt"]["pid"] for e in matched}
    spot_recall = len(matched_gt) / max(1, len(gt))
    spot_prec = len(matched) / max(1, len(ev))
    correct = sum(1 for e in matched if e["pred_make"] == e["gt"]["make"])
    e2e_acc = correct / max(1, len(matched))
    dur = (min(args.t1, end_f / fps) - args.t0) / 60

    print(f"\n=== END-TO-END on FROZEN {args.game} (no GT windows) ===")
    print(f"window {args.t0:.0f}-{end_f/fps:.0f}s ({dur:.1f} min)")
    print(f"GT shots={len(gt)}  spotted events={len(ev)}")
    print(f"SPOT recall={spot_recall:.3f} ({len(matched_gt)}/{len(gt)})  "
          f"precision={spot_prec:.3f} ({len(matched)}/{len(ev)})")
    print(f"END-TO-END make/miss acc on matched: {e2e_acc:.3f} ({correct}/{len(matched)})")
    # also classifier acc per outcome
    mk = [e for e in matched if e["gt"]["make"]]
    ms = [e for e in matched if not e["gt"]["make"]]
    if mk:
        print(f"  make recall: {sum(e['pred_make'] for e in mk)/len(mk):.3f} ({len(mk)} makes)")
    if ms:
        print(f"  miss recall: {sum(not e['pred_make'] for e in ms)/len(ms):.3f} ({len(ms)} misses)")
    (REPO / f"data/near_detector/e2e_{args.game}.json").write_text(json.dumps(
        {"gt_n": len(gt), "events": ev, "spot_recall": spot_recall,
         "spot_precision": spot_prec, "e2e_acc": e2e_acc}, indent=1, default=str))


if __name__ == "__main__":
    main()
