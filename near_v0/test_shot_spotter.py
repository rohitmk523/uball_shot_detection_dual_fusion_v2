#!/usr/bin/env python3
"""Shot-detection TEST: does the detector find shot events from video alone,
with NO ground-truth windows? (the production-critical capability).

Run the near-angle detector over a game's NR video, locate the static rim,
and spot a "shot event" whenever the ball enters the rim zone (inside the
hoop box or within ~1.2 hoop-heights above it) after being absent. Cluster
consecutive in-zone frames into events; score event times against GT shots.

Metrics:
  - recall   = GT shots with a spotted event in [t_start-2, t_end+2.5]
  - precision= spotted events that match some GT shot
  - extra/false events per minute

Usage: --game 72c08cb7 --video <NR.mp4> [--t0 S --t1 S] [--stride N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "near_v0/weights/near_det_v1_best.pt"
BALL, HOOP = 0, 1
CONF = 0.30
IMGSZ = 960
ZONE_ABOVE = 1.2          # hoop-heights above rim still counts as "arriving"
EVENT_GAP_S = 1.2         # min gap between distinct events
MIN_EVENT_FRAMES = 2


def detect(model, frame):
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    balls, hoops = [], []
    for b in r.boxes:
        xyxy = [float(v) for v in b.xyxy[0]]
        (balls if int(b.cls[0]) == BALL else hoops).append(
            (float(b.conf[0]), xyxy))
    return balls, hoops


def in_zone(ball_box, rim):
    bx = (ball_box[0] + ball_box[2]) / 2
    by = (ball_box[1] + ball_box[3]) / 2
    x1, y1, x2, y2 = rim
    h = y2 - y1
    return x1 - 0.2 * (x2 - x1) <= bx <= x2 + 0.2 * (x2 - x1) and \
        (y1 - ZONE_ABOVE * h) <= by <= y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    man = json.loads((REPO / f"data/client_report/triangulation_test/"
                      f"june_{args.game}/shots_right.json").read_text())
    gt = [{"t0": float(s["t_start"]), "t1": float(s["t_end"]),
           "gt": s["gt"], "pid": s["play_id"][:8]} for s in man
          if args.t0 <= float(s["t_start"]) <= args.t1]

    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.t0 * fps))
    end_f = int(min(args.t1, 1e9) * fps)

    # rim = running median hoop box (static camera)
    rim_acc, hits = None, []
    fidx = int(args.t0 * fps)
    nproc = 0
    while fidx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - int(args.t0 * fps)) % args.stride == 0:
            balls, hoops = detect(model, frame)
            if hoops:
                hb = max(hoops)[1]
                rim_acc = np.array(hb) if rim_acc is None else \
                    0.98 * rim_acc + 0.02 * np.array(hb)
            if rim_acc is not None:
                for _c, bb in balls:
                    if in_zone(bb, rim_acc):
                        hits.append(fidx / fps)
                        break
            nproc += 1
        fidx += 1
    cap.release()

    # cluster hit timestamps into events
    events = []
    for t in hits:
        if not events or t - events[-1][-1] > EVENT_GAP_S:
            events.append([t])
        else:
            events[-1].append(t)
    events = [e for e in events if len(e) >= MIN_EVENT_FRAMES]
    ev_times = [float(np.median(e)) for e in events]

    # score
    matched_gt, matched_ev = set(), set()
    for gi, g in enumerate(gt):
        for ei, et in enumerate(ev_times):
            if g["t0"] - 2.0 <= et <= g["t1"] + 2.5:
                matched_gt.add(gi); matched_ev.add(ei)
    recall = len(matched_gt) / max(1, len(gt))
    prec = len(matched_ev) / max(1, len(ev_times))
    dur_min = (min(args.t1, end_f / fps) - args.t0) / 60
    print(f"\n=== SHOT-SPOTTER on {args.game} (no GT windows used to spot) ===")
    print(f"window {args.t0:.0f}-{args.t1 if args.t1<1e8 else end_f/fps:.0f}s "
          f"({dur_min:.1f} min), {nproc} frames @stride {args.stride}")
    print(f"GT shots: {len(gt)}   spotted events: {len(ev_times)}")
    print(f"RECALL  (GT shots found):     {recall:.3f} "
          f"({len(matched_gt)}/{len(gt)})")
    print(f"PRECISION(events that match): {prec:.3f} "
          f"({len(matched_ev)}/{len(ev_times)})")
    print(f"false events / min: {(len(ev_times)-len(matched_ev))/max(dur_min,1):.2f}")
    miss = [gt[i]['pid']+":"+gt[i]['gt'] for i in range(len(gt)) if i not in matched_gt]
    if miss:
        print(f"missed GT shots: {miss}")
    out = REPO / f"data/near_detector/spotter_{args.game}.json"
    out.write_text(json.dumps({"gt": gt, "events": ev_times,
                               "recall": recall, "precision": prec}, indent=1))


if __name__ == "__main__":
    main()
