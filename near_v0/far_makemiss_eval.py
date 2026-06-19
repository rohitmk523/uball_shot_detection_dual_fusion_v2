#!/usr/bin/env python3
"""FAR-angle geometric make/miss: YOLO vs RF-DETR, apples-to-apples.

Far make/miss is PURELY geometric (pipeline/geometry_features.py): for a shot,
  g_far_dmin = min over the window of |ball_center - rim_center| / rim_width
  g_far_bdet = fraction of window frames with a ball detection
  says_miss  = (g_far_dmin > 2.0) AND (g_far_bdet > 0.6)   -> predict MISS
So far detection quality flows directly into the verdict (unlike near's
classifier). We evaluate on the GT shot windows (no spotter): for each GT shot
run the far detector over [t_start, t_end], compute says_miss, compare to GT.

Both detectors run in ONE video pass (each frame predicted by both) so the
comparison is on identical frames. Reports far make/miss accuracy per detector.

  python near_v0/far_makemiss_eval.py --game G --video FR.mp4 \
      --manifest frozen_manifests/G.json --t0 T0 --t1 T1 \
      --yolo weights/far_v16_best.pt --rfdetr weights/rfdetr_far_best.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BALL, HOOP = 0, 1
FAR_MISS_DMIN, FAR_MISS_BDET = 2.0, 0.6
CONF = 0.10                       # far uses low conf for recall (per_camera_verdict.py)
PAD = 0.4                         # widen the GT window a touch (s) for trajectory


def yolo_predict(model, dev):
    def f(frame):
        r = model.predict(frame, conf=CONF, iou=0.5, imgsz=960, device=dev, verbose=False)[0]
        return [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in r.boxes]
    return f


def rfdetr_predict(model):
    def f(frame):
        d = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=CONF)
        return [(int(d.class_id[i]), [float(v) for v in d.xyxy[i]]) for i in range(len(d.xyxy))]
    return f


def g_far_from_boxes(per_frame):
    """per_frame: list of box-lists [(cls,xyxy)]. Return (dmin, bdet, says_miss)."""
    n = len(per_frame)
    nball = 0
    dmins = []
    for boxes in per_frame:
        balls = [b for c, b in boxes if c == BALL]
        rims = [b for c, b in boxes if c == HOOP]
        if balls:
            nball += 1
        if balls and rims:
            rb = max(rims, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))  # biggest rim
            rcx, rcy, rw = (rb[0]+rb[2])/2, (rb[1]+rb[3])/2, (rb[2]-rb[0])
            if rw <= 1:
                continue
            for bb in balls:
                bcx, bcy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                dmins.append(float(np.hypot(bcx-rcx, bcy-rcy) / rw))
    bdet = nball / max(1, n)
    dmin = min(dmins) if dmins else 10.0
    says_miss = int(dmin > FAR_MISS_DMIN and bdet > FAR_MISS_BDET)
    return dmin, bdet, says_miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--rfdetr", required=True)
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    from ultralytics import YOLO
    yolo = yolo_predict(YOLO(a.yolo), dev)
    from rfdetr import RFDETRSmall
    rf = RFDETRSmall(pretrain_weights=a.rfdetr, resolution=1280)
    try:
        rf.optimize_for_inference()
    except Exception as e:
        print(f"[rfdetr] optimize skipped: {e}", flush=True)
    rfd = rfdetr_predict(rf)

    man = json.loads(Path(a.manifest).read_text())
    gt = [{"t0": float(s["t_start"]), "t1": float(s["t_end"]),
           "make": s["gt"].endswith("MAKE"), "gt": s["gt"], "pid": s["play_id"][:8]}
          for s in man if a.t0 <= float(s["t_start"]) <= a.t1]
    print(f"{a.game}: {len(gt)} GT shots in window", flush=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    rows = []
    for i, s in enumerate(gt):
        f0 = int((s["t0"] - PAD) * fps); f1 = int((s["t1"] + PAD) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
        pf_y, pf_r = [], []
        for _ in range(max(0, f1 - f0)):
            ok, fr = cap.read()
            if not ok:
                break
            pf_y.append(yolo(fr))
            pf_r.append(rfd(fr))
        yd, yb, ym = g_far_from_boxes(pf_y)
        rd, rb, rm = g_far_from_boxes(pf_r)
        rows.append({"pid": s["pid"], "make": s["make"], "gt": s["gt"],
                     "yolo": {"dmin": yd, "bdet": yb, "says_miss": ym},
                     "rfdetr": {"dmin": rd, "bdet": rb, "says_miss": rm}})
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(gt)} shots", flush=True)
    cap.release()

    def acc(key):
        # far-alone verdict: pred_miss = says_miss ; pred_make = not says_miss
        c = sum(1 for r in rows if (r[key]["says_miss"] == 1) == (not r["make"]))
        return c / max(1, len(rows))
    ya, ra = acc("yolo"), acc("rfdetr")
    # miss recall (the signal far actually contributes): of GT misses, how many says_miss
    def miss_recall(key):
        ms = [r for r in rows if not r["make"]]
        return sum(1 for r in ms if r[key]["says_miss"] == 1) / max(1, len(ms)), len(ms)
    ymr, nm = miss_recall("yolo"); rmr, _ = miss_recall("rfdetr")
    print(f"\n=== FAR make/miss (geometric) {a.game}  GT shots={len(rows)} ({nm} misses) ===")
    print(f"YOLO   far acc={ya:.3f}  miss_recall={ymr:.3f}")
    print(f"RFDETR far acc={ra:.3f}  miss_recall={rmr:.3f}")
    outp = Path(f"/work/far_{a.game}.json")
    outp.write_text(json.dumps({"game": a.game, "n": len(rows), "n_miss": nm,
                                "yolo_acc": ya, "rfdetr_acc": ra,
                                "yolo_miss_recall": ymr, "rfdetr_miss_recall": rmr,
                                "rows": rows}, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
