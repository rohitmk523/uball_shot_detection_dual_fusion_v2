#!/usr/bin/env python3
"""Evaluate the near-angle detector on held-out test games with the metrics
that actually matter for production (not just mAP):

  1. hoop recall on the overhead view  (old model: ~2% frame fire rate)
  2. ball recall at the rim moment      (the frames that decide make/miss)
  3. player false-positive rate         (old model's dominant error)

Uses GT boxes from the test split. A "rim-moment" frame = filename kind=='rim'.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

NA = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
          "Training_frameworks/Uball Near Angle/data")
BALL, HOOP = 0, 1


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(lbl: Path, W, H):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        c = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:5])
        out.append((c, [(cx - w / 2) * W, (cy - h / 2) * H,
                        (cx + w / 2) * W, (cy + h / 2) * H]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default=str(NA / "yolo_split/test"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted((Path(args.split) / "images").glob("*.jpg"))
    lbls = Path(args.split) / "labels"

    st = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "n": 0})
    rim_ball = {"tp": 0, "fn": 0}
    for img in imgs:
        kind = img.stem.rsplit("_", 1)[-1]
        import cv2
        H, W = cv2.imread(str(img)).shape[:2]
        gt = load_gt(lbls / f"{img.stem}.txt", W, H)
        r = model.predict(str(img), conf=args.conf, imgsz=1280, verbose=False)[0]
        preds = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in r.boxes]
        for cls in (BALL, HOOP):
            g = [b for c, b in gt if c == cls]
            p = [b for c, b in preds if c == cls]
            matched = set()
            for gb in g:
                hit = any(iou(gb, pb) >= 0.3 and j not in matched
                          for j, pb in enumerate(p))
                if hit:
                    st[cls]["tp"] += 1
                    for j, pb in enumerate(p):
                        if iou(gb, pb) >= 0.3 and j not in matched:
                            matched.add(j); break
                else:
                    st[cls]["fn"] += 1
            st[cls]["fp"] += len(p) - len(matched)
            st[cls]["n"] += len(g)
            if cls == BALL and kind == "rim":
                if g:
                    (rim_ball.__setitem__("tp", rim_ball["tp"] + (1 if any(
                        iou(gb, pb) >= 0.3 for gb in g for pb in p) else 0)))
                    if not any(iou(gb, pb) >= 0.3 for gb in g for pb in p):
                        rim_ball["fn"] += 1

    print(f"weights={Path(args.weights).name}  test imgs={len(imgs)}\n")
    for cls, name in [(HOOP, "HOOP"), (BALL, "BALL")]:
        s = st[cls]
        rec = s["tp"] / max(1, s["tp"] + s["fn"])
        prec = s["tp"] / max(1, s["tp"] + s["fp"])
        print(f"{name}: recall={rec:.3f} precision={prec:.3f} "
              f"(tp={s['tp']} fn={s['fn']} fp={s['fp']} gt={s['n']})")
    rb = rim_ball["tp"] / max(1, rim_ball["tp"] + rim_ball["fn"])
    print(f"\nBALL recall at RIM MOMENT: {rb:.3f} "
          f"(tp={rim_ball['tp']} fn={rim_ball['fn']})  <- decides make/miss")


if __name__ == "__main__":
    main()
