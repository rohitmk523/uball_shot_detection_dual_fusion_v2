#!/usr/bin/env python3
"""Sweep SERVING-side detector settings (conf, imgsz, TTA) on the held-out test
split to see if ball-at-rim recall can be lifted with NO retraining. Ball recall
at the rim moment is what caps spotter recall.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

NA = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
          "Training_frameworks/Uball Near Angle/data")
W_PATH = ("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
          "uball_shot_detection_dual_fusion_v2/near_v0/weights/near_det_v1_best.pt")
BALL, HOOP = 0, 1


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def load_gt(lbl, W, H):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        c = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:5])
        out.append((c, [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]))
    return out


def main():
    model = YOLO(W_PATH)
    imgs = sorted((NA / "yolo_split/test/images").glob("*.jpg"))
    lbls = NA / "yolo_split/test/labels"
    # preload sizes + gt + rim-kind
    items = []
    for img in imgs:
        H, W = cv2.imread(str(img)).shape[:2]
        items.append((img, W, H, load_gt(lbls / f"{img.stem}.txt", W, H),
                      img.stem.rsplit("_", 1)[-1] == "rim"))

    configs = list(itertools.product([0.15, 0.25], [1280, 1536], [False, True]))
    print(f"{'conf':>5} {'imgsz':>6} {'TTA':>5} | {'ballRec':>7} {'ballPrec':>8} "
          f"{'rimRec':>7} {'hoopRec':>7}")
    for conf, imgsz, tta in configs:
        bt = bf = bfp = 0
        ht = hf = 0
        rt = rf = 0
        for img, W, H, gt, is_rim in items:
            r = model.predict(str(img), conf=conf, imgsz=imgsz, augment=tta,
                              verbose=False)[0]
            preds = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in r.boxes]
            for cls, tcnt in [(BALL, "b"), (HOOP, "h")]:
                g = [b for c, b in gt if c == cls]
                p = [b for c, b in preds if c == cls]
                matched = set()
                hits = 0
                for gb in g:
                    found = False
                    for j, pb in enumerate(p):
                        if j not in matched and iou(gb, pb) >= 0.3:
                            matched.add(j); found = True; break
                    hits += found
                if cls == BALL:
                    bt += hits; bf += len(g)-hits; bfp += len(p)-len(matched)
                    if is_rim and g:
                        rt += 1 if hits else 0
                        rf += 0 if hits else 1
                else:
                    ht += hits; hf += len(g)-hits
        brec = bt/max(1, bt+bf); bprec = bt/max(1, bt+bfp)
        rrec = rt/max(1, rt+rf); hrec = ht/max(1, ht+hf)
        print(f"{conf:>5} {imgsz:>6} {str(tta):>5} | {brec:7.3f} {bprec:8.3f} "
              f"{rrec:7.3f} {hrec:7.3f}")


if __name__ == "__main__":
    main()
