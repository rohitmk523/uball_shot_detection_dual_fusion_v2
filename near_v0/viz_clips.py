#!/usr/bin/env python3
"""Render annotated review clips: rim-crop window with detector ball+hoop boxes
and the make/miss verdict (p_make, predicted, ground-truth) burned in. Picks
correct AND wrong examples from the end-to-end results so the behaviour is
visible. Outputs an mp4 + a montage PNG per clip.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_end_to_end import crop_rect, CONF, IMGSZ, BALL, HOOP

DET_W = REPO / "near_v0/weights/near_det_v1_best.pt"
OUT = REPO / "data/near_detector/review_clips"
GAME = "e74164e6"
VIDEO = f"{REPO}/data/client_report/triangulation_test/june_{GAME}/{GAME}_NR_full.mp4"


def g(e):
    x = e["gt"]
    return ast.literal_eval(x) if isinstance(x, str) else x


def det_rim(det, cap, fps, dev, n=25):
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes = []
    for f in np.linspace(N * 0.2, N * 0.8, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        r = det.predict(fr, conf=0.3, imgsz=IMGSZ, device=dev, verbose=False)[0]
        hs = [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls[0]) == HOOP]
        if hs:
            boxes.append(max(hs, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
    return np.median(np.array(boxes), 0)   # [x1,y1,x2,y2] (matches crop_rect)


def render(det, cap, fps, dev, e, rim, tag, idx):
    gg = g(e)
    x1, y1, side = crop_rect(rim)
    pred = "MAKE" if e["pred_make"] else "MISS"
    truth = "MAKE" if gg["make"] else "MISS"
    correct = (e["pred_make"] == gg["make"])
    t = e["t"]
    lo = int((t - 1.3) * fps)
    hi = int((t + 1.3) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    name = f"{idx:02d}_{tag}_{gg['gt']}_p{e['p_make']:.2f}"
    vw = cv2.VideoWriter(str(OUT / f"{name}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 12, (side, side))
    frames_drawn = []
    fi = lo
    while fi <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        r = det.predict(frame, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)[0]
        crop = np.ascontiguousarray(frame[y1:y1+side, x1:x1+side])
        if crop.shape[0] != side or crop.shape[1] != side or crop.size == 0:
            fi += 1
            continue
        for b in r.boxes:
            cls = int(b.cls[0])
            bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
            # to crop coords
            cx1, cy1 = int(bx1-x1), int(by1-y1)
            cx2, cy2 = int(bx2-x1), int(by2-y1)
            col = (0, 255, 0) if cls == BALL else (255, 160, 0)
            cv2.rectangle(crop, (cx1, cy1), (cx2, cy2), col, 2)
            if cls == BALL:
                cv2.putText(crop, f"{float(b.conf[0]):.2f}", (cx1, cy1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        border = (0, 200, 0) if correct else (0, 0, 255)
        cv2.rectangle(crop, (0, 0), (side-1, side-1), border, 6)
        cv2.putText(crop, f"f{fi}  pred={pred} p={e['p_make']:.2f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(crop, f"GT={truth}  {'OK' if correct else 'WRONG'}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, border, 2)
        vw.write(crop)
        frames_drawn.append(crop)
        fi += 1
    vw.release()
    # montage: 8 evenly-spaced frames
    if frames_drawn:
        sel = np.linspace(0, len(frames_drawn)-1, 8).astype(int)
        small = [cv2.resize(frames_drawn[i], (220, 220)) for i in sel]
        row1 = np.hstack(small[:4]); row2 = np.hstack(small[4:])
        cv2.imwrite(str(OUT / f"{name}.png"), np.vstack([row1, row2]))
    return name, correct


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(REPO / f"data/near_detector/e2e_{GAME}.json"))
    matched = [e for e in d["events"] if e.get("gt") and e["gt"] not in (None, "None")]
    # dedup per shot
    by = {}
    for e in matched:
        pid = g(e)["pid"]
        if pid not in by or abs(e["t"]-g(e)["t1"]) < abs(by[pid]["t"]-g(by[pid])["t1"]):
            by[pid] = e
    matched = list(by.values())
    cmake = [e for e in matched if g(e)["make"] and e["pred_make"]]
    cmiss = [e for e in matched if not g(e)["make"] and not e["pred_make"]]
    wrong = [e for e in matched if e["pred_make"] != g(e)["make"]]
    picks = ([("correct_make", e) for e in cmake[:2]] +
             [("correct_miss", e) for e in cmiss[:2]] +
             [("WRONG", e) for e in wrong[:3]])
    print(f"picks: {len(cmake)} correct-make, {len(cmiss)} correct-miss, {len(wrong)} wrong")

    det = YOLO(str(DET_W))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    rim = det_rim(det, cap, fps, dev)
    out = []
    for i, (tag, e) in enumerate(picks):
        name, correct = render(det, cap, fps, dev, e, rim, tag, i)
        out.append(name)
        print(f"rendered {name} ({'OK' if correct else 'WRONG'})")
    cap.release()
    (OUT / "index.json").write_text(json.dumps(out, indent=1))
    print(f"\n{len(out)} clips -> {OUT}")


if __name__ == "__main__":
    main()
