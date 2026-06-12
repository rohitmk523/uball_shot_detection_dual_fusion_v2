#!/usr/bin/env python3
"""Phase 0: run YOLO ball+hoop on phase0 clips; save detections JSON,
annotated videos, and rim-window frame montages for visual inspection.

Detector = existing near-angle YOLO11n weights (a tool, not "near logic").
Low conf threshold on purpose -- Phase 0 observes detector behavior,
including flicker and weak detections.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
P0 = REPO / "data/client_report/near_angle/phase0"
WEIGHTS = ("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
           "Uball_dual_angle_shot_detection/weights/near_angle_weights/"
           "basketball_yolo11n3/weights/best.pt")
CONF = 0.15
IMGSZ = 1280
BALL, HOOP = 0, 1


def detect_clip(model, clip: Path, det_path: Path, ann_path: Path):
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        dets = []
        for b in r.boxes:
            xyxy = [round(float(v), 1) for v in b.xyxy[0]]
            c, cf = int(b.cls[0]), round(float(b.conf[0]), 3)
            dets.append({"cls": c, "conf": cf, "xyxy": xyxy})
            color = (0, 200, 255) if c == BALL else (255, 120, 0)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'ball' if c == BALL else 'hoop'} {cf:.2f}",
                        (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
        cv2.putText(frame, f"f{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        vw.write(frame)
        frames.append(dets)
        idx += 1
    cap.release()
    vw.release()
    det_path.write_text(json.dumps({"clip": clip.name, "fps": fps,
                                    "n_frames": idx, "frames": frames}))
    return frames, fps


def hoop_box(frames):
    """Median hoop box over confident detections (hoop is static)."""
    boxes = [d["xyxy"] for fr in frames for d in fr
             if d["cls"] == HOOP and d["conf"] > 0.4]
    if not boxes:
        return None
    return np.median(np.array(boxes), axis=0)


def montage(clip: Path, frames, out_png: Path, max_rows=4, cols=6):
    """Tile the rim-window frames (ball near hoop) as crops with boxes."""
    hb = hoop_box(frames)
    cap = cv2.VideoCapture(str(clip))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if hb is None:
        cx, cy, hw = W / 2, H / 3, 300.0
    else:
        cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
        hw = hb[2] - hb[0]
    # crop: 2.4 hoop-widths wide, extends 1 hw above and 2.2 below hoop center
    x1 = int(max(0, cx - 1.2 * hw)); x2 = int(min(W, cx + 1.2 * hw))
    y1 = int(max(0, cy - 1.0 * hw)); y2 = int(min(H, cy + 2.2 * hw))

    # frames of interest: ball center within 1.6 hw of hoop center
    interest = []
    for i, fr in enumerate(frames):
        for d in fr:
            if d["cls"] != BALL:
                continue
            bx = (d["xyxy"][0] + d["xyxy"][2]) / 2
            by = (d["xyxy"][1] + d["xyxy"][3]) / 2
            if abs(bx - cx) < 1.6 * hw and abs(by - cy) < 1.6 * hw:
                interest.append(i)
                break
    if interest:
        lo = max(0, min(interest) - 3)
        hi = min(len(frames) - 1, max(interest) + 6)
    else:  # no ball seen near rim -- show middle of clip
        lo, hi = len(frames) // 3, min(len(frames) - 1, len(frames) // 3 + 23)
    n_cells = max_rows * cols
    idxs = list(range(lo, hi + 1))
    if len(idxs) > n_cells:  # uniform subsample
        idxs = [idxs[int(k * (len(idxs) - 1) / (n_cells - 1))]
                for k in range(n_cells)]

    cells = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    want = set(idxs)
    i = 0
    store = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            f = frame.copy()
            for d in frames[i]:
                xx1, yy1, xx2, yy2 = [int(v) for v in d["xyxy"]]
                color = (0, 200, 255) if d["cls"] == BALL else (255, 120, 0)
                cv2.rectangle(f, (xx1, yy1), (xx2, yy2), color, 2)
            crop = f[y1:y2, x1:x2]
            crop = cv2.resize(crop, (300, int(300 * (y2 - y1) / (x2 - x1))))
            cv2.putText(crop, f"f{i}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            store[i] = crop
        i += 1
    cap.release()
    cells = [store[k] for k in idxs if k in store]
    if not cells:
        return False
    ch, cw = cells[0].shape[:2]
    rows = int(np.ceil(len(cells) / cols))
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for k, c in enumerate(cells):
        r, q = divmod(k, cols)
        grid[r * ch:(r + 1) * ch, q * cw:(q + 1) * cw] = c
    cv2.imwrite(str(out_png), grid)
    return True


def main():
    subdir = sys.argv[1] if len(sys.argv) > 1 else "clips"
    clips = sorted((P0 / subdir).glob("*.mp4"))
    clips = [c for c in clips if "_ann" not in c.name]
    (P0 / "detections").mkdir(exist_ok=True)
    (P0 / "annotated").mkdir(exist_ok=True)
    (P0 / "montage").mkdir(exist_ok=True)
    model = YOLO(WEIGHTS)
    for c in clips:
        det_path = P0 / "detections" / (c.stem + ".json")
        ann_path = P0 / "annotated" / (c.stem + "_ann.mp4")
        if det_path.exists():
            frames = json.loads(det_path.read_text())["frames"]
        else:
            frames, _ = detect_clip(model, c, det_path, ann_path)
        ok = montage(c, frames, P0 / "montage" / (c.stem + ".png"))
        n_ball = sum(1 for fr in frames for d in fr if d["cls"] == BALL)
        n_hoop = sum(1 for fr in frames if any(d["cls"] == HOOP for d in fr))
        print(f"{c.stem}: {len(frames)}fr ball_dets={n_ball} "
              f"hoop_frames={n_hoop} montage={'ok' if ok else 'SKIP'}")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
