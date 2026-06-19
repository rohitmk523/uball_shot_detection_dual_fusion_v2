#!/usr/bin/env python3
"""Extract per-frame ball+rim box TRACKS for one game with BOTH YOLO and
RF-DETR detectors, so the fusion comparison is apples-to-apples (same shots,
same frames, only the detector differs). Output schema matches what
pipeline/geometry_features.py + pipeline/p2_dataset consume:
  game_id, play_id, angle, frame_idx, ball_x, ball_y, ball_w, ball_h,
                                       rim_x, rim_y, rim_w, rim_h

Per shot we extract the RELEVANT far + near angle (RIGHT->FR,NR ; LEFT->FL,NL);
the other two angles are absent (geometry treats them as sentinel, which is the
detection-weighted behaviour). Writes tracks_yolo_<g>.parquet and
tracks_rfdetr_<g>.parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BALL, HOOP = 0, 1
FAR_CONF, NEAR_CONF = 0.10, 0.30      # far low conf (per_camera_verdict), near 0.30
PAD = 0.5


def yolo_pred(model, dev, imgsz, conf):
    def f(frame):
        r = model.predict(frame, conf=conf, iou=0.5, imgsz=imgsz, device=dev, verbose=False)[0]
        return [(int(b.cls[0]), [float(v) for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes]
    return f


def rfdetr_pred(model, conf):
    def f(frame):
        d = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=conf)
        return [(int(d.class_id[i]), [float(v) for v in d.xyxy[i]], float(d.confidence[i]))
                for i in range(len(d.xyxy))]
    return f


def best_boxes(boxes):
    """Return (ball_xyxy|None, ball_conf, rim_xyxy|None, rim_conf): biggest rim,
    ball nearest that rim (else highest-conf ball)."""
    rims = [(b, cf) for c, b, cf in boxes if c == HOOP]
    balls = [(b, cf) for c, b, cf in boxes if c == BALL]
    rim, rconf = max(rims, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1])) if rims else (None, np.nan)
    ball, bconf = (None, np.nan)
    if balls:
        if rim is not None:
            rcx, rcy = (rim[0]+rim[2])/2, (rim[1]+rim[3])/2
            ball, bconf = min(balls, key=lambda x: ((x[0][0]+x[0][2])/2-rcx)**2 + ((x[0][1]+x[0][3])/2-rcy)**2)
        else:
            ball, bconf = max(balls, key=lambda x: x[1])
    return ball, bconf, rim, rconf


def xywh(b):
    return (b[0], b[1], b[2]-b[0], b[3]-b[1]) if b else (np.nan,)*4


def extract_angle(video, shots, angle, gid, predictors):
    """shots: list of (pid8,t0,t1,cls). predictors: dict name->predict_fn.
    Returns dict name-> list of row dicts."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    rows = {k: [] for k in predictors}
    for pid, t0, t1, cls in shots:
        f0 = int(max(0, (t0-PAD)*fps)); f1 = int((t1+PAD)*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for fi in range(f0, f1):
            ok, fr = cap.read()
            if not ok:
                break
            for name, pred in predictors.items():
                ball, bconf, rim, rconf = best_boxes(pred(fr))
                bx, by, bw, bh = xywh(ball); rx, ry, rw, rh = xywh(rim)
                rows[name].append(dict(game_id=gid, play_id=pid, classification=cls,
                                       angle=angle, frame_idx=fi,
                                       ball_x=bx, ball_y=by, ball_w=bw, ball_h=bh, ball_conf=bconf,
                                       rim_x=rx, rim_y=ry, rim_w=rw, rim_h=rh, rim_conf=rconf))
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--shots", required=True)         # demo_data2/<g>.json
    ap.add_argument("--fr", required=True); ap.add_argument("--fl", required=True)
    ap.add_argument("--nr", required=True); ap.add_argument("--nl", required=True)
    ap.add_argument("--yolo-far", required=True); ap.add_argument("--yolo-near", required=True)
    ap.add_argument("--rfdetr-far", required=True); ap.add_argument("--rfdetr-near", required=True)
    ap.add_argument("--outdir", default="/work")
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    from ultralytics import YOLO
    from rfdetr import RFDETRSmall
    yf = yolo_pred(YOLO(a.yolo_far), dev, 960, FAR_CONF)
    yn = yolo_pred(YOLO(a.yolo_near), dev, 960, NEAR_CONF)
    rf_far = RFDETRSmall(pretrain_weights=a.rfdetr_far, resolution=1280)
    rf_near = RFDETRSmall(pretrain_weights=a.rfdetr_near, resolution=1280)
    for m in (rf_far, rf_near):
        try:
            m.optimize_for_inference()
        except Exception as e:
            print(f"[rfdetr] optimize skipped: {e}", flush=True)
    rff = rfdetr_pred(rf_far, FAR_CONF); rfn = rfdetr_pred(rf_near, NEAR_CONF)

    gid = a.game
    shots = json.loads(Path(a.shots).read_text())["shots"]
    R = [(s["pid8"], s["t0"], s["t1"], s["gt"]) for s in shots if s["basket"] == "RIGHT"]
    L = [(s["pid8"], s["t0"], s["t1"], s["gt"]) for s in shots if s["basket"] == "LEFT"]
    print(f"{gid}: {len(R)} RIGHT, {len(L)} LEFT shots", flush=True)

    yolo_rows, rf_rows = [], []
    jobs = [(a.fr, R, "FR", yf, rff), (a.nr, R, "NR", yn, rfn),
            (a.fl, L, "FL", yf, rff), (a.nl, L, "NL", yn, rfn)]
    for video, shots_a, ang, y_pred, r_pred in jobs:
        if not shots_a:
            continue
        print(f"  {ang}: {len(shots_a)} shots", flush=True)
        out = extract_angle(video, shots_a, ang, gid, {"yolo": y_pred, "rfdetr": r_pred})
        yolo_rows += out["yolo"]; rf_rows += out["rfdetr"]

    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(yolo_rows).to_parquet(f"{a.outdir}/tracks_yolo_{gid}.parquet")
    pd.DataFrame(rf_rows).to_parquet(f"{a.outdir}/tracks_rfdetr_{gid}.parquet")
    print(f"wrote tracks_yolo_{gid}.parquet ({len(yolo_rows)} rows) + "
          f"tracks_rfdetr_{gid}.parquet ({len(rf_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
