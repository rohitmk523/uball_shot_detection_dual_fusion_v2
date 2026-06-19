#!/usr/bin/env python3
"""Render a highlight reel of the FUSION ERROR shots (GT != prediction) for one
game: FAR (top-left) + NEAR (bottom-left) at the shot, and a right panel showing
GROUND TRUTH vs SYSTEM prediction + the far/near probabilities, so errors can be
eyeballed (is it a genuine depth illusion?). One game per call; concat across
games for the full reel.

  python near_v0/error_reel.py --game G --fr FR.mp4 --fl FL.mp4 --nr NR.mp4 \
      --nl NL.mp4 --manifest errors_G.json --out G_err.mp4 \
      --far-w far_v16_best.pt --near-w near_det_v1_best.pt --start 0 --total 34
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BALL, HOOP = 0, 1
VW, VH = 1216, 540
RWp, RHp = 1920 - VW, 1080
ASPECT = VW / VH
BG = (16, 17, 21); GREEN = (76, 217, 100); RED = (255, 90, 70)
WHITE = (244, 245, 248); GREY = (138, 142, 152); AMBER = (255, 200, 60)


def _findfont():
    import os, glob
    for c in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] +
              glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)):
        if os.path.exists(c):
            return c
    return None


FB = _findfont(); _FC = {}


def font(s):
    if s not in _FC:
        _FC[s] = ImageFont.truetype(FB, s) if FB else ImageFont.load_default()
    return _FC[s]


def ct(dr, xy, s, sz, fill, a="mm"):
    dr.text(xy, s, font=font(sz), fill=fill, anchor=a)


def bgr(im):
    return np.array(im.convert("RGB"))[:, :, ::-1].copy()


def crop_aspect(frame, cx, cy, h, W=1920, H=1080):
    w = h * ASPECT
    x0 = int(np.clip(cx-w/2, 0, W-w)); y0 = int(np.clip(cy-h/2, 0, H-h))
    return frame[y0:int(y0+h), x0:int(x0+w)]


def label_bar(frame, text):
    im = Image.fromarray(frame[:, :, ::-1]); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, frame.shape[1], 54], fill=(10, 11, 14, 200))
    ct(dr, (24, 27), text, 28, WHITE, "lm")
    return np.array(im)[:, :, ::-1].copy()


def panel_info(s, idx, n):
    im = Image.new("RGB", (RWp, RHp), BG); dr = ImageDraw.Draw(im)
    ct(dr, (RWp-40, 46), f"error {idx}/{n}", 26, GREY, "rm")
    ct(dr, (RWp//2, 60), "SHOT-DETECTION ERROR", 30, AMBER)
    ct(dr, (RWp//2, 110), s["gt"].replace("_", " "), 22, GREY)
    # GT vs SYSTEM
    gt_make = s["gt_make"]; pred_make = s["pred_make"]
    ct(dr, (RWp//2, 210), "GROUND TRUTH", 26, GREY)
    gc = GREEN if gt_make else RED
    dr.rounded_rectangle([RWp//2-180, 248, RWp//2+180, 350], radius=22, fill=gc)
    ct(dr, (RWp//2, 299), "MAKE" if gt_make else "MISS", 62, (16, 17, 21))
    ct(dr, (RWp//2, 430), "SYSTEM PREDICTED", 26, GREY)
    pc = GREEN if pred_make else RED
    dr.rounded_rectangle([RWp//2-180, 468, RWp//2+180, 570], radius=22, fill=pc)
    ct(dr, (RWp//2, 519), "MAKE" if pred_make else "MISS", 62, (16, 17, 21))
    ct(dr, (RWp//2, 640), "WRONG", 40, RED)
    ct(dr, (RWp//2, 690), s["err"], 22, GREY)
    # per-camera votes
    ct(dr, (RWp//2, 780), "camera votes (prob of MAKE)", 20, GREY)
    fc = GREEN if s["prob_far"] >= 0.5 else RED
    nc = GREEN if s["prob_near"] >= 0.5 else RED
    ct(dr, (RWp//2-120, 830), f"FAR {s['prob_far']:.2f}", 26, fc)
    ct(dr, (RWp//2+120, 830), f"NEAR {s['prob_near']:.2f}", 26, nc)
    ct(dr, (RWp//2, RHp-40), s["pid8"], 18, GREY)
    return bgr(im)


def med_hoop(model, vid, dev, imgsz, ts):
    cap = cv2.VideoCapture(vid); fps = cap.get(5); hs = []
    for t in ts:
        cap.set(1, int(t*fps)); ok, fr = cap.read()
        if not ok:
            continue
        r = model.predict(fr, conf=0.3, imgsz=imgsz, device=dev, verbose=False)[0]
        b = [[float(v) for v in x.xyxy[0]] for x in r.boxes if int(x.cls[0]) == HOOP]
        if b:
            hs.append(max(b, key=lambda z: (z[2]-z[0])*(z[3]-z[1])))
    cap.release()
    return np.median(np.array(hs), 0) if hs else None


def main():
    ap = argparse.ArgumentParser()
    for k in ["game", "fr", "fl", "nr", "nl", "manifest", "out", "far-w", "near-w"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--total", type=int, default=0)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    far = YOLO(getattr(a, "far_w")); near = YOLO(getattr(a, "near_w"))
    shots = json.loads(Path(a.manifest).read_text())
    total = a.total or len(shots)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1920, 1080))
    for j, s in enumerate(shots):
        right_v = a.fr if s["basket"] == "RIGHT" else a.fl
        near_v = a.nr if s["basket"] == "RIGHT" else a.nl
        capF = cv2.VideoCapture(right_v); capN = cv2.VideoCapture(near_v)
        fps = capF.get(5) or 29.97
        ts = np.linspace(max(0, s["t0"]-1), s["t1"], 5)
        fh = med_hoop(far, right_v, dev, 1280, ts); nh = med_hoop(near, near_v, dev, 960, ts)
        f_cx, f_cy = ((fh[0]+fh[2])/2, (fh[1]+fh[3])/2-80) if fh is not None else (960, 400)
        n_cx, n_cy = ((nh[0]+nh[2])/2, (nh[1]+nh[3])/2-40) if nh is not None else (960, 540)
        info = panel_info(s, a.start+j+1, total)
        cs = max(s["t0"]-0.3, s["t1"]-2.8); ce = s["t1"]+0.6
        capF.set(1, int(cs*fps)); capN.set(1, int(cs*fps))
        for _ in range(int((ce-cs)*fps)):
            okf, frf = capF.read(); okn, frn = capN.read()
            if not (okf and okn):
                break
            fc = cv2.resize(crop_aspect(frf, f_cx, f_cy, 540), (VW, VH))
            fc = label_bar(fc, f"FAR ANGLE  ·  {s['basket']} BASKET")
            nc = cv2.resize(crop_aspect(frn, n_cx, n_cy, 470), (VW, VH))
            nc = label_bar(nc, f"NEAR ANGLE  ·  {s['basket']} BASKET")
            vw.write(np.hstack([np.vstack([fc, nc]), info]))
        capF.release(); capN.release()
        print(f"  {a.game} err {j+1}/{len(shots)} done", flush=True)
    vw.release()
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
