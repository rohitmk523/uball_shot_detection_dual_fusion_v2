#!/usr/bin/env python3
"""Per-game 3-panel demo (AWS-runnable, self-contained). All shots of a game.
Layout: LEFT = FAR (top) + NEAR (bottom) stacked. RIGHT (full height) = the
basket/rim with green=made / red=missed shot dots + the make/miss verdict.
The dot = the DETECTED ball center at the rim crossing (a real observation,
relative to the detected rim). make/miss = the validated fused classifier
verdict. No fabricated arc/depth/L-R numbers.
Usage: --game G --nr NR.mp4 --fr FR.mp4 --shots shots.json --out out.mp4 ...
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BALL, HOOP = 0, 1
VW, VH = 1216, 540          # far/near panels (left, stacked)
RWp, RHp = 1920 - VW, 1080  # right panel (full height)
ASPECT = VW / VH
BG = (16, 17, 21); GREEN = (76, 217, 100); RED = (255, 90, 70)
ORANGE = (255, 149, 40); WHITE = (244, 245, 248); GREY = (138, 142, 152)


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
    return frame[y0:int(y0+h), x0:int(x0+w)], x0, y0, int(w), int(h)


def draw_boxes(frame, boxes, x0, y0, sx, sy):
    for cls, b in boxes:
        p1 = (int((b[0]-x0)*sx), int((b[1]-y0)*sy)); p2 = (int((b[2]-x0)*sx), int((b[3]-y0)*sy))
        cv2.rectangle(frame, p1, p2, (90, 220, 90) if cls == BALL else (40, 150, 255), 3)


def label_bar(frame, text, make=None):
    im = Image.fromarray(frame[:, :, ::-1]); dr = ImageDraw.Draw(im, "RGBA"); W = frame.shape[1]
    dr.rectangle([0, 0, W, 58], fill=(10, 11, 14, 190)); ct(dr, (26, 29), text, 30, WHITE, "lm")
    if make is not None:
        vc = GREEN if make else RED
        dr.rounded_rectangle([W-200, 12, W-26, 50], radius=12, fill=vc+(255,))
        ct(dr, (W-113, 31), "MAKE" if make else "MISS", 28, (16, 17, 21))
    return np.array(im)[:, :, ::-1].copy()


def panel_right(points, cur, make, idx, n, gt, acc):
    im = Image.new("RGB", (RWp, RHp), BG); dr = ImageDraw.Draw(im)
    ct(dr, (40, 46), "SHOT DETECTION", 32, WHITE, "lm")
    ct(dr, (RWp-40, 46), f"{idx}/{n}", 26, GREY, "rm")
    # verdict pill
    vc = GREEN if make else RED
    dr.rounded_rectangle([RWp//2-190, 96, RWp//2+190, 214], radius=26, fill=vc)
    ct(dr, (RWp//2, 153), "MAKE" if make else "MISS", 72, (16, 17, 21))
    ct(dr, (RWp//2, 252), gt.replace("_", " "), 24, GREY)
    # rim map (basket, top-down)
    cx, cy, R = RWp//2, 660, 270
    dr.ellipse([cx-R-11, cy-R-11, cx+R+11, cy+R+11], outline=ORANGE, width=22)
    dr.ellipse([cx-7, cy-7, cx+7, cy+7], fill=(70, 72, 80))
    ct(dr, (40, 372), "SHOT LOCATIONS", 28, WHITE, "lm")
    ct(dr, (RWp-200, 372), "made", 22, GREEN, "lm"); dr.ellipse([RWp-232, 364, RWp-216, 380], fill=GREEN)
    ct(dr, (RWp-105, 372), "miss", 22, RED, "lm")
    dr.line([(RWp-135, 364), (RWp-121, 380)], fill=RED, width=4); dr.line([(RWp-135, 380), (RWp-121, 364)], fill=RED, width=4)
    for i, (nx, ny, mk) in enumerate(points):
        px, py = int(cx+nx*R*0.92), int(cy+ny*R*0.92); hi = (i == cur)
        if hi:
            dr.ellipse([px-21, py-21, px+21, py+21], outline=WHITE, width=3)
        if mk:
            r = 14 if hi else 10; dr.ellipse([px-r, py-r, px+r, py+r], fill=GREEN)
        else:
            s = 14 if hi else 10
            dr.line([(px-s, py-s), (px+s, py+s)], fill=RED, width=5); dr.line([(px-s, py+s), (px+s, py-s)], fill=RED, width=5)
    ct(dr, (cx, RHp-58), "dot = detected ball position at the rim", 19, GREY)
    ct(dr, (cx, RHp-30), f"make/miss accuracy this game: {acc:.0%}", 20, GREY)
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


def near_cross(near, vid, dev, t, hoop, fps):
    """detected ball center at the rim crossing (closest approach to rim center),
    normalized to the rim box -> position on the rim map."""
    cxr, cyr = (hoop[0]+hoop[2])/2, (hoop[1]+hoop[3])/2
    rimw, rimh = hoop[2]-hoop[0], hoop[3]-hoop[1]
    cap = cv2.VideoCapture(vid); cap.set(1, int((t-0.8)*fps)); best = None
    for _ in range(int(1.8*fps)):
        ok, fr = cap.read()
        if not ok:
            break
        r = near.predict(fr, conf=0.3, imgsz=960, device=dev, verbose=False)[0]
        for x in r.boxes:
            if int(x.cls[0]) != BALL:
                continue
            b = [float(v) for v in x.xyxy[0]]; bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
            dd = (bx-cxr)**2+(by-cyr)**2
            if best is None or dd < best[0]:
                best = (dd, (bx-cxr)/(rimw/2), (by-cyr)/(rimh/2))
    cap.release()
    return (0.0, 0.0) if best is None else (float(np.clip(best[1], -1.1, 1.1)), float(np.clip(best[2], -1.1, 1.1)))


def main():
    ap = argparse.ArgumentParser()
    for k in ["game", "nr", "fr", "nl", "fl", "shots", "out", "near-w", "far-w"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--clip", type=float, default=2.6)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    near = YOLO(getattr(a, "near_w")); far = YOLO(getattr(a, "far_w"))
    data = json.load(open(a.shots)); shots = data["shots"]; acc = data.get("acc", 0)
    nR = sum(s["basket"] == "RIGHT" for s in shots)
    print(f"{a.game}: {len(shots)} shots (R={nR} L={len(shots)-nR}) near-acc {acc}")

    fps = cv2.VideoCapture(a.nr).get(5) or 29.97
    # per-basket camera config: (near_vid, far_vid, near_hoop, far_hoop, n_center, f_center)
    cfg = {}
    for bk, nv, fv in [("RIGHT", a.nr, a.fr), ("LEFT", a.nl, a.fl)]:
        bs = [s for s in shots if s["basket"] == bk]
        if not bs:
            continue
        ts = np.linspace(bs[0]["t0"], bs[-1]["t0"], 8)
        nh = med_hoop(near, nv, dev, 960, ts); fh = med_hoop(far, fv, dev, 1280, ts)
        if nh is None or fh is None:
            print(f"WARN: no hoop for {bk}, skipping that basket"); continue
        cfg[bk] = dict(nv=nv, fv=fv, nh=nh, fh=fh,
                       nc=((nh[0]+nh[2])/2, (nh[1]+nh[3])/2 - 40),
                       fc=((fh[0]+fh[2])/2, (fh[1]+fh[3])/2 - 80),
                       capN=cv2.VideoCapture(nv), capF=cv2.VideoCapture(fv))
        print(f"  {bk}: near {[round(v) for v in nh]} far {[round(v) for v in fh]}")
    shots = [s for s in shots if s["basket"] in cfg]

    pts = []
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1920, 1080))
    for i, s in enumerate(shots):
        t = s["t0"]; mk = bool(s["pred_make"]); c = cfg[s["basket"]]
        nxn, nyn = near_cross(near, c["nv"], dev, t, c["nh"], fps)
        pts.append((nxn, nyn, mk))
        right = panel_right(pts, i, mk, i+1, len(shots), s["gt"], acc)
        bktag = "RIGHT BASKET" if s["basket"] == "RIGHT" else "LEFT BASKET"
        c["capN"].set(1, int((t-a.clip*0.55)*fps)); c["capF"].set(1, int((t-a.clip*0.55)*fps))
        for _ in range(int(a.clip*fps)):
            okn, frn = c["capN"].read(); okf, frf = c["capF"].read()
            if not (okn and okf):
                break
            rF = far.predict(frf, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
            fb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rF.boxes]
            fcrop, fx0, fy0, fw, fh = crop_aspect(frf, c["fc"][0], c["fc"][1], 540)
            fc = cv2.resize(fcrop, (VW, VH)); draw_boxes(fc, fb, fx0, fy0, VW/fw, VH/fh)
            fc = label_bar(fc, f"FAR ANGLE  ·  {bktag}")
            rN = near.predict(frn, conf=0.30, imgsz=960, device=dev, verbose=False)[0]
            nb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rN.boxes]
            ncrop, nx0, ny0, nw, nh = crop_aspect(frn, c["nc"][0], c["nc"][1], 470)
            nc = cv2.resize(ncrop, (VW, VH)); draw_boxes(nc, nb, nx0, ny0, VW/nw, VH/nh)
            nc = label_bar(nc, f"NEAR ANGLE  ·  {bktag}", make=mk)
            vw.write(np.hstack([np.vstack([fc, nc]), right]))
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(shots)} rendered", flush=True)
    for c in cfg.values():
        c["capN"].release(); c["capF"].release()
    vw.release()
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
