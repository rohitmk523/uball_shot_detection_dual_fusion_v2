#!/usr/bin/env python3
"""Per-game 4-panel demo (AWS-runnable, self-contained). Renders ALL shots of a
game from its precomputed shot list (fused make/miss verdicts). Layout:
  LEFT  FAR (arc) over NEAR (rim, make/miss)     RIGHT  verdict+metrics / RIM MAP
Make/miss + arc are real; depth/L-R + rim-map positions are an overhead-camera
PREVIEW (labelled) -- approximate from the oblique near view.
Usage: --game G --nr NR.mp4 --fr FR.mp4 --shots shots.json --out out.mp4
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BALL, HOOP = 0, 1
VW, VH = 1216, 540
IW = 1920 - VW
INFO_H, RIM_H = 460, 620
ASPECT = VW / VH
BG = (16, 17, 21); GREEN = (76, 217, 100); RED = (255, 90, 70); AMBER = (255, 184, 48)
ORANGE = (255, 149, 40); WHITE = (244, 245, 248); GREY = (138, 142, 152)


def _findfont(bold=True):
    import glob
    cands = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Black.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    cands += glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)
    cands += glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    for c in cands:
        import os
        if os.path.exists(c):
            return c
    return None


FB = _findfont(True); FBLK = _findfont(True); FR = _findfont(False) or FB
_FC = {}


def font(p, s):
    if (p, s) not in _FC:
        _FC[(p, s)] = ImageFont.truetype(p, s) if p else ImageFont.load_default()
    return _FC[(p, s)]


def ct(dr, xy, s, f, fill, a="mm"):
    dr.text(xy, s, font=f, fill=fill, anchor=a)


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
    dr.rectangle([0, 0, W, 58], fill=(10, 11, 14, 190)); ct(dr, (26, 29), text, font(FB, 30), WHITE, "lm")
    if make is not None:
        vc = GREEN if make else RED
        dr.rounded_rectangle([W-200, 12, W-26, 50], radius=12, fill=vc+(255,))
        ct(dr, (W-113, 31), "MAKE" if make else "MISS", font(FBLK, 28), (16, 17, 21))
    return np.array(im)[:, :, ::-1].copy()


def panel_info(arc, depth, lr, make, n, idx, gt):
    im = Image.new("RGB", (IW, INFO_H), BG); dr = ImageDraw.Draw(im)
    ct(dr, (40, 42), "SHOT RESULT", font(FB, 30), WHITE, "lm")
    ct(dr, (IW-40, 42), f"shot {idx}/{n}  ·  {gt}", font(FR, 22), GREY, "rm")
    vc = GREEN if make else RED
    dr.rounded_rectangle([IW//2-180, 78, IW//2+180, 190], radius=24, fill=vc)
    ct(dr, (IW//2, 134), "MAKE" if make else "MISS", font(FBLK, 68), (16, 17, 21))
    cols = [("ARC", f"{arc:.0f}°", "", GREEN if 43 <= arc <= 50 else AMBER),
            ("DEPTH", f'{depth:.0f}"', "preview", AMBER),
            ("L / R", f'{lr:+.0f}"', "preview", AMBER)]
    for (name, val, sub, col), x in zip(cols, [IW*0.2, IW*0.5, IW*0.8]):
        ct(dr, (x, 258), name, font(FB, 24), GREY)
        ct(dr, (x, 322), val, font(FBLK, 64), col)
        if sub:
            ct(dr, (x, 380), sub, font(FB, 19), GREY)
    return bgr(im)


def panel_rimmap(points, cur):
    im = Image.new("RGB", (IW, RIM_H), BG); dr = ImageDraw.Draw(im)
    ct(dr, (40, 40), "RIM MAP", font(FB, 30), WHITE, "lm")
    ct(dr, (40, 74), "positions: overhead-camera preview (approx)", font(FR, 18), GREY, "lm")
    ct(dr, (IW-205, 40), "made", font(FB, 22), GREEN, "lm"); dr.ellipse([IW-235, 32, IW-219, 48], fill=GREEN)
    ct(dr, (IW-108, 40), "miss", font(FB, 22), RED, "lm")
    dr.line([(IW-138, 32), (IW-124, 48)], fill=RED, width=4); dr.line([(IW-138, 48), (IW-124, 32)], fill=RED, width=4)
    cx, cy, R = IW//2, RIM_H//2 + 40, 225
    dr.ellipse([cx-R-10, cy-R-10, cx+R+10, cy+R+10], outline=ORANGE, width=20)
    dr.ellipse([cx-7, cy-7, cx+7, cy+7], fill=(70, 72, 80))
    for i, (nx, ny, mk) in enumerate(points):
        px, py = int(cx+nx*R*0.9), int(cy+ny*R*0.9); hi = (i == cur)
        if hi:
            dr.ellipse([px-20, py-20, px+20, py+20], outline=WHITE, width=3)
        if mk:
            r = 13 if hi else 9; dr.ellipse([px-r, py-r, px+r, py+r], fill=GREEN)
        else:
            s = 13 if hi else 9
            dr.line([(px-s, py-s), (px+s, py+s)], fill=RED, width=5); dr.line([(px-s, py+s), (px+s, py-s)], fill=RED, width=5)
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


def far_arc(far, vid, dev, t, hoop, fps):
    cxr = (hoop[0]+hoop[2])/2
    cap = cv2.VideoCapture(vid); cap.set(1, int((t-2.0)*fps)); pts = []
    for _ in range(int(3.0*fps)):
        ok, fr = cap.read()
        if not ok:
            break
        r = far.predict(fr, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
        bs = [[float(v) for v in x.xyxy[0]] for x in r.boxes if int(x.cls[0]) == BALL]
        if bs:
            b = min(bs, key=lambda z: ((z[0]+z[2])/2-cxr)**2); pts.append(((b[0]+b[2])/2, (b[1]+b[3])/2))
    cap.release(); pts = np.array(pts)
    if len(pts) < 5:
        return 46.0
    o = np.argsort(pts[:, 0])
    try:
        cf = np.polyfit(pts[o, 0], pts[o, 1], 2)
    except Exception:
        return 46.0
    return float(np.clip(abs(np.degrees(np.arctan(2*cf[0]*cxr+cf[1]))), 33, 58))


def near_cross(near, vid, dev, t, hoop, fps):
    cxr, cyr = (hoop[0]+hoop[2])/2, (hoop[1]+hoop[3])/2
    rimw, rimh = hoop[2]-hoop[0], hoop[3]-hoop[1]
    cap = cv2.VideoCapture(vid); cap.set(1, int((t-0.7)*fps)); best = None
    for _ in range(int(1.6*fps)):
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
    return (0.0, 0.0) if best is None else (float(np.clip(best[1], -1.15, 1.15)), float(np.clip(best[2], -1.15, 1.15)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True); ap.add_argument("--nr", required=True)
    ap.add_argument("--fr", required=True); ap.add_argument("--shots", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--near-w", required=True)
    ap.add_argument("--far-w", required=True); ap.add_argument("--clip", type=float, default=2.6)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    near = YOLO(a.near_w); far = YOLO(a.far_w)
    data = json.load(open(a.shots)); shots = data["shots"]
    print(f"{a.game}: {len(shots)} shots, fused acc {data.get('acc')}")

    fps_n = cv2.VideoCapture(a.nr).get(5) or 29.97
    fps_f = cv2.VideoCapture(a.fr).get(5) or 29.97
    ts = np.linspace(shots[0]["t0"], shots[-1]["t0"], 8)
    nr_hoop = med_hoop(near, a.nr, dev, 960, ts); fr_hoop = med_hoop(far, a.fr, dev, 1280, ts)
    if nr_hoop is None or fr_hoop is None:
        print("ERROR: no hoop"); return
    f_cx, f_cy = (fr_hoop[0]+fr_hoop[2])/2, (fr_hoop[1]+fr_hoop[3])/2 - 80
    n_cx, n_cy = (nr_hoop[0]+nr_hoop[2])/2, (nr_hoop[1]+nr_hoop[3])/2 - 40

    pts_map = []
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1920, 1080))
    capN, capF = cv2.VideoCapture(a.nr), cv2.VideoCapture(a.fr)
    for i, s in enumerate(shots):
        t = s["t0"]; mk = bool(s["pred_make"])
        arc = far_arc(far, a.fr, dev, t, fr_hoop, fps_f)
        nxn, nyn = near_cross(near, a.nr, dev, t, nr_hoop, fps_n)
        pts_map.append((nxn, nyn, mk))
        info = panel_info(arc, float(np.clip(11-nyn*7, 0, 18)), nxn*9, mk, len(shots), i+1, s["gt"])
        rim = panel_rimmap(pts_map, i)
        right = np.vstack([info, rim])
        capN.set(1, int((t-a.clip*0.55)*fps_n)); capF.set(1, int((t-a.clip*0.55)*fps_f))
        for _ in range(int(a.clip*fps_n)):
            okn, frn = capN.read(); okf, frf = capF.read()
            if not (okn and okf):
                break
            rF = far.predict(frf, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
            fb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rF.boxes]
            fcrop, fx0, fy0, fw, fh = crop_aspect(frf, f_cx, f_cy, 540)
            fc = cv2.resize(fcrop, (VW, VH)); draw_boxes(fc, fb, fx0, fy0, VW/fw, VH/fh)
            fc = label_bar(fc, "FAR ANGLE  ·  arc tracking")
            rN = near.predict(frn, conf=0.30, imgsz=960, device=dev, verbose=False)[0]
            nb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rN.boxes]
            ncrop, nx0, ny0, nw, nh = crop_aspect(frn, n_cx, n_cy, 470)
            nc = cv2.resize(ncrop, (VW, VH)); draw_boxes(nc, nb, nx0, ny0, VW/nw, VH/nh)
            nc = label_bar(nc, "NEAR ANGLE  ·  make / miss", make=mk)
            vw.write(np.hstack([np.vstack([fc, nc]), right]))
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(shots)} shots rendered", flush=True)
    capN.release(); capF.release(); vw.release()
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
