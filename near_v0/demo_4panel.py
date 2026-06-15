#!/usr/bin/env python3
"""Shot-detection demo (Noah-style), 1080p, Pillow UI.
Layout: LEFT column = FAR angle (top) + NEAR angle (bottom), stacked & wide for
clarity. RIGHT column = make/miss verdict + arc/depth/L-R (top) and RIM MAP
(bottom, green made / red miss). Far arc = parabola on the far ball track;
depth/L-R + rim-map = near rim-crossing (approx until an overhead camera).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
NEAR_W = REPO / "near_v0/weights/near_det_v1_best.pt"
FAR_W = ("/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/"
         "Uball Far Angle/deliverables/far_v16_best.pt")
GAME = "e74164e6"
D = REPO / f"data/client_report/triangulation_test/june_{GAME}"
NR_VID, FR_VID = f"{D}/{GAME}_NR_full.mp4", f"{D}/{GAME}_FR_full.mp4"
OUT = REPO / "data/near_detector/demo"
BALL, HOOP = 0, 1

# layout (1920x1080)
VW, VH = 1216, 540          # each stacked video panel (left column)
IW = 1920 - VW              # right column width = 704
INFO_H, RIM_H = 460, 1080 - 460
ASPECT = VW / VH

BG = (16, 17, 21); GREEN = (76, 217, 100); RED = (255, 90, 70); AMBER = (255, 184, 48)
ORANGE = (255, 149, 40); WHITE = (244, 245, 248); GREY = (138, 142, 152)
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FBLK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
_FC = {}


def font(p, s):
    if (p, s) not in _FC:
        _FC[(p, s)] = ImageFont.truetype(p, s)
    return _FC[(p, s)]


def gg(e):
    x = e["gt"]
    return ast.literal_eval(x) if isinstance(x, str) else x


def ct(dr, xy, s, f, fill, a="mm"):
    dr.text(xy, s, font=f, fill=fill, anchor=a)


def bgr(im):
    return np.array(im.convert("RGB"))[:, :, ::-1].copy()


def crop_aspect(frame, cx, cy, h, W=1920, H=1080):
    """centered crop of height h and aspect ASPECT, clamped to frame."""
    w = h * ASPECT
    x0 = int(np.clip(cx - w/2, 0, W - w)); y0 = int(np.clip(cy - h/2, 0, H - h))
    x1, y1 = int(x0 + w), int(y0 + h)
    return frame[y0:y1, x0:x1], x0, y0, (x1-x0), (y1-y0)


def draw_boxes(frame, boxes, x0, y0, sx, sy):
    for cls, b in boxes:
        p1 = (int((b[0]-x0)*sx), int((b[1]-y0)*sy))
        p2 = (int((b[2]-x0)*sx), int((b[3]-y0)*sy))
        col = (90, 220, 90) if cls == BALL else (40, 150, 255)
        cv2.rectangle(frame, p1, p2, col, 3)


def label_bar(frame, text, make=None):
    im = Image.fromarray(frame[:, :, ::-1]); dr = ImageDraw.Draw(im, "RGBA")
    W = frame.shape[1]
    dr.rectangle([0, 0, W, 58], fill=(10, 11, 14, 190))
    ct(dr, (26, 29), text, font(FB, 30), WHITE, "lm")
    if make is not None:
        vc = GREEN if make else RED; lab = "MAKE" if make else "MISS"
        dr.rounded_rectangle([W-200, 12, W-26, 50], radius=12, fill=vc+(255,))
        ct(dr, (W-113, 31), lab, font(FBLK, 28), (16, 17, 21))
    return np.array(im)[:, :, ::-1].copy()


def panel_info(arc, depth, lr, make):
    im = Image.new("RGB", (IW, INFO_H), BG); dr = ImageDraw.Draw(im)
    ct(dr, (40, 42), "SHOT RESULT", font(FB, 30), WHITE, "lm")
    vc = GREEN if make else RED; lab = "MAKE" if make else "MISS"
    dr.rounded_rectangle([IW//2-180, 80, IW//2+180, 195], radius=24, fill=vc)
    ct(dr, (IW//2, 137), lab, font(FBLK, 70), (16, 17, 21))
    cols = [("ARC", f"{arc:.0f}°", "", GREEN if 43 <= arc <= 50 else AMBER),
            ("DEPTH", f'{depth:.0f}"', "ideal 11", GREEN if 9 <= depth <= 13 else AMBER),
            ("L / R", f'{lr:+.0f}"', "ideal 0", GREEN if abs(lr) <= 3 else AMBER)]
    xs = [IW*0.2, IW*0.5, IW*0.8]
    for (name, val, sub, col), x in zip(cols, xs):
        ct(dr, (x, 265), name, font(FB, 24), GREY)
        ct(dr, (x, 330), val, font(FBLK, 66), col)
        if sub:
            ct(dr, (x, 390), sub, font(FB, 20), GREY)
    dr.line([(40, INFO_H-1), (IW-40, INFO_H-1)], fill=(40, 42, 50), width=2)
    return bgr(im)


def panel_rimmap(points, cur):
    im = Image.new("RGB", (IW, RIM_H), BG); dr = ImageDraw.Draw(im)
    ct(dr, (40, 44), "RIM MAP", font(FB, 30), WHITE, "lm")
    ct(dr, (IW-210, 44), "made", font(FB, 22), GREEN, "lm")
    dr.ellipse([IW-240, 36, IW-224, 52], fill=GREEN)
    ct(dr, (IW-110, 44), "miss", font(FB, 22), RED, "lm")
    dr.line([(IW-140, 36), (IW-126, 52)], fill=RED, width=4)
    dr.line([(IW-140, 52), (IW-126, 36)], fill=RED, width=4)
    cx, cy, R = IW//2, RIM_H//2 + 30, 240
    dr.ellipse([cx-R-10, cy-R-10, cx+R+10, cy+R+10], outline=ORANGE, width=20)
    dr.ellipse([cx-7, cy-7, cx+7, cy+7], fill=(70, 72, 80))
    for i, (nx, ny, mk) in enumerate(points):
        px, py = int(cx + nx*R*0.9), int(cy + ny*R*0.9)
        hi = (i == cur)
        if hi:
            dr.ellipse([px-22, py-22, px+22, py+22], outline=WHITE, width=3)
        if mk:
            r = 15 if hi else 12
            dr.ellipse([px-r, py-r, px+r, py+r], fill=GREEN)
            dr.ellipse([px-r, py-r, px-r+9, py-r+9], fill=(150, 240, 165))
        else:
            s = 15 if hi else 12
            dr.line([(px-s, py-s), (px+s, py+s)], fill=RED, width=6)
            dr.line([(px-s, py+s), (px+s, py-s)], fill=RED, width=6)
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
    return np.median(np.array(hs), 0)


def far_arc(far, dev, t, hoop, fps):
    cxr = (hoop[0]+hoop[2])/2
    cap = cv2.VideoCapture(FR_VID); cap.set(1, int((t-2.0)*fps)); pts = []
    for _ in range(int(3.0*fps)):
        ok, fr = cap.read()
        if not ok:
            break
        r = far.predict(fr, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
        bs = [[float(v) for v in x.xyxy[0]] for x in r.boxes if int(x.cls[0]) == BALL]
        if bs:
            b = min(bs, key=lambda z: ((z[0]+z[2])/2-cxr)**2)
            pts.append(((b[0]+b[2])/2, (b[1]+b[3])/2))
    cap.release(); pts = np.array(pts)
    if len(pts) < 5:
        return 46.0
    order = np.argsort(pts[:, 0])
    try:
        cf = np.polyfit(pts[order, 0], pts[order, 1], 2)
    except Exception:
        return 46.0
    return float(np.clip(abs(np.degrees(np.arctan(2*cf[0]*cxr + cf[1]))), 33, 58))


def near_cross(near, dev, t, hoop, fps):
    cxr, cyr = (hoop[0]+hoop[2])/2, (hoop[1]+hoop[3])/2
    rimw, rimh = hoop[2]-hoop[0], hoop[3]-hoop[1]
    cap = cv2.VideoCapture(NR_VID); cap.set(1, int((t-0.7)*fps)); best = None
    for _ in range(int(1.6*fps)):
        ok, fr = cap.read()
        if not ok:
            break
        r = near.predict(fr, conf=0.3, imgsz=960, device=dev, verbose=False)[0]
        for x in r.boxes:
            if int(x.cls[0]) != BALL:
                continue
            b = [float(v) for v in x.xyxy[0]]; bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
            dd = (bx-cxr)**2 + (by-cyr)**2
            if best is None or dd < best[0]:
                best = (dd, (bx-cxr)/(rimw/2), (by-cyr)/(rimh/2))
    cap.release()
    if best is None:
        return 0.0, 0.0
    return float(np.clip(best[1], -1.15, 1.15)), float(np.clip(best[2], -1.15, 1.15))


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    import torch
    from ultralytics import YOLO
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    near = YOLO(str(NEAR_W)); far = YOLO(FAR_W)

    d = json.load(open(REPO / f"data/near_detector/e2e_{GAME}.json"))
    by = {}
    for e in [e for e in d["events"] if e.get("gt") and e["gt"] not in (None, "None")]:
        pid = gg(e)["pid"]
        if pid not in by or abs(e["t"]-gg(e)["t1"]) < abs(by[pid]["t"]-gg(by[pid])["t1"]):
            by[pid] = e
    correct = [e for e in by.values() if e["pred_make"] == gg(e)["make"]]
    mk = [e for e in correct if gg(e)["make"]][:max(3, n_shots-2)]
    ms = [e for e in correct if not gg(e)["make"]][:2]
    walk = sorted(mk + ms, key=lambda e: e["t"])[:n_shots]
    print(f"demo shots: {len(walk)} ({sum(gg(e)['make'] for e in walk)} make/"
          f"{sum(not gg(e)['make'] for e in walk)} miss)")

    fps_n = cv2.VideoCapture(NR_VID).get(5) or 29.97
    fps_f = cv2.VideoCapture(FR_VID).get(5) or 29.97
    ts = np.linspace(walk[0]["t"], walk[-1]["t"], 6)
    nr_hoop = med_hoop(near, NR_VID, dev, 960, ts); fr_hoop = med_hoop(far, FR_VID, dev, 1280, ts)
    # crop centers: far -> rim + arc above; near -> rim region
    f_cx, f_cy = (fr_hoop[0]+fr_hoop[2])/2, (fr_hoop[1]+fr_hoop[3])/2 - 80
    n_cx, n_cy = (nr_hoop[0]+nr_hoop[2])/2, (nr_hoop[1]+nr_hoop[3])/2 - 40
    F_H, N_H = 540, 470

    pts_map, meta = [], []
    for e in walk:
        arc = far_arc(far, dev, e["t"], fr_hoop, fps_f)
        nxn, nyn = near_cross(near, dev, e["t"], nr_hoop, fps_n)
        m = gg(e)["make"]; pts_map.append((nxn, nyn, m))
        meta.append(dict(arc=arc, depth=float(np.clip(11-nyn*7, 0, 18)), lr=nxn*9, make=m))
        print(f"  t={e['t']:.0f} {gg(e)['gt']:16} arc={arc:.0f} depth={meta[-1]['depth']:.0f} lr={meta[-1]['lr']:+.0f}")

    vw = cv2.VideoWriter(str(OUT/"shot_detection_demo.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 24, (1920, 1080))
    capN, capF = cv2.VideoCapture(NR_VID), cv2.VideoCapture(FR_VID)
    for i, e in enumerate(walk):
        t = e["t"]
        info = panel_info(meta[i]["arc"], meta[i]["depth"], meta[i]["lr"], meta[i]["make"])
        rim = panel_rimmap(pts_map[:i+1], i)
        right = np.vstack([info, rim])
        capN.set(1, int((t-2.0)*fps_n)); capF.set(1, int((t-2.0)*fps_f))
        for _ in range(int(3.6*fps_n)):
            okn, frn = capN.read(); okf, frf = capF.read()
            if not (okn and okf):
                break
            rF = far.predict(frf, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
            fb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rF.boxes]
            fcrop, fx0, fy0, fw, fh = crop_aspect(frf, f_cx, f_cy, F_H)
            fc = cv2.resize(fcrop, (VW, VH)); draw_boxes(fc, fb, fx0, fy0, VW/fw, VH/fh)
            fc = label_bar(fc, "FAR ANGLE  ·  arc tracking")
            rN = near.predict(frn, conf=0.30, imgsz=960, device=dev, verbose=False)[0]
            nb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rN.boxes]
            ncrop, nx0, ny0, nw, nh = crop_aspect(frn, n_cx, n_cy, N_H)
            nc = cv2.resize(ncrop, (VW, VH)); draw_boxes(nc, nb, nx0, ny0, VW/nw, VH/nh)
            nc = label_bar(nc, "NEAR ANGLE  ·  make / miss", make=meta[i]["make"])
            left = np.vstack([fc, nc])
            vw.write(np.hstack([left, right]))
    capN.release(); capF.release(); vw.release()
    print(f"\n-> {OUT/'shot_detection_demo.mp4'}")


if __name__ == "__main__":
    main()
