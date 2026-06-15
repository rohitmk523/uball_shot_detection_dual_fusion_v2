#!/usr/bin/env python3
"""4-panel shot-detection demo (Noah-style), 1080p, Pillow-rendered UI.
  top-left  FAR angle (arc)        top-right NEAR angle (rim + make/miss)
  bot-left  METRICS arc/depth/L-R  bot-right RIM MAP (green made / red miss)
Data panels are rendered once per shot in high quality and composited onto the
per-frame video panels. Far arc = parabola fit on the far ball track; depth/L-R
+ rim-map = near rim-crossing position (approx until an overhead camera).
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
PW, PH = 960, 540           # panel -> 1920x1080 grid

BG = (16, 17, 21); CARD = (28, 30, 36); GREEN = (76, 217, 100); RED = (255, 69, 58)
AMBER = (255, 184, 48); ORANGE = (255, 149, 40); WHITE = (244, 245, 248); GREY = (138, 142, 152)
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FBLK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
_FC = {}


def font(path, sz):
    k = (path, sz)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(path, sz)
    return _FC[k]


def gg(e):
    x = e["gt"]
    return ast.literal_eval(x) if isinstance(x, str) else x


def ctext(dr, xy, s, fnt, fill, anchor="mm"):
    dr.text(xy, s, font=fnt, fill=fill, anchor=anchor)


def bgr(pil_img):
    return np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()


def panel_metrics(arc, depth, lr, make):
    im = Image.new("RGB", (PW, PH), BG); dr = ImageDraw.Draw(im)
    ctext(dr, (40, 44), "SHOT METRICS", font(FB, 34), WHITE, "lm")
    cols = [("ARC", f"{arc:.0f}", "°", GREEN if 43 <= arc <= 50 else AMBER, 200),
            ("DEPTH", f'{depth:.0f}"', "ideal 11", GREEN if 9 <= depth <= 13 else AMBER, 480),
            ("L / R", f'{lr:+.0f}"', "ideal 0", GREEN if abs(lr) <= 3 else AMBER, 760)]
    dr.line([(330, 120), (330, 300)], fill=(55, 58, 66), width=2)
    dr.line([(610, 120), (610, 300)], fill=(55, 58, 66), width=2)
    for name, val, sub, col, x in cols:
        ctext(dr, (x, 150), name, font(FB, 26), GREY)
        ctext(dr, (x, 215), val, font(FBLK, 92), col)
        ctext(dr, (x, 280), sub, font(FB, 22), GREY)
    # verdict pill
    vc = GREEN if make else RED; lab = "MAKE" if make else "MISS"
    dr.rounded_rectangle([330, 360, 630, 470], radius=22, fill=(vc[0], vc[1], vc[2]))
    ctext(dr, (480, 416), lab, font(FBLK, 64), (16, 17, 21))
    ctext(dr, (480, 505), "auto-detected make / miss", font(FB, 22), GREY)
    return bgr(im)


def panel_rimmap(points, cur):
    im = Image.new("RGB", (PW, PH), BG); dr = ImageDraw.Draw(im)
    ctext(dr, (40, 44), "RIM MAP", font(FB, 34), WHITE, "lm")
    ctext(dr, (PW-220, 44), "made", font(FB, 24), GREEN, "lm")
    dr.ellipse([PW-250, 36, PW-232, 54], fill=GREEN)
    ctext(dr, (PW-120, 44), "miss", font(FB, 24), RED, "lm")
    dr.line([(PW-150, 36), (PW-134, 52)], fill=RED, width=4)
    dr.line([(PW-150, 52), (PW-134, 36)], fill=RED, width=4)
    cx, cy, R = PW//2, PH//2 + 26, 190
    dr.ellipse([cx-R-9, cy-R-9, cx+R+9, cy+R+9], outline=ORANGE, width=18)   # rim
    dr.ellipse([cx-6, cy-6, cx+6, cy+6], fill=(70, 72, 80))                  # center
    for i, (nx, ny, mk) in enumerate(points):
        px, py = int(cx + nx*R*0.92), int(cy + ny*R*0.92)
        hi = (i == cur)
        if hi:
            dr.ellipse([px-19, py-19, px+19, py+19], outline=WHITE, width=3)
        if mk:
            r = 13 if hi else 10
            dr.ellipse([px-r, py-r, px+r, py+r], fill=GREEN)
            dr.ellipse([px-r, py-r, px-r+8, py-r+8], fill=(150, 240, 165))
        else:
            s = 13 if hi else 10
            dr.line([(px-s, py-s), (px+s, py+s)], fill=RED, width=5)
            dr.line([(px-s, py+s), (px+s, py-s)], fill=RED, width=5)
    return bgr(im)


def label_bar(frame, text, make=None):
    """draw a crisp gradient label bar + optional make/miss banner via PIL."""
    im = Image.fromarray(frame[:, :, ::-1]); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, PW, 64], fill=(10, 11, 14, 180))
    ctext(dr, (28, 32), text, font(FB, 32), WHITE, "lm")
    if make is not None:
        vc = GREEN if make else RED; lab = "MAKE" if make else "MISS"
        dr.rectangle([0, PH-66, PW, PH], fill=(10, 11, 14, 205))
        dr.rounded_rectangle([28, PH-56, 190, PH-12], radius=14, fill=vc+(255,))
        ctext(dr, (109, PH-34), lab, font(FBLK, 34), (16, 17, 21))
    return np.array(im)[:, :, ::-1].copy()


def draw_boxes(frame, boxes, x0, y0, sx, sy):
    for cls, b in boxes:
        p1 = (int((b[0]-x0)*sx), int((b[1]-y0)*sy))
        p2 = (int((b[2]-x0)*sx), int((b[3]-y0)*sy))
        col = (90, 220, 90) if cls == BALL else (40, 150, 255)
        cv2.rectangle(frame, p1, p2, col, 3)
    return frame


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


def nr_crop(rim, W=1920, H=1080, scale=1.6):
    cx = (rim[0]+rim[2])/2; w = rim[2]-rim[0]; side = int(scale*w)
    x1 = int(np.clip(cx-side/2, 0, W-side))
    cy = (rim[1]+rim[3])/2; y1 = int(np.clip(cy-side*0.55, 0, max(0, H-side)))
    return x1, y1, min(side, W-x1, H-y1)


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
    return float(np.clip(best[1], -1.2, 1.2)), float(np.clip(best[2], -1.2, 1.2))


def main():
    n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    near = __import__("ultralytics").YOLO(str(NEAR_W)); far = __import__("ultralytics").YOLO(FAR_W)

    d = json.load(open(REPO / f"data/near_detector/e2e_{GAME}.json"))
    matched = [e for e in d["events"] if e.get("gt") and e["gt"] not in (None, "None")]
    by = {}
    for e in matched:
        pid = gg(e)["pid"]
        if pid not in by or abs(e["t"]-gg(e)["t1"]) < abs(by[pid]["t"]-gg(by[pid])["t1"]):
            by[pid] = e
    correct = [e for e in by.values() if e["pred_make"] == gg(e)["make"]]
    mk = [e for e in correct if gg(e)["make"]][:max(3, n_shots-2)]
    ms = [e for e in correct if not gg(e)["make"]][:2]
    walk = sorted(mk + ms, key=lambda e: e["t"])[:n_shots]
    print(f"demo shots: {len(walk)} ({sum(gg(e)['make'] for e in walk)} make / "
          f"{sum(not gg(e)['make'] for e in walk)} miss)")

    fps_n = cv2.VideoCapture(NR_VID).get(5) or 29.97
    fps_f = cv2.VideoCapture(FR_VID).get(5) or 29.97
    ts = np.linspace(walk[0]["t"], walk[-1]["t"], 6)
    nr_hoop = med_hoop(near, NR_VID, dev, 960, ts); fr_hoop = med_hoop(far, FR_VID, dev, 1280, ts)
    nx1, ny1, nside = nr_crop(nr_hoop)
    fcx = (fr_hoop[0]+fr_hoop[2])/2
    fx0, fx1 = max(0, int(fcx-380)), int(fcx+380)
    fy0, fy1 = max(0, int(fr_hoop[1]-320)), int(fr_hoop[3]+140)

    pts_map, meta = [], []
    for e in walk:
        arc = far_arc(far, dev, e["t"], fr_hoop, fps_f)
        nxn, nyn = near_cross(near, dev, e["t"], nr_hoop, fps_n)
        m = gg(e)["make"]
        pts_map.append((nxn, nyn, m))
        meta.append(dict(arc=arc, depth=float(np.clip(11-nyn*7, 0, 18)), lr=nxn*9, make=m))
        print(f"  t={e['t']:.0f} {gg(e)['gt']:16} arc={arc:.0f} depth={meta[-1]['depth']:.0f} lr={meta[-1]['lr']:+.0f}")

    vw = cv2.VideoWriter(str(OUT/"shot_detection_demo.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), 24, (PW*2, PH*2))
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(OUT/"shot_detection_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 24, (PW*2, PH*2))
    capN, capF = cv2.VideoCapture(NR_VID), cv2.VideoCapture(FR_VID)
    for i, e in enumerate(walk):
        t = e["t"]
        met_np = panel_metrics(meta[i]["arc"], meta[i]["depth"], meta[i]["lr"], meta[i]["make"])
        rim_np = panel_rimmap(pts_map[:i+1], i)
        capN.set(1, int((t-2.0)*fps_n)); capF.set(1, int((t-2.0)*fps_f))
        for _ in range(int(3.5*fps_n)):
            okn, frn = capN.read(); okf, frf = capF.read()
            if not (okn and okf):
                break
            rF = far.predict(frf, conf=0.25, imgsz=1280, device=dev, verbose=False)[0]
            fb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rF.boxes]
            fc = cv2.resize(frf[fy0:fy1, fx0:fx1], (PW, PH))
            draw_boxes(fc, fb, fx0, fy0, PW/(fx1-fx0), PH/(fy1-fy0))
            fc = label_bar(fc, "FAR ANGLE  ·  arc tracking")
            rN = near.predict(frn, conf=0.30, imgsz=960, device=dev, verbose=False)[0]
            nb = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in rN.boxes]
            nc = cv2.resize(frn[ny1:ny1+nside, nx1:nx1+nside], (PW, PH))
            draw_boxes(nc, nb, nx1, ny1, PW/nside, PH/nside)
            nc = label_bar(nc, "NEAR ANGLE  ·  make / miss", make=meta[i]["make"])
            grid = np.vstack([np.hstack([fc, nc]), np.hstack([met_np, rim_np])])
            vw.write(grid)
    capN.release(); capF.release(); vw.release()
    print(f"\n-> {OUT/'shot_detection_demo.mp4'}")


if __name__ == "__main__":
    main()
