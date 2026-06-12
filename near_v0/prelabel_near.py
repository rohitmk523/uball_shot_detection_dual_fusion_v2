#!/usr/bin/env python3
"""Pre-label near-angle (overhead rim view) frames with Grounding DINO.

Differences vs the far-angle auto_label_sam3.py (deliberate, from Phase 0):
  - NO ball-inside-hoop rejection: at the rim moment the ball is legitimately
    inside the hoop bbox -- those are the most valuable boxes.
  - Hoop = the rim+net circle seen from above (bottom-center of frame).
    Spatial prior: hoop center must be in the lower-center region.
  - Boxes only (no SAM2 mask refinement) -- prelabels get human-verified
    in annotate_server, masks add nothing for YOLO txt labels.

Classes: 0 = Basketball, 1 = Basketball Hoop (same as annotation tool).

Usage:
  python prelabel_near.py --mode sanity  --frames-dir D --out D_overlays -n 12
  python prelabel_near.py --mode full    --frames-dir D --labels-dir L [--overlays O]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

BALL_PROMPT = "basketball."
HOOP_PROMPT = "basketball hoop net."
BALL_CONF = 0.30          # generous: prelabels are human-verified
HOOP_CONF = 0.20
# hoop spatial prior (fraction of frame): center-x in [0.25,0.75], y in [0.55,1.0]
HOOP_X = (0.25, 0.75)
HOOP_Y = (0.55, 1.05)
HOOP_AR = (0.7, 2.2)      # w/h of overhead rim circle crop (wide-ish ok)
BALL_MAX_FRAC = 0.25      # ball box wider than 25% of frame = bogus


def load_model():
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device).eval()

    def predict(img: Image.Image, prompt: str):
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.15,
            target_sizes=[img.size[::-1]])[0]
        return [(float(s), [float(v) for v in b])
                for s, b in zip(res["scores"], res["boxes"])]
    return predict, device


def label_image(predict, img: Image.Image):
    W, H = img.size
    dets = []
    # hoop: best candidate passing spatial+aspect prior
    hoops = []
    for conf, (x1, y1, x2, y2) in predict(img, HOOP_PROMPT):
        if conf < HOOP_CONF:
            continue
        cx, cy, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, x2 - x1, y2 - y1
        if not (HOOP_X[0] < cx < HOOP_X[1] and HOOP_Y[0] < cy < HOOP_Y[1]):
            continue
        if not (HOOP_AR[0] < w / max(h, 1) < HOOP_AR[1]):
            continue
        hoops.append((conf, [x1, y1, x2, y2]))
    if hoops:
        conf, box = max(hoops)
        dets.append((1, conf, box))
    for conf, (x1, y1, x2, y2) in predict(img, BALL_PROMPT):
        if conf < BALL_CONF or (x2 - x1) > BALL_MAX_FRAC * W:
            continue
        dets.append((0, conf, [x1, y1, x2, y2]))
    return dets


def to_yolo(dets, W, H):
    lines = []
    for cls, _conf, (x1, y1, x2, y2) in dets:
        lines.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                     f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    return "\n".join(lines)


def overlay(img: Image.Image, dets, out: Path):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for cls, conf, box in dets:
        color = (255, 60, 60) if cls == 0 else (60, 220, 60)
        d.rectangle(box, outline=color, width=4)
        d.text((box[0] + 4, box[1] + 4), f"{'ball' if cls==0 else 'hoop'} {conf:.2f}",
               fill=color)
    im.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "full"], required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--labels-dir")
    ap.add_argument("--overlays")
    ap.add_argument("-n", type=int, default=12)
    args = ap.parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    if args.mode == "sanity":
        frames = frames[:: max(1, len(frames) // args.n)][: args.n]
    predict, device = load_model()
    print(f"grounding-dino on {device}, {len(frames)} frames")

    ov_dir = Path(args.overlays) if args.overlays else None
    lb_dir = Path(args.labels_dir) if args.labels_dir else None
    for p in (ov_dir, lb_dir):
        if p:
            p.mkdir(parents=True, exist_ok=True)
    n_ball = n_hoop = 0
    for i, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        dets = label_image(predict, img)
        n_ball += sum(1 for c, *_ in dets if c == 0)
        n_hoop += sum(1 for c, *_ in dets if c == 1)
        if lb_dir:
            (lb_dir / f"{fp.stem}.txt").write_text(to_yolo(dets, *img.size))
        if ov_dir and (args.mode == "sanity" or i % 25 == 0):
            overlay(img, dets, ov_dir / f"{fp.stem}_ov.jpg")
        if args.mode == "full" and i % 50 == 0:
            print(f"{i}/{len(frames)}", flush=True)
    print(f"done: {len(frames)} frames, {n_ball} ball boxes, {n_hoop} hoop boxes")


if __name__ == "__main__":
    sys.exit(main())
