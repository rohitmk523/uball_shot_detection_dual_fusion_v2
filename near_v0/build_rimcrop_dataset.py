#!/usr/bin/env python3
"""Phase 1: build the rim-crop classifier dataset (NEAR_ANGLE_PLAN.md +
PHASE0_TAXONOMY.md section 4 spec).

Per GT shot (Supabase plays, data/near_detector/plays_all.json):
  1. rim box = median DINO hoop box for that game/angle
     (data/near_detector/labels/*.txt), frozen per game -- rim is static.
  2. cut a working segment [t1-3.0s, t1+3.0s] from the NL/NR video
  3. anchor = peak frame-diff motion energy INSIDE the rim crop
     (taxonomy section 4: never global motion -- fixes the teal-ball misses)
  4. save stride-1 square crop video: anchor-0.7s .. +1.4s (core+post window),
     320x320, plus meta JSON (label, peak energy, rim box, timings).

Labels here are binary make/miss from GT; no_event/unreadable get assigned in
review using the recorded peak-energy + later spot checks.

Usage:
  --game GID8 --angle NR|NL --video PATH_OR_URL  (one game/angle pass)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ND = REPO / "data/near_detector"
OUT = REPO / "data/near_rimcrop"

SEG_BEFORE, SEG_AFTER = 3.0, 3.0      # working segment around t1
WIN_BEFORE, WIN_AFTER = 0.7, 1.4      # saved window around motion anchor
CROP_SCALE = 1.6                       # square side = 1.6 x hoop bbox width
OUT_SIZE = 320


def rim_box_for(gid: str, cam: str):
    """Median hoop box (px) from DINO prelabels; None if no labels yet."""
    boxes = []
    for lp in (ND / "labels").glob(f"{gid}_{cam}_*.txt"):
        for line in lp.read_text().splitlines():
            p = line.split()
            if p and p[0] == "1":
                cx, cy, w, h = (float(v) for v in p[1:5])
                boxes.append((cx * 1920, cy * 1080, w * 1920, h * 1080))
    if len(boxes) < 5:
        return None
    cx, cy, w, h = (float(np.median([b[i] for b in boxes])) for i in range(4))
    return cx, cy, w, h


def crop_rect(rim, W=1920, H=1080):
    cx, cy, w, _h = rim
    side = int(CROP_SCALE * w)
    # bias upward slightly: include entry arc above the rim
    x1 = int(np.clip(cx - side / 2, 0, W - side))
    y1 = int(np.clip(cy - side * 0.55, 0, max(0, H - side)))
    side = min(side, W - x1, H - y1)
    return x1, y1, side


def cut_segment(video: str, t0: float, dur: float, out_path: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.2f}", "-i", video,
         "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-an", str(out_path)], capture_output=True, timeout=300)
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 50_000


def process_shot(video: str, play: dict, rim, out_dir: Path) -> dict | None:
    t1 = play["t1"]
    seg_t0 = max(0.0, t1 - SEG_BEFORE)
    with tempfile.TemporaryDirectory() as td:
        seg = Path(td) / "seg.mp4"
        if not cut_segment(video, seg_t0, SEG_BEFORE + SEG_AFTER, seg):
            return None
        cap = cv2.VideoCapture(str(seg))
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) < int(fps * 2):
            return None

        x1, y1, side = crop_rect(rim)
        prev, energy = None, np.zeros(len(frames))
        for i, f in enumerate(frames):
            g = cv2.cvtColor(f[y1:y1 + side, x1:x1 + side], cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            if prev is not None:
                energy[i] = float(np.mean(cv2.absdiff(g, prev)))
            prev = g
        peak = int(np.argmax(energy))
        lo = max(0, peak - int(WIN_BEFORE * fps))
        hi = min(len(frames), peak + int(WIN_AFTER * fps))

        name = f"{play['gid8']}_{play['cam']}_{play['pid8']}"
        vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (OUT_SIZE, OUT_SIZE))
        for f in frames[lo:hi]:
            vw.write(cv2.resize(f[y1:y1 + side, x1:x1 + side],
                                (OUT_SIZE, OUT_SIZE)))
        vw.release()
        return {"name": name, "gt": play["cls"],
                "make": play["cls"].endswith("MAKE"),
                "gid8": play["gid8"], "cam": play["cam"], "pid8": play["pid8"],
                "t1_game": t1, "anchor_in_seg": round(peak / fps, 3),
                "peak_energy": round(float(energy[peak]), 3),
                "n_frames": hi - lo, "fps": round(fps, 3),
                "rim_box": [round(v, 1) for v in rim],
                "crop": [x1, y1, side]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--angle", required=True, choices=["NR", "NL"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    plays = json.loads((ND / "plays_all.json").read_text())
    want_angle = "RIGHT" if args.angle == "NR" else "LEFT"
    plays = [{**p, "t0": float(p["t0"]), "t1": float(p["t1"]), "cam": args.angle}
             for p in plays if p["gid8"] == args.game and p["angle"] == want_angle]
    if not plays:
        print(f"no plays for {args.game} {args.angle}", file=sys.stderr)
        return 1
    rim = rim_box_for(args.game, args.angle)
    if rim is None:
        print(f"no rim box for {args.game} {args.angle} -- run prelabels first",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out) / f"{args.game}_{args.angle}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, fails = [], 0
    for i, p in enumerate(plays):
        m = process_shot(args.video, p, rim, out_dir)
        if m:
            meta.append(m)
        else:
            fails += 1
        if i % 20 == 0:
            print(f"{i}/{len(plays)}", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=0))
    print(f"{args.game}_{args.angle}: {len(meta)} clips, {fails} failed")
    return 0 if fails < len(plays) * 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
