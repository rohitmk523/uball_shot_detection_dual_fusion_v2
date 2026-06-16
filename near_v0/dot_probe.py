#!/usr/bin/env python3
"""Probe the rim-map dot position (new near_cross) for a pool of candidate
shots WITHOUT rendering. Dumps (pid8, basket, make, gt, nx, ny, ok) to JSON so
we can verify the localization fix on real data (makes -> near center, rim-outs
-> at the edge) and curate a clean reel before the expensive render.

One basket per run (each basket uses its own near video: NR=RIGHT, NL=LEFT).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_game import med_hoop, near_cross  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--basket", required=True)
    ap.add_argument("--near-w", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    near = YOLO(getattr(a, "near_w"))
    data = json.load(open(a.shots))
    shots = [s for s in data["shots"] if s["basket"] == a.basket]
    print(f"{a.basket}: {len(shots)} shots", flush=True)
    if not shots:
        json.dump([], open(a.out, "w"))
        return
    fps = cv2.VideoCapture(a.near).get(5) or 29.97
    ts = np.linspace(shots[0]["t0"], shots[-1]["t0"], 8)
    nh = med_hoop(near, a.near, dev, 960, ts)
    out = []
    for s in shots:
        nx, ny, _, ok = near_cross(near, a.near, dev, s["t0"], s["t1"], nh, fps)
        out.append({
            "pid8": s["pid8"], "basket": a.basket, "make": bool(s["pred_make"]),
            "gt": s["gt"], "prob": s.get("prob"), "t0": s["t0"], "t1": s["t1"],
            "nx": nx, "ny": ny, "ok": ok,
        })
        sx = f"{nx:+.2f}" if nx is not None else "  None"
        sy = f"{ny:+.2f}" if ny is not None else "  None"
        print(f"  {s['pid8']} {a.basket} {'MK' if s['pred_make'] else 'MS'} "
              f"nx={sx} ny={sy} ok={ok} {s['gt']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=0)
    print(f"PROBE_DONE {a.basket} {len(out)}", flush=True)


if __name__ == "__main__":
    main()
