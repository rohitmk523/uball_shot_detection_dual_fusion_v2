#!/usr/bin/env python3
"""Repair pass for a validation game: refill failed clips, re-run the
shots whose results are missing/bare-UNDECIDED, re-merge final_robust.

Network-failed clip downloads leave shots with zero dual samples ->
bare "UNDECIDED" in the merge. This: (1) re-runs extraction (good clips
skip instantly, failed ones redownload), (2) re-runs L1 for the affected
shots only, (3) hires for those still UND, (4) re-merges.

Usage: python pipeline/repair_val_game.py --game-id cc5deb39 --sync-frames 7
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAGS = {"DESCENT_BOUNDS": "1", "DESCENT_SKIP_NOISY": "1",
         "DESCENT_MAX_SKIPS": "8", "DESCENT_Y_GUARD": "1"}
MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}


def cat(g: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if g in MAKE and v.startswith("MAKE"):
        return "TP"
    if g in MISS and v.startswith("MISS"):
        return "TN"
    if g in MISS and v.startswith("MAKE"):
        return "FP"
    return "FN"


def merge_game(G: Path) -> dict:
    shots = json.loads((G / "shots_right.json").read_text())
    rows, roll = [], defaultdict(int)
    for s in shots:
        v_l1 = v_hr = ""
        rp = G / "results_sam3" / f"{s['name']}.json"
        if rp.exists():
            v_l1 = json.loads(rp.read_text()).get("verdict", "")
        rph = G / "results_hires_sam3" / f"{s['name']}.json"
        if rph.exists():
            v_hr = json.loads(rph.read_text()).get("verdict", "")
        if v_hr and not v_hr.startswith("UNDECIDED"):
            v, layer = v_hr, "hires"
        elif v_l1 and not v_l1.startswith("UNDECIDED"):
            v, layer = v_l1, "L1"
        else:
            v, layer = (v_hr or v_l1 or "UNDECIDED"), "und"
        c = cat(s["gt"], v)
        roll[c] += 1
        rows.append(dict(name=s["name"], gt=s["gt"], verdict=v,
                         layer=layer, cat=c))
    (G / "final_robust.json").write_text(json.dumps(rows, indent=1))
    return dict(roll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--sync-frames", type=int, required=True)
    ap.add_argument("--dir-prefix", default="val_")
    args = ap.parse_args()
    gid = args.game_id
    G = ROOT / f"data/client_report/triangulation_test/{args.dir_prefix}{gid}"
    py = sys.executable

    # 1. refill clips (two passes; good clips skip via ffprobe check)
    for _ in range(2):
        subprocess.run([py, "-u", "pipeline/extract_val_clips.py",
                        "--game-id", gid, "--dir-prefix", args.dir_prefix,
                        "--sync-frames", str(args.sync_frames)], cwd=ROOT)

    # 2. shots needing re-run: results missing OR bare UNDECIDED with 0 samples
    shots = json.loads((G / "shots_right.json").read_text())
    redo = []
    for s in shots:
        rp = G / "results_sam3" / f"{s['name']}.json"
        if not rp.exists():
            redo.append(s["name"])
            continue
        d = json.loads(rp.read_text())
        if d.get("n_samples", 0) == 0:
            redo.append(s["name"])
    print(f"[repair] {gid}: {len(redo)} shots to re-run")

    env = {**os.environ, **FLAGS, "CALIB_JUNE_SAM3": "1",
           "CALIB_JUNE_JSON": str(ROOT / "data/client_report/triangulation_test"
                                  f"/calibration_{args.dir_prefix}{gid}_sam3.json")}
    if redo:
        subprocess.run([py, "-u", "pipeline/triangulate_shot.py",
                        "--shots-json", str(G / "clips/shots_pipeline.json"),
                        "--clips-dir", str(G / "clips"),
                        "--out-dir", str(G / "results_sam3"),
                        "--only", ",".join(redo),
                        "--conf", "0.20", "--imgsz", "640"],
                       env=env, cwd=ROOT)
        still_und = []
        for n in redo:
            rp = G / "results_sam3" / f"{n}.json"
            if rp.exists() and json.loads(rp.read_text()).get(
                    "verdict", "").startswith("UNDECIDED"):
                still_und.append(n)
        print(f"[repair] {len(still_und)} still UND -> hires")
        if still_und:
            subprocess.run([py, "-u", "pipeline/triangulate_shot.py",
                            "--shots-json", str(G / "clips/shots_pipeline.json"),
                            "--clips-dir", str(G / "clips"),
                            "--out-dir", str(G / "results_hires_sam3"),
                            "--only", ",".join(still_und),
                            "--conf", "0.05", "--imgsz", "1280"],
                           env=env, cwd=ROOT)

    # 3. re-merge
    roll = merge_game(G)
    tp, tn = roll.get("TP", 0), roll.get("TN", 0)
    fp, fn, und = roll.get("FP", 0), roll.get("FN", 0), roll.get("UND", 0)
    dec = tp + tn + fp + fn
    n = dec + und
    print(f"=== {args.dir_prefix}{gid} REPAIRED: TP={tp} TN={tn} FP={fp} FN={fn} "
          f"UND={und}  dec={100*(tp+tn)/dec:.1f}%  ovr={100*(tp+tn)/n:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
