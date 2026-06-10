#!/usr/bin/env python3
"""Out-of-sample validation: run the production pipeline on the May games.

Same structure as rerun_june_hybrid.py but:
  - val_<gid8> directory prefix, calibration_val_<gid8>_sam3.json
  - PRODUCTION FLAG SET enabled (the June-tuned verdict guards):
      DESCENT_BOUNDS=1 DESCENT_SKIP_NOISY=1 DESCENT_MAX_SKIPS=8 DESCENT_Y_GUARD=1
  - also rescores with flags OFF for the legacy baseline comparison

Usage:
  python pipeline/run_val_games.py --game-id 77715f25
  python pipeline/run_val_games.py --all
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
GAMES = ("77715f25", "cc1710c4", "fb677f72")
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
    if g in MAKE and v.startswith("MISS"):
        return "FN"
    return "?"


def run_one(gid: str, prefix: str = "val_") -> dict:
    G = ROOT / f"data/client_report/triangulation_test/{prefix}{gid}"
    clips = G / "clips"
    manifest = clips / "shots_pipeline.json"
    calib = ROOT / f"data/client_report/triangulation_test/calibration_{prefix}{gid}_sam3.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not calib.exists():
        raise FileNotFoundError(calib)

    results = G / "results_sam3"
    results_hires = G / "results_hires_sam3"
    for d in (results, results_hires):
        d.mkdir(exist_ok=True)
        for f in d.glob("*.json"):   # clear stale results (esp. old hires)
            f.unlink()

    env = {**os.environ, **FLAGS,
           "CALIB_JUNE_SAM3": "1",
           "CALIB_JUNE_JSON": str(calib)}

    print(f"[L1] val_{gid}")
    r = subprocess.run(
        [sys.executable, "pipeline/triangulate_shot.py",
         "--shots-json", str(manifest),
         "--clips-dir", str(clips),
         "--out-dir", str(results),
         "--conf", "0.20", "--imgsz", "640"],
        env=env, cwd=ROOT)
    print(f"  L1 exit: {r.returncode}")

    shots = json.loads((G / "shots_right.json").read_text())
    und = []
    for s in shots:
        rp = results / f"{s['name']}.json"
        if rp.exists() and json.loads(rp.read_text()).get(
                "verdict", "").startswith("UNDECIDED"):
            und.append(s["name"])
    print(f"[hi-res] {len(und)} UND shots")
    if und:
        r = subprocess.run(
            [sys.executable, "pipeline/triangulate_shot.py",
             "--shots-json", str(manifest),
             "--clips-dir", str(clips),
             "--out-dir", str(results_hires),
             "--only", ",".join(und),
             "--conf", "0.05", "--imgsz", "1280"],
            env=env, cwd=ROOT)
        print(f"  hi-res exit: {r.returncode}")

    rows = []
    roll: dict[str, int] = defaultdict(int)
    for s in shots:
        v_l1 = v_hr = ""
        rp = results / f"{s['name']}.json"
        if rp.exists():
            v_l1 = json.loads(rp.read_text()).get("verdict", "")
        rph = results_hires / f"{s['name']}.json"
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
    return dict(gid=gid, roll=dict(roll))


def tally(label: str, roll: dict) -> tuple:
    tp, tn = roll.get("TP", 0), roll.get("TN", 0)
    fp, fn, und = roll.get("FP", 0), roll.get("FN", 0), roll.get("UND", 0)
    dec = tp + tn + fp + fn
    n = dec + und
    print(f"=== {label}: TP={tp} TN={tn} FP={fp} FN={fn} UND={und}  "
          f"dec={100*(tp+tn)/dec if dec else 0:.1f}%  "
          f"ovr={100*(tp+tn)/n if n else 0:.1f}%")
    return tp, tn, fp, fn, und


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prefix", default="val_",
                    help="game dir prefix (val_ or june_)")
    args = ap.parse_args()
    targets = GAMES if args.all else (args.game_id,)
    agg: dict[str, int] = defaultdict(int)
    for gid in targets:
        res = run_one(gid, args.prefix)
        t = tally(f"{args.prefix}{gid}", res["roll"])
        for k, v in zip(("TP", "TN", "FP", "FN", "UND"), t):
            agg[k] += v
    if len(targets) > 1:
        tally("VALIDATION AGGREGATE", agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
