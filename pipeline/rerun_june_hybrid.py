#!/usr/bin/env python3
"""Re-run L1 + hi-res-on-UND with hybrid calibration (AUTO FR + SAM3 NR).

Same structure as rerun_june_sam3.py / rerun_june_auto.py; output dirs are
``results_hybrid/`` + ``results_hires_hybrid/`` and tally is written to
``final_hybrid.json`` per game.

Usage:
  python pipeline/rerun_june_hybrid.py --game-id 4692eb2b
  python pipeline/rerun_june_hybrid.py --all
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
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def rerun_one(gid: str) -> dict:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    clips = G / "clips"
    pipeline_manifest = clips / "shots_pipeline.json"
    if not pipeline_manifest.exists():
        raise FileNotFoundError(f"missing {pipeline_manifest}")

    hybrid_calib = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_hybrid.json"
    if not hybrid_calib.exists():
        raise FileNotFoundError(f"missing hybrid calibration: {hybrid_calib.name}")

    results = G / "results_hybrid"
    results_hires = G / "results_hires_hybrid"
    for d in (results, results_hires):
        d.mkdir(exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()

    env = {**os.environ,
           "CALIB_JUNE_HYBRID": "1",
           "CALIB_JUNE_JSON": str(hybrid_calib)}

    print(f"[L1-hybrid] {gid}")
    r = subprocess.run(
        [sys.executable, "pipeline/triangulate_shot.py",
         "--shots-json", str(pipeline_manifest),
         "--clips-dir", str(clips),
         "--out-dir", str(results),
         "--conf", "0.20", "--imgsz", "640"],
        env=env)
    print(f"  L1 exit: {r.returncode}")

    shots = json.loads((G / "shots_right.json").read_text())
    und_targets = []
    for s in shots:
        rp = results / f"{s['name']}.json"
        if not rp.exists():
            continue
        d = json.loads(rp.read_text())
        if d.get('verdict', '').startswith('UNDECIDED'):
            und_targets.append(s['name'])
    print(f"[hi-res-hybrid] {len(und_targets)} UND shots")
    if und_targets:
        r = subprocess.run(
            [sys.executable, "pipeline/triangulate_shot.py",
             "--shots-json", str(pipeline_manifest),
             "--clips-dir", str(clips),
             "--out-dir", str(results_hires),
             "--only", ",".join(und_targets),
             "--conf", "0.05", "--imgsz", "1280"],
            env=env)
        print(f"  hi-res exit: {r.returncode}")

    MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
    MISS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}

    def cat(g, v):
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

    rows = []
    roll = defaultdict(int)
    for s in shots:
        v_l1 = v_hr = ""
        rp = results / f"{s['name']}.json"
        if rp.exists():
            v_l1 = json.loads(rp.read_text()).get('verdict', '')
        rph = results_hires / f"{s['name']}.json"
        if rph.exists():
            v_hr = json.loads(rph.read_text()).get('verdict', '')
        if v_hr and not v_hr.startswith('UNDECIDED'):
            v, layer = v_hr, "hires"
        elif v_l1 and not v_l1.startswith('UNDECIDED'):
            v, layer = v_l1, "L1"
        elif v_hr:
            v, layer = v_hr, "hires-und"
        else:
            v, layer = v_l1 or "UNDECIDED", "L1-und"
        c = cat(s['gt'], v)
        roll[c] += 1
        rows.append(dict(name=s['name'], gt=s['gt'], verdict=v,
                         layer=layer, cat=c))
    (G / "final_hybrid.json").write_text(json.dumps(rows, indent=2))
    return dict(gid=gid, roll=dict(roll))


def print_tally(gid: str, roll: dict) -> tuple[int, int, int, int, int]:
    tp = roll.get('TP', 0); tn = roll.get('TN', 0)
    fp = roll.get('FP', 0); fn = roll.get('FN', 0); und = roll.get('UND', 0)
    n = tp + tn + fp + fn + und
    dec = tp + tn + fp + fn
    acc_d = 100 * (tp + tn) / dec if dec else 0
    acc_o = 100 * (tp + tn) / n if n else 0
    print(f"\n=== {gid} HYBRID FINAL ===")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn} UND={und}")
    print(f"  Decided: {tp + tn}/{dec} = {acc_d:.1f}%")
    print(f"  Overall: {tp + tn}/{n} = {acc_o:.1f}%")
    return tp, tn, fp, fn, und


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.game_id and not args.all:
        print("specify --game-id or --all"); return 1

    target = GAMES if args.all else (args.game_id,)
    agg = defaultdict(int)
    for gid in target:
        print("\n" + "=" * 60)
        result = rerun_one(gid)
        tp, tn, fp, fn, und = print_tally(gid, result['roll'])
        agg['TP'] += tp; agg['TN'] += tn
        agg['FP'] += fp; agg['FN'] += fn; agg['UND'] += und

    if args.all:
        print("\n" + "=" * 60)
        print_tally("AGGREGATE (4 games)", agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
