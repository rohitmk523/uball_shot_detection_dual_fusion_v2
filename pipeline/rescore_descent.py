#!/usr/bin/env python3
"""Re-apply descent_verdict on cached samples for an existing results dir.

Used to test changes to descent_verdict / gap_stop rules without re-running
YOLO. Writes updated verdicts back into each json (verdict + descent fields).
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from triangulate_shot import (   # noqa: E402
    descent_verdict, nr_rebound_check, sanitize_samples,
    RIM_X, RIM_Y, RIM_Z,
)

MAKE = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def classify(g: str, v: str) -> str:
    if v.startswith("UNDECIDED"): return "UND"
    if g in MAKE and v.startswith("MAKE"): return "TP"
    if g in MISS and v.startswith("MISS"): return "TN"
    if g in MISS and v.startswith("MAKE"): return "FP"
    if g in MAKE and v.startswith("MISS"): return "FN"
    return "?"


def rescore_one(samples: list[dict]) -> tuple[dict, str]:
    # DESCENT_BOUNDS=1: drop physically-impossible triangulations (outside
    # court bounds) BEFORE apex selection. See triangulate_shot.sanitize_samples
    # — fixes depth-degenerate samples hijacking argmax(z) on FT/3PT arcs.
    if os.environ.get("DESCENT_BOUNDS") == "1":
        samples = sanitize_samples(samples)
    if len(samples) < 3:
        return {}, "UNDECIDED (no samples)"
    zs = [s['X_cm'][2] for s in samples]
    apex_idx = int(np.argmax(zs))
    apex_x, apex_y, apex_z = samples[apex_idx]['X_cm']
    apex_dxy = float(np.hypot(apex_x - RIM_X, apex_y - RIM_Y))
    info, descent_v = descent_verdict(samples, apex_idx, apex_dxy, apex_z)
    # NR-rebound override
    if descent_v.startswith("MAKE"):
        rebound, max_cy, min_cy_after, dt_reb = nr_rebound_check(
            samples, apex_idx)
        post_apex_z_min = min(
            (samples[i]['X_cm'][2]
             for i in range(apex_idx, min(apex_idx+30, len(samples)))),
            default=9999)
        if rebound and post_apex_z_min > 150:
            descent_v = (f"MISS (NR-rebound: cy {max_cy:.0f}->{min_cy_after:.0f}, "
                         f"D={max_cy-min_cy_after:.0f}px)")
    return info or {}, descent_v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Path to results dir containing per-shot json files")
    ap.add_argument("--manifest", required=True,
                    help="Path to shots manifest (with name + gt fields)")
    ap.add_argument("--write", action="store_true",
                    help="Write updated verdicts back into the json files")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    manifest = json.loads(Path(args.manifest).read_text())
    gt_map = {s['name']: s['gt'] for s in manifest}

    rows: list[dict] = []
    changes = 0
    for f in sorted(rdir.glob("*.json")):
        if f.name == "summary.json": continue
        d = json.loads(f.read_text())
        name = d['name']
        if name not in gt_map: continue
        samples = d.get('samples', [])
        old_v = d.get('verdict', '')
        info, new_v = rescore_one(samples)
        gt = gt_map[name]
        old_c = classify(gt, old_v); new_c = classify(gt, new_v)
        rows.append(dict(name=name, gt=gt, old=old_v, new=new_v,
                         old_c=old_c, new_c=new_c))
        if old_v != new_v:
            changes += 1
            marker = ("+" if old_c in ("FP","FN","UND") and new_c in ("TP","TN")
                      else "-" if old_c in ("TP","TN") and new_c in ("FP","FN")
                      else ".")
            print(f"  {marker} {name:18s} {gt:18s} {old_c:3s}->{new_c:3s}")
            print(f"      old: {old_v[:78]}")
            print(f"      new: {new_v[:78]}")
            if args.write:
                d['verdict'] = new_v
                d.setdefault('rescore', {}).update(info)
                f.write_text(json.dumps(d, indent=2))

    print(f"\n  {changes} verdicts changed (write={args.write})")

    for tag, key in [("OLD", "old"), ("NEW", "new")]:
        roll: dict[str, int] = defaultdict(int)
        for r in rows: roll[classify(r['gt'], r[key])] += 1
        n = sum(roll.values()); dec = roll['TP']+roll['TN']+roll['FP']+roll['FN']
        acc = 100*(roll['TP']+roll['TN'])/dec if dec else 0
        print(f"  {tag:3s}: TP={roll['TP']} TN={roll['TN']} "
              f"FP={roll['FP']} FN={roll['FN']} UND={roll['UND']}  "
              f"acc={acc:.1f}%  ({dec}/{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
