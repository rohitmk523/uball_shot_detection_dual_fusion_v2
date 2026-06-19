#!/usr/bin/env python3
"""Final merge: triangulation baseline -> UND-rerun (confidence filtered) ->
per-camera ensemble (only on STILL-UND cases, only strong signals).

Each layer can only PROMOTE UND to decided; never override an already-decided
verdict. This guarantees we never DROP accuracy below the previous layer.
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_und_results import is_high_conf

ROOT = Path(__file__).resolve().parent.parent
FG = ROOT / "data/client_report/triangulation_test/full_game"
MAIN = FG / "results"
RERUN = FG / "results_und_conf10"
ENS = json.loads((FG / "ensemble_88.json").read_text())
ENS_MAP = {r['name']: r for r in ENS}

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def classify(gt: str, verdict: str) -> str:
    if verdict.startswith("UNDECIDED"): return "UND"
    if gt in MAKE_LABELS and verdict.startswith("MAKE"):  return "TP"
    if gt in MISS_LABELS and verdict.startswith("MISS"):  return "TN"
    if gt in MISS_LABELS and verdict.startswith("MAKE"):  return "FP"
    if gt in MAKE_LABELS and verdict.startswith("MISS"):  return "FN"
    return "?"


def main():
    manifest = json.loads((FG / "shots_88.json").read_text())
    gt_map = {s['name']: s['gt'] for s in manifest}

    # Layer 1: original triangulation
    final = {}
    layer_provenance = {}
    for f in MAIN.glob("*.json"):
        if f.name == "summary.json": continue
        d = json.loads(f.read_text())
        final[d['name']] = d.get('verdict', '')
        layer_provenance[d['name']] = "tri"

    # Layer 2: UND-rerun (conf=0.10) — promote UND if rerun verdict is HIGH-CONF
    promoted_l2 = []
    for name, v in list(final.items()):
        if v.startswith("UNDECIDED"):
            rp = RERUN / f"{name}.json"
            if rp.exists():
                rv = json.loads(rp.read_text()).get('verdict','')
                if is_high_conf(rv):
                    final[name] = rv
                    layer_provenance[name] = "und-rerun"
                    promoted_l2.append((name, gt_map[name], rv))

    # Layer 3: Per-camera ensemble — promote STILL-UND if STRONG single-camera
    # signal (no weak signals; no overrides on decided).
    promoted_l3 = []
    for name, v in list(final.items()):
        if not v.startswith("UNDECIDED"): continue
        e = ENS_MAP.get(name)
        if not e: continue
        fr, nr = e['fr'], e['nr']
        # STRONG = exact "MAKE" or "MISS" (no -weak suffix)
        new_v = None; why = None
        if fr == "MAKE" and nr != "MISS":
            new_v = "MAKE"; why = f"FR strong MAKE, NR={nr}"
        elif nr == "MAKE" and fr != "MISS":
            new_v = "MAKE"; why = f"NR strong MAKE, FR={fr}"
        elif fr == "MISS" and nr != "MAKE":
            new_v = "MISS"; why = f"FR strong MISS, NR={nr}"
        elif nr == "MISS" and fr != "MAKE":
            new_v = "MISS"; why = f"NR strong MISS, FR={fr}"
        if new_v:
            final[name] = f"{new_v} (ENSEMBLE: {why})"
            layer_provenance[name] = "ensemble"
            promoted_l3.append((name, gt_map[name], new_v, why))

    # Tally
    by_class = defaultdict(lambda: defaultdict(int))
    roll = defaultdict(int)
    for name, v in final.items():
        gt = gt_map[name]; cat = classify(gt, v)
        by_class[gt][cat] += 1; roll[cat] += 1

    print(f"=== L2 promotions (UND-rerun): {len(promoted_l2)} ===")
    for name, gt, v in promoted_l2:
        c = classify(gt, v)
        print(f"  {name:14s} {gt:18s} -> {v[:70]:70s} [{c}]")

    print(f"\n=== L3 promotions (per-camera ensemble): {len(promoted_l3)} ===")
    for name, gt, v, why in promoted_l3:
        c = classify(gt, v)
        print(f"  {name:14s} {gt:18s} -> {v:6s} ({why:35s}) [{c}]")

    print(f"\n=== TIERED FINAL ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]; n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp+tn+fp+fn; acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-"*60)
    tp,tn,fp,fn,und = (roll['TP'], roll['TN'], roll['FP'], roll['FN'], roll['UND'])
    n = tp+tn+fp+fn+und; decided = tp+tn+fp+fn
    acc = 100*(tp+tn)/decided if decided else 0
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")


if __name__ == "__main__":
    main()
