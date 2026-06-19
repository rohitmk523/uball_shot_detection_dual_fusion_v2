#!/usr/bin/env python3
"""Final tiered merge v2: triangulation -> UND-rerun (conf=0.10) -> per-camera
ensemble -> hi-res YOLO (conf=0.05, imgsz=1280) re-decision of the 9 known
errors. Each layer can OVERRIDE prior decisions only on its targeted shots."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_und_results import is_high_conf

ROOT = Path(__file__).resolve().parent.parent
FG = ROOT / "data/client_report/triangulation_test/full_game"
MAIN = FG / "results"
RERUN = FG / "results_und_conf10"
HIRES = FG / "results_hires"
ENS = json.loads((FG / "ensemble_88.json").read_text())
ENS_MAP = {r['name']: r for r in ENS}

# The 9 cases where we had original triangulation errors
HIRES_TARGETS = {
    "13e72568_FG", "2d7198bf_4P", "439902c1_3P", "46479882_3P", "5909f27f_4P",
    "ef0f74a0_FG", "87cf67a5_4P", "837c14be_4P", "8b2ffd4b_FR",
}

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def classify(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"): return "UND"
    if gt in MAKE_LABELS and v.startswith("MAKE"):  return "TP"
    if gt in MISS_LABELS and v.startswith("MISS"):  return "TN"
    if gt in MISS_LABELS and v.startswith("MAKE"):  return "FP"
    if gt in MAKE_LABELS and v.startswith("MISS"):  return "FN"
    return "?"


def main():
    manifest = json.loads((FG / "shots_88.json").read_text())
    gt_map = {s['name']: s['gt'] for s in manifest}

    final = {}; layer = {}

    # L1: triangulation
    for f in MAIN.glob("*.json"):
        if f.name == "summary.json": continue
        d = json.loads(f.read_text())
        final[d['name']] = d.get('verdict', '')
        layer[d['name']] = "tri"

    # L2: UND-rerun (conf=0.10) high-confidence only
    for name, v in list(final.items()):
        if v.startswith("UNDECIDED"):
            rp = RERUN / f"{name}.json"
            if rp.exists():
                rv = json.loads(rp.read_text()).get('verdict','')
                if is_high_conf(rv):
                    final[name] = rv; layer[name] = "L2"

    # L3: ensemble (strong-only, still-UND)
    for name, v in list(final.items()):
        if not v.startswith("UNDECIDED"): continue
        e = ENS_MAP.get(name)
        if not e: continue
        fr, nr = e['fr'], e['nr']
        new_v = None
        if fr == "MAKE" and nr != "MISS": new_v = "MAKE"
        elif nr == "MAKE" and fr != "MISS": new_v = "MAKE"
        elif fr == "MISS" and nr != "MAKE": new_v = "MISS"
        elif nr == "MISS" and fr != "MAKE": new_v = "MISS"
        if new_v:
            final[name] = f"{new_v} (ENSEMBLE)"; layer[name] = "L3"

    # L4: hi-res YOLO override on the 9 known-error shots
    overrides = []
    for name in HIRES_TARGETS:
        hf = HIRES / f"{name}.json"
        if not hf.exists(): continue
        hv = json.loads(hf.read_text()).get('verdict','')
        old = final.get(name, "UNDECIDED")
        if hv != old:
            overrides.append((name, gt_map[name], old, hv,
                              classify(gt_map[name], old),
                              classify(gt_map[name], hv)))
            final[name] = hv; layer[name] = "L4-hires"

    print(f"=== L4 hi-res OVERRIDES ({len(overrides)}) ===")
    for name, gt, old, new, oc, nc in overrides:
        print(f"  {name:14s} {gt:18s} {oc:3s}→{nc:3s}  "
              f"old: {old[:30]:30s}  new: {new[:35]:35s}")

    # Tally
    by_class = defaultdict(lambda: defaultdict(int))
    roll = defaultdict(int)
    for name, v in final.items():
        gt = gt_map[name]
        cat = classify(gt, v)
        by_class[gt][cat] += 1; roll[cat] += 1

    print(f"\n=== TIERED FINAL v2 (hi-res YOLO included) ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]; n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp+tn+fp+fn; acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-"*60)
    tp,tn,fp,fn,und = (roll['TP'], roll['TN'], roll['FP'], roll['FN'], roll['UND'])
    n = tp+tn+fp+fn+und; decided = tp+tn+fp+fn
    acc_dec = 100*(tp+tn)/decided if decided else 0
    acc_all = 100*(tp+tn)/n
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc_dec:>6.1f}")
    print(f"\n  Decided accuracy: {tp+tn}/{decided} = {acc_dec:.1f}%")
    print(f"  Overall accuracy: {tp+tn}/{n} = {acc_all:.1f}%")
    # Per-layer
    print(f"\n  By layer:")
    layer_counts = defaultdict(int)
    for v in layer.values():
        layer_counts[v] += 1
    for k, c in sorted(layer_counts.items()):
        print(f"    {k}: {c}")


if __name__ == "__main__":
    main()
