#!/usr/bin/env python3
"""Re-vote on existing ensemble_88.json (per-camera signals already cached)
using the updated ensemble_vote logic. No YOLO re-run needed."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from per_camera_verdict import ensemble_vote, MAKE_LABELS, MISS_LABELS


def main():
    src = Path("data/client_report/triangulation_test/full_game/ensemble_88.json")
    rows = json.loads(src.read_text())

    def cat(g, v):
        if v.startswith("UNDECIDED"): return "UND"
        if g in MAKE_LABELS and v.startswith("MAKE"):  return "TP"
        if g in MISS_LABELS and v.startswith("MISS"):  return "TN"
        if g in MISS_LABELS and v.startswith("MAKE"):  return "FP"
        if g in MAKE_LABELS and v.startswith("MISS"):  return "FN"
        return "?"

    by_class = defaultdict(lambda: defaultdict(int))
    roll = defaultdict(int)
    changes = []
    for r in rows:
        old_final = r['final']
        old_cat = r['after']
        new_final, reason = ensemble_vote(r['tri'], r['fr'], r['nr'])
        new_cat = cat(r['gt'], new_final)
        if old_cat != new_cat:
            changes.append((r['name'], r['gt'], old_cat, new_cat,
                            r['tri'][:30], r['fr'], r['nr']))
        by_class[r['gt']][new_cat] += 1
        roll[new_cat] += 1

    if changes:
        print(f"=== changes from previous ensemble run ({len(changes)}) ===")
        for n, g, oc, nc, t, fr, nr in changes:
            print(f"  {n:14s} GT={g:18s} {oc:3s}→{nc:3s}  "
                  f"tri={t:30s}  FR={fr:9s}  NR={nr:9s}")
    print(f"\n=== ENSEMBLE FINAL (re-voted) ===")
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
