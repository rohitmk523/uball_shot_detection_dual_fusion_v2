#!/usr/bin/env python3
"""Read all per-shot JSON results and print a per-class confusion matrix."""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data/client_report/triangulation_test/full_game/results"

MAKE_LABELS = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}


def classify(gt: str, verdict: str) -> str:
    if verdict.startswith("UNDECIDED"):
        return "UND"
    if gt in MAKE_LABELS and verdict.startswith("MAKE"):  return "TP"
    if gt in MISS_LABELS and verdict.startswith("MISS"):  return "TN"
    if gt in MISS_LABELS and verdict.startswith("MAKE"):  return "FP"
    if gt in MAKE_LABELS and verdict.startswith("MISS"):  return "FN"
    return "?"


def main():
    files = sorted(RESULTS.glob("*.json"))
    files = [f for f in files if f.name != "summary.json"]
    print(f"loaded {len(files)} shot results from {RESULTS}")

    rows = []
    by_class = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    for f in files:
        d = json.loads(f.read_text())
        v = d.get('verdict') or ""
        gt = d.get('gt') or ""
        cat = classify(gt, v)
        by_class[gt][cat] += 1
        overall[cat] += 1
        rows.append((d['name'], gt, v, cat))

    # Per-class table
    print("\n=== per-class breakdown ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]
        n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp + tn + fp + fn
        acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-" * 60)
    tp, tn, fp, fn, und = (overall['TP'], overall['TN'],
                           overall['FP'], overall['FN'], overall['UND'])
    n = tp + tn + fp + fn + und
    decided = tp + tn + fp + fn
    acc = 100*(tp+tn)/decided if decided else 0
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")

    # Error cases
    print(f"\n=== FALSE POSITIVES (predicted MAKE, was MISS): {fp} ===")
    for name, gt, v, cat in rows:
        if cat == 'FP':
            print(f"  {name:18s} {gt:18s} -> {v[:90]}")
    print(f"\n=== FALSE NEGATIVES (predicted MISS, was MAKE): {fn} ===")
    for name, gt, v, cat in rows:
        if cat == 'FN':
            print(f"  {name:18s} {gt:18s} -> {v[:90]}")
    print(f"\n=== UNDECIDED: {und} ===")
    for name, gt, v, cat in rows:
        if cat == 'UND':
            print(f"  {name:18s} {gt:18s} -> {v[:90]}")


if __name__ == "__main__":
    sys.exit(main())
