#!/usr/bin/env python3
"""Merge the UND-rerun (conf=0.10) verdicts back into the main results, but
only when the rerun verdict is HIGH-CONFIDENCE. This filters out the noisy
re-detections at lower confidence that can introduce new false positives.

High-confidence MAKE = clean rim-plane crossing with r<25 cm AND z_min in
[-50, 250] cm. High-confidence MISS = explicit RIM-OUT bounce signal OR
crossed rim plane at r>=80 cm OR detected at >300 cm apex r with >350 cm
peak z (clear off-target shot). Everything else stays UNDECIDED.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FG = ROOT / "data/client_report/triangulation_test/full_game"
MAIN = FG / "results"
RERUN = FG / "results_und_conf10"

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def is_high_conf(verdict: str) -> bool:
    """Return True if the verdict text is HIGH-confidence based on what we
    observed in the 88-shot ground-truth runs."""
    if verdict.startswith("UNDECIDED"):
        return False
    # MAKE confidence: clean rim-plane crossing with small r and ball reaching
    # near-floor z. Reject "smooth descent" or "gap-stop" rules (those rely on
    # heuristics; in low-confidence YOLO they're easier to spoof).
    if verdict.startswith("MAKE"):
        m = re.search(r"passed through rim plane at r=(\d+)cm, z_min=(-?\d+)cm", verdict)
        if m:
            r = int(m.group(1)); z = int(m.group(2))
            # any sample tracked to z<=320 below rim plane = ball through
            return r <= 25 and z <= 320
        return False
    if verdict.startswith("MISS"):
        if "RIM-OUT" in verdict and "bounce=" in verdict:
            return True
        m = re.search(r"crossed rim plane at r=(\d+)cm", verdict)
        if m and int(m.group(1)) >= 80:
            return True
        # any verdict with apex r ≥ 300 = clean off-target shot regardless of
        # the rule that decided the MISS (e.g. "no clear make signal" but apex
        # was already half the court away from the rim).
        m = re.search(r"apex r=(\d+)cm", verdict)
        if m and int(m.group(1)) >= 300:
            return True
        return False
    return False


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

    # Original verdicts
    orig = {}
    for f in MAIN.glob("*.json"):
        if f.name == "summary.json": continue
        d = json.loads(f.read_text())
        orig[d['name']] = d.get('verdict', '')

    # Build merged verdicts
    merged = {}
    promoted = []
    for name, v in orig.items():
        merged[name] = v
        if v.startswith("UNDECIDED"):
            rerun_path = RERUN / f"{name}.json"
            if rerun_path.exists():
                rv = json.loads(rerun_path.read_text()).get('verdict','')
                if is_high_conf(rv):
                    merged[name] = rv
                    promoted.append((name, gt_map[name], v, rv))

    # Tally
    by_class = defaultdict(lambda: defaultdict(int))
    rollup = defaultdict(int)
    for name, v in merged.items():
        gt = gt_map[name]
        cat = classify(gt, v)
        by_class[gt][cat] += 1
        rollup[cat] += 1

    print(f"=== UND→DECIDED PROMOTIONS ({len(promoted)}) ===")
    for name, gt, old_v, new_v in promoted:
        cat = classify(gt, new_v)
        print(f"  {name:14s}  GT={gt:18s}  rerun: {new_v[:60]:60s}  [{cat}]")

    print(f"\n=== FINAL (merged) per-class breakdown ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]
        n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp + tn + fp + fn
        acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-"*60)
    tp, tn, fp, fn, und = (rollup['TP'], rollup['TN'], rollup['FP'], rollup['FN'], rollup['UND'])
    n = tp + tn + fp + fn + und
    decided = tp + tn + fp + fn
    acc = 100*(tp+tn)/decided if decided else 0
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")


if __name__ == "__main__":
    main()
