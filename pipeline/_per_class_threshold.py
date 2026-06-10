#!/usr/bin/env python3
"""Find the optimal MAKE_R_CM per shot class.

For each class (FG, 3PT, 4PT, FT), iterate over thresholds and report
TP/TN/FP/FN counts + accuracy. The class-specific optimums inform whether
a per-class threshold map would meaningfully outperform a single global
threshold.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rescore_descent import rescore_one  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
THRESHOLDS = tuple(range(15, 56, 2))

CLASS_MAP = {
    "FG":  ("FG_MAKE", "FG_MISS"),
    "3PT": ("3PT_MAKE", "3PT_MISS"),
    "4PT": ("4PT_MAKE", "4PT_MISS"),
    "FT":  ("FREE_THROW_MAKE", "FREE_THROW_MISS"),
}
MAKE_GT = {v[0] for v in CLASS_MAP.values()}
MISS_GT = {v[1] for v in CLASS_MAP.values()}


def cat(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if gt in MAKE_GT and v.startswith("MAKE"):
        return "TP"
    if gt in MISS_GT and v.startswith("MISS"):
        return "TN"
    if gt in MISS_GT and v.startswith("MAKE"):
        return "FP"
    if gt in MAKE_GT and v.startswith("MISS"):
        return "FN"
    return "?"


def rescore_with_thr(samples: list[dict], thr: int) -> str:
    if not samples or len(samples) < 3:
        return "UNDECIDED (no samples)"
    saved = os.environ.get("MAKE_R_CM")
    os.environ["MAKE_R_CM"] = str(thr)
    try:
        _, v = rescore_one(samples)
    finally:
        if saved is None:
            os.environ.pop("MAKE_R_CM", None)
        else:
            os.environ["MAKE_R_CM"] = saved
    return v


def merge(v_l1: str, v_hr: str) -> str:
    if v_hr and not v_hr.startswith('UNDECIDED'):
        return v_hr
    if v_l1 and not v_l1.startswith('UNDECIDED'):
        return v_l1
    return v_hr or v_l1 or "UNDECIDED"


# Load samples
shots: list[dict] = []
for gid in GAMES:
    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    src_shots = json.loads((G / "shots_right.json").read_text())
    for s in src_shots:
        l1 = G / f"results_sam3/{s['name']}.json"
        hr = G / f"results_hires_sam3/{s['name']}.json"
        shots.append(dict(
            name=s['name'], gt=s['gt'],
            l1_samples=json.loads(l1.read_text()).get('samples', []) if l1.exists() else [],
            hr_samples=json.loads(hr.read_text()).get('samples', []) if hr.exists() else [],
        ))

print(f"loaded {len(shots)} shots")
for cls_key, (make_gt, miss_gt) in CLASS_MAP.items():
    n_make = sum(1 for s in shots if s['gt'] == make_gt)
    n_miss = sum(1 for s in shots if s['gt'] == miss_gt)
    if n_make + n_miss == 0:
        continue
    print()
    print(f"=== Class {cls_key} (n_MAKE={n_make} n_MISS={n_miss}) ===")
    print(f"{'thr':>5s}  {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} {'UND':>3}  "
          f"{'dec_pct':>8} {'ovr_pct':>8}")
    best_dec = None
    for thr in THRESHOLDS:
        roll = defaultdict(int)
        for s in shots:
            if s['gt'] not in (make_gt, miss_gt):
                continue
            v_l1 = rescore_with_thr(s['l1_samples'], thr)
            v_hr = rescore_with_thr(s['hr_samples'], thr)
            v = merge(v_l1, v_hr)
            roll[cat(s['gt'], v)] += 1
        tp = roll['TP']; tn = roll['TN']
        fp = roll['FP']; fn = roll['FN']; und = roll['UND']
        n = tp + tn + fp + fn + und
        dec = tp + tn + fp + fn
        acc_d = 100 * (tp + tn) / dec if dec else 0
        acc_o = 100 * (tp + tn) / n if n else 0
        marker = ""
        if best_dec is None or acc_d > best_dec[1]:
            best_dec = (thr, acc_d, acc_o, dict(roll))
        print(f"  {thr:>3}cm  {tp:>3} {tn:>3} {fp:>3} {fn:>3} {und:>3}  "
              f"{acc_d:>7.1f}% {acc_o:>7.1f}%{marker}")
    print(f"  -> best DECIDED thr={best_dec[0]}cm "
          f"({best_dec[1]:.1f}% dec / {best_dec[2]:.1f}% ovr)")
