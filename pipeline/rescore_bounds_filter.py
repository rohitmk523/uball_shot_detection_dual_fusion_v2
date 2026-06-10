#!/usr/bin/env python3
"""Test: filter physically-impossible triangulated samples BEFORE the
descent verdict, on cached samples (no YOLO re-run).

Root cause found 2026-06-10: in the FT-arc region (ball high above the FT
line, far from rim) FR/NR rays are near-parallel, so a few samples
triangulate to physically impossible points (X up to 647 m). Those garbage
samples hijack apex selection (argmax z) -> apex_dxy > 1000 -> "CLEAN MISS"
for real makes, and break the post-apex walker (XY teleport) -> UNDECIDED.

Court-bounds filter (half-court world frame, generous margins):
    800 < X < 2600 cm   (court depth incl. behind-baseline margin)
      0 < Y < 1430 cm   (court width)
    -50 < z < 1100 cm   (floor .. above-backboard)

Compares confusion matrices for unfiltered vs filtered rescore on all 280
June shots, overall and per class. Prints every flipped shot.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rescore_descent import rescore_one  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}
CLS = {"FREE_THROW": "FT", "FG": "FG", "3PT": "3PT", "4PT": "4PT"}

X_LO, X_HI = 800.0, 2600.0
Y_LO, Y_HI = 0.0, 1430.0
Z_LO, Z_HI = -50.0, 1100.0


def in_bounds(s: dict) -> bool:
    x, y, z = s["X_cm"]
    return (X_LO < x < X_HI) and (Y_LO < y < Y_HI) and (Z_LO < z < Z_HI)


def cat(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if gt in MAKE and v.startswith("MAKE"):
        return "TP"
    if gt in MISS and v.startswith("MISS"):
        return "TN"
    if gt in MISS and v.startswith("MAKE"):
        return "FP"
    if gt in MAKE and v.startswith("MISS"):
        return "FN"
    return "?"


def verdict_for(samples: list[dict]) -> str:
    if len(samples) < 3:
        return "UNDECIDED (no samples)"
    _, v = rescore_one(samples)
    return v


def merge(v_l1: str, v_hr: str) -> str:
    if v_hr and not v_hr.startswith("UNDECIDED"):
        return v_hr
    if v_l1 and not v_l1.startswith("UNDECIDED"):
        return v_l1
    return v_hr or v_l1 or "UNDECIDED"


def gt_class(gt: str) -> str:
    for prefix, c in CLS.items():
        if gt.startswith(prefix):
            return c
    return "?"


def main() -> int:
    rows = []
    n_filtered_samples = 0
    n_total_samples = 0
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        for s in json.loads((G / "shots_right.json").read_text()):
            l1_p = G / f"results_sam3/{s['name']}.json"
            hr_p = G / f"results_hires_sam3/{s['name']}.json"
            l1 = (json.loads(l1_p.read_text()).get("samples", [])
                  if l1_p.exists() else [])
            hr = (json.loads(hr_p.read_text()).get("samples", [])
                  if hr_p.exists() else [])
            l1_f = [x for x in l1 if in_bounds(x)]
            hr_f = [x for x in hr if in_bounds(x)]
            n_total_samples += len(l1) + len(hr)
            n_filtered_samples += (len(l1) - len(l1_f)) + (len(hr) - len(hr_f))

            v_base = merge(verdict_for(l1), verdict_for(hr))
            v_filt = merge(verdict_for(l1_f), verdict_for(hr_f))
            rows.append(dict(gid=gid, name=s["name"], gt=s["gt"],
                             cls=gt_class(s["gt"]),
                             base=v_base, filt=v_filt,
                             base_c=cat(s["gt"], v_base),
                             filt_c=cat(s["gt"], v_filt)))

    pct = 100 * n_filtered_samples / max(n_total_samples, 1)
    print(f"samples dropped by bounds filter: {n_filtered_samples}"
          f"/{n_total_samples} ({pct:.1f}%)\n")

    # ---- flips ----
    GOOD, BAD = ("TP", "TN"), ("FP", "FN")
    n_up = n_down = 0
    print("[flipped shots]")
    for r in rows:
        if r["base_c"] == r["filt_c"]:
            continue
        up = r["filt_c"] in GOOD and r["base_c"] not in GOOD
        down = r["base_c"] in GOOD and r["filt_c"] not in GOOD
        n_up += up
        n_down += down
        marker = "+" if up else "-" if down else "."
        print(f"  {marker} {r['gid']} {r['name']:22s} {r['gt']:18s} "
              f"{r['base_c']:3s}->{r['filt_c']}")
        print(f"      base: {r['base'][:84]}")
        print(f"      filt: {r['filt'][:84]}")
    print(f"\n  improvements: {n_up}   regressions: {n_down}")

    # ---- rollups ----
    def rollup(key: str, subset=None) -> str:
        roll: dict[str, int] = defaultdict(int)
        for r in rows:
            if subset and r["cls"] != subset:
                continue
            roll[r[key]] += 1
        tp, tn = roll["TP"], roll["TN"]
        dec = tp + tn + roll["FP"] + roll["FN"]
        n = dec + roll["UND"]
        acc_d = 100 * (tp + tn) / dec if dec else 0
        acc_o = 100 * (tp + tn) / n if n else 0
        return (f"TP={tp:3d} TN={tn:3d} FP={roll['FP']:3d} FN={roll['FN']:3d} "
                f"UND={roll['UND']:3d}  dec={acc_d:5.1f}%  ovr={acc_o:5.1f}%")

    print("\n[overall]")
    print(f"  base: {rollup('base_c')}")
    print(f"  filt: {rollup('filt_c')}")
    for c in ("FT", "FG", "3PT", "4PT"):
        print(f"\n[{c}]")
        print(f"  base: {rollup('base_c', c)}")
        print(f"  filt: {rollup('filt_c', c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
