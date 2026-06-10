#!/usr/bin/env python3
"""A/B/C the June depth-degeneracy fixes on cached samples (280 shots).

  A: baseline                       (no env)
  B: DESCENT_BOUNDS=1               (court-bounds filter before apex)
  C: DESCENT_BOUNDS=1 + DESCENT_SKIP_NOISY=1  (B + skip-don't-break walker)

Prints per-variant confusion overall + per class, and B->C flips.
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
MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}
VARIANTS = {
    "A:base": {},
    "B:bounds": {"DESCENT_BOUNDS": "1"},
    "C:bounds+skip": {"DESCENT_BOUNDS": "1", "DESCENT_SKIP_NOISY": "1"},
}


def cat(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if gt in MAKE and v.startswith("MAKE"):
        return "TP"
    if gt in MISS and v.startswith("MISS"):
        return "TN"
    if gt in MISS and v.startswith("MAKE"):
        return "FP"
    return "FN"


def gt_class(gt: str) -> str:
    return ("FT" if gt.startswith("FREE_THROW") else
            "3PT" if gt.startswith("3PT") else
            "4PT" if gt.startswith("4PT") else "FG")


def verdict(samples: list[dict]) -> str:
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


def set_env(env: dict) -> None:
    for k in ("DESCENT_BOUNDS", "DESCENT_SKIP_NOISY"):
        os.environ.pop(k, None)
    os.environ.update(env)


def main() -> int:
    shots = []
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        for s in json.loads((G / "shots_right.json").read_text()):
            l1_p = G / f"results_sam3/{s['name']}.json"
            hr_p = G / f"results_hires_sam3/{s['name']}.json"
            shots.append(dict(
                gid=gid, name=s["name"], gt=s["gt"], cls=gt_class(s["gt"]),
                l1=(json.loads(l1_p.read_text()).get("samples", [])
                    if l1_p.exists() else []),
                hr=(json.loads(hr_p.read_text()).get("samples", [])
                    if hr_p.exists() else []),
            ))
    print(f"loaded {len(shots)} shots")

    results: dict[str, list[str]] = {}
    for vname, env in VARIANTS.items():
        set_env(env)
        results[vname] = [cat(s["gt"], merge(verdict(s["l1"]), verdict(s["hr"])))
                          for s in shots]
    set_env({})

    def rollup(cats: list[str], cls: str | None = None) -> str:
        roll: dict[str, int] = defaultdict(int)
        for s, c in zip(shots, cats):
            if cls and s["cls"] != cls:
                continue
            roll[c] += 1
        tp, tn = roll["TP"], roll["TN"]
        dec = tp + tn + roll["FP"] + roll["FN"]
        n = dec + roll["UND"]
        return (f"TP={tp:3d} TN={tn:3d} FP={roll['FP']:3d} FN={roll['FN']:3d} "
                f"UND={roll['UND']:3d}  dec={100*(tp+tn)/dec if dec else 0:5.1f}%"
                f"  ovr={100*(tp+tn)/n if n else 0:5.1f}%")

    for cls in (None, "FT", "FG", "3PT", "4PT"):
        print(f"\n[{cls or 'overall'}]")
        for vname in VARIANTS:
            print(f"  {vname:14s} {rollup(results[vname], cls)}")

    print("\n[B->C flips]")
    GOOD = ("TP", "TN")
    n_up = n_down = 0
    for s, b, c in zip(shots, results["B:bounds"], results["C:bounds+skip"]):
        if b == c:
            continue
        up = c in GOOD and b not in GOOD
        down = b in GOOD and c not in GOOD
        n_up += up
        n_down += down
        print(f"  {'+' if up else '-' if down else '.'} {s['gid']} "
              f"{s['name']:22s} {s['gt']:18s} {b}->{c}")
    print(f"  improvements: {n_up}  regressions: {n_down}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
