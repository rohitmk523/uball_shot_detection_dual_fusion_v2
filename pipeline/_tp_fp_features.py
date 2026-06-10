#!/usr/bin/env python3
"""TP-vs-FP feature distributions for guard design (June games).

For every shot the pipeline decides MAKE under bounds+skip, compute candidate
discriminative features from the RAW cached samples and compare the TP (real
makes) vs FP (real misses) populations. Any guard threshold must separate
these two populations — this is the ROC grounding for the synthesis agents'
proposals.

Features per decided-MAKE shot (computed on the verdict-winning source):
  reappear_rim_h : after stop_t, count of in-bounds raw samples with
                   z in (250, 420) and r > 60 within 1.2 s (rim-out rebound
                   reappearance signature; ball through net falls BELOW rim)
  reappear_any   : same but any r, z > rim_z (ball back above rim plane)
  y_off_stop     : |y - RIM_Y| of last walked sample
  n_src          : raw sample count of winning source (sparsity)
  gap_raw_n      : for gap-stop verdicts, # of raw in-bounds samples whose t
                   falls inside (stop_t, stop_t + 0.30] (manufactured gap if >0)
  z_min_walk     : min z of walked samples (makes should descend deep)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
os.environ["DESCENT_BOUNDS"] = "1"
os.environ["DESCENT_SKIP_NOISY"] = "1"
os.environ["DESCENT_MAX_SKIPS"] = "8"

from rescore_descent import rescore_one  # noqa: E402
from triangulate_shot import RIM_X, RIM_Y, RIM_Z, sample_in_court  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}


def features(samples_raw: list[dict], info: dict, verdict: str) -> dict:
    stop_t = info.get("stop_t")
    feats = dict(n_src=len(samples_raw),
                 y_off_stop=-1.0, reappear_rim_h=0, reappear_any=0,
                 gap_raw_n=-1, z_min_walk=info.get("z_min", -1.0),
                 stop_t=stop_t if stop_t is not None else -1.0)
    if stop_t is None:
        return feats
    sx, sy, sz = info.get("stop_xyz", (0, 0, 0))
    feats["y_off_stop"] = abs(sy - RIM_Y)
    post_raw = [s for s in samples_raw
                if s["t"] > stop_t + 1e-9 and sample_in_court(s)]
    for s in post_raw:
        x, y, z = s["X_cm"]
        if s["t"] - stop_t > 1.2:
            continue
        r = float(np.hypot(x - RIM_X, y - RIM_Y))
        if 250 < z < 420 and r > 60:
            feats["reappear_rim_h"] += 1
        if z > RIM_Z:
            feats["reappear_any"] += 1
    if verdict.startswith("MAKE (gap-stop"):
        feats["gap_raw_n"] = sum(
            1 for s in samples_raw
            if stop_t < s["t"] <= stop_t + 0.30 and sample_in_court(s))
    return feats


def main() -> int:
    rows = []
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        for s in json.loads((G / "shots_right.json").read_text()):
            l1p = G / f"results_sam3/{s['name']}.json"
            hrp = G / f"results_hires_sam3/{s['name']}.json"
            l1 = (json.loads(l1p.read_text()).get("samples", [])
                  if l1p.exists() else [])
            hr = (json.loads(hrp.read_text()).get("samples", [])
                  if hrp.exists() else [])
            i_l1, v_l1 = (rescore_one(l1) if len(l1) >= 3
                          else ({}, "UNDECIDED (no samples)"))
            i_hr, v_hr = (rescore_one(hr) if len(hr) >= 3
                          else ({}, "UNDECIDED (no samples)"))
            if v_hr and not v_hr.startswith("UNDECIDED"):
                v, info, src = v_hr, i_hr, hr
            elif v_l1 and not v_l1.startswith("UNDECIDED"):
                v, info, src = v_l1, i_l1, l1
            else:
                continue
            if not v.startswith("MAKE"):
                continue
            lab = ("TP" if s["gt"] in MAKE else
                   "FP" if s["gt"] in MISS else "?")
            rule = ("gap-stop" if "gap-stop" in v else
                    "rattled" if "rattled" in v else
                    "smooth" if "smooth descent" in v else "cross")
            rows.append(dict(gid=gid, name=s["name"], lab=lab, rule=rule,
                             **features(src, info or {}, v)))

    out = ROOT / "data/client_report/triangulation_test/tp_fp_features.json"
    out.write_text(json.dumps(rows, indent=1))

    def show(feat: str, fmt: str = "6.1f") -> None:
        for lab in ("TP", "FP"):
            vals = [r[feat] for r in rows if r["lab"] == lab and r[feat] >= 0]
            if not vals:
                continue
            q = np.percentile(vals, [10, 50, 90])
            frac_pos = np.mean([v > 0 for v in vals])
            print(f"  {lab} {feat:14s} n={len(vals):3d}  "
                  f"p10={q[0]:{fmt}} med={q[1]:{fmt}} p90={q[2]:{fmt}}  "
                  f">0: {100*frac_pos:4.0f}%")

    n_tp = sum(1 for r in rows if r["lab"] == "TP")
    n_fp = sum(1 for r in rows if r["lab"] == "FP")
    print(f"decided MAKE shots: TP={n_tp} FP={n_fp}\n")
    for f in ("reappear_rim_h", "reappear_any", "y_off_stop",
              "n_src", "gap_raw_n", "z_min_walk"):
        show(f)
        print()

    # rule-stratified reappearance (the headline guard candidate)
    print("[reappear_rim_h > 0 rate by rule]")
    for rule in ("gap-stop", "rattled", "smooth", "cross"):
        for lab in ("TP", "FP"):
            sub = [r for r in rows if r["lab"] == lab and r["rule"] == rule]
            if not sub:
                continue
            pos = sum(1 for r in sub if r["reappear_rim_h"] > 0)
            print(f"  {rule:9s} {lab}: {pos}/{len(sub)}")
    print(f"\nrows -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
