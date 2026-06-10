#!/usr/bin/env python3
"""Diagnose the FT false-negative cluster.

16/38 FT makes score MISS (threshold-insensitive 15->43cm) on the June games.
For each FT shot, re-run the descent verdict on cached samples and extract:
  - which rule fired (verdict string prefix)
  - rim-plane crossing offset (dx, dy) vs rim center, when a crossing exists
  - apex stats / sample counts

If the (dx, dy) offsets of FN shots cluster in one direction -> systematic
bias (sync error or calibration). If random -> detection quality on FT clips.
FT TP shots are printed too as the control group.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rescore_descent import rescore_one  # noqa: E402
from triangulate_shot import RIM_X, RIM_Y, RIM_Z  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
FT_GTS = ("FREE_THROW_MAKE", "FREE_THROW_MISS")


def merge(v_l1: str, v_hr: str, i_l1: dict, i_hr: dict) -> tuple[str, dict, str]:
    """Same precedence as _per_class_threshold: hi-res wins if decided."""
    if v_hr and not v_hr.startswith("UNDECIDED"):
        return v_hr, i_hr, "hr"
    if v_l1 and not v_l1.startswith("UNDECIDED"):
        return v_l1, i_l1, "l1"
    return (v_hr or v_l1 or "UNDECIDED"), (i_hr or i_l1 or {}), "und"


def shot_stats(samples: list[dict]) -> dict:
    if not samples:
        return dict(n=0)
    zs = [s["X_cm"][2] for s in samples]
    apex_idx = int(np.argmax(zs))
    ax, ay, az = samples[apex_idx]["X_cm"]
    return dict(
        n=len(samples),
        apex_z=az,
        apex_dxy=float(np.hypot(ax - RIM_X, ay - RIM_Y)),
        apex_idx=apex_idx,
        n_post_apex=len(samples) - apex_idx - 1,
    )


def main() -> int:
    rows = []
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        for s in json.loads((G / "shots_right.json").read_text()):
            if s["gt"] not in FT_GTS:
                continue
            l1_p = G / f"results_sam3/{s['name']}.json"
            hr_p = G / f"results_hires_sam3/{s['name']}.json"
            l1_samples = (json.loads(l1_p.read_text()).get("samples", [])
                          if l1_p.exists() else [])
            hr_samples = (json.loads(hr_p.read_text()).get("samples", [])
                          if hr_p.exists() else [])
            i_l1, v_l1 = (rescore_one(l1_samples) if len(l1_samples) >= 3
                          else ({}, "UNDECIDED (no samples)"))
            i_hr, v_hr = (rescore_one(hr_samples) if len(hr_samples) >= 3
                          else ({}, "UNDECIDED (no samples)"))
            v, info, src = merge(v_l1, v_hr, i_l1, i_hr)
            st = shot_stats(hr_samples if src == "hr" else l1_samples)
            rows.append(dict(gid=gid, name=s["name"], gt=s["gt"],
                             verdict=v, src=src, info=info or {}, stats=st))

    # ---- Classify ----
    def cat(r):
        gt_make = r["gt"] == "FREE_THROW_MAKE"
        v = r["verdict"]
        if v.startswith("UNDECIDED"):
            return "UND"
        if v.startswith("MAKE"):
            return "TP" if gt_make else "FP"
        return "FN" if gt_make else "TN"

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[cat(r)].append(r)
    print(f"FT shots: {len(rows)}  "
          + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_cat.items())))

    # ---- Detail per category ----
    for c in ("FN", "TP", "UND", "FP"):
        print(f"\n{'=' * 100}\n[{c}]  n={len(by_cat[c])}")
        for r in sorted(by_cat[c], key=lambda r: r["gid"]):
            st, info = r["stats"], r["info"]
            cross = info.get("cross_xy")
            cross_txt = ""
            if cross:
                dx, dy = cross[0] - RIM_X, cross[1] - RIM_Y
                cross_txt = f"  cross dx={dx:+6.1f} dy={dy:+6.1f} r={info.get('cross_r_cm', 0):5.1f}"
            print(f"  {r['gid']}  {r['name']:22s} [{r['src']}] "
                  f"n={st.get('n', 0):3d} post={st.get('n_post_apex', 0):3d} "
                  f"apex_z={st.get('apex_z', 0):4.0f} apex_r={st.get('apex_dxy', 0):4.0f}"
                  f"{cross_txt}")
            print(f"      -> {r['verdict'][:90]}")

    # ---- Bias analysis on crossings ----
    print(f"\n{'=' * 100}\n[crossing-offset bias check]")
    for c in ("FN", "TP"):
        dxs, dys = [], []
        for r in by_cat[c]:
            cross = r["info"].get("cross_xy")
            if cross:
                dxs.append(cross[0] - RIM_X)
                dys.append(cross[1] - RIM_Y)
        if dxs:
            print(f"  {c}: n_cross={len(dxs)}  "
                  f"dx mean={np.mean(dxs):+6.1f} med={np.median(dxs):+6.1f} sd={np.std(dxs):5.1f}   "
                  f"dy mean={np.mean(dys):+6.1f} med={np.median(dys):+6.1f} sd={np.std(dys):5.1f}")
        else:
            print(f"  {c}: no crossings recorded")

    # ---- FN verdict-type rollup ----
    print(f"\n[FN verdict types]")
    types = defaultdict(int)
    for r in by_cat["FN"]:
        v = r["verdict"]
        key = v.split(":")[0].split("(")[1] if "(" in v else v
        types[key.strip()] += 1
    for k, n in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
