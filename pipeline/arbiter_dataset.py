#!/usr/bin/env python3
"""Assemble the hybrid-arbiter dataset: fusion outputs x triangulation
outputs per shot, for the games present in BOTH systems.

Fusion side  : data/p3_fresh_predictions.parquet (label, pred, prob)
Triangulation: val_<gid8>/{shots_right.json, results_sam3, results_hires_sam3}
               — verdicts re-derived via rescore_one under the production
               flag set so the info dict (y_off_stop, cross_r_cm, ...) is
               available as features.

Output: data/client_report/triangulation_test/arbiter_dataset.json
One row per overlap shot:
  gid, play_id, name, gt_make, fus_make, fus_prob,
  tri_verdict, tri_rule, tri_cat,
  n_l1, n_hr, n_used, y_off_stop, cross_r, cross_y, z_min, bounce_cm,
  apex_r, apex_z, n_post

NOTE: run with the production env so verdicts match the validated runs:
  DESCENT_BOUNDS=1 DESCENT_SKIP_NOISY=1 DESCENT_MAX_SKIPS=8 DESCENT_Y_GUARD=1
(set automatically below).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
os.environ.update({"DESCENT_BOUNDS": "1", "DESCENT_SKIP_NOISY": "1",
                   "DESCENT_MAX_SKIPS": "8", "DESCENT_Y_GUARD": "1"})

import numpy as np  # noqa: E402
from rescore_descent import rescore_one  # noqa: E402
from triangulate_shot import RIM_X, RIM_Y  # noqa: E402

MAKE = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}


def tri_rule(v: str) -> str:
    for key, rule in (("gap-stop", "gap_stop"), ("rattled in", "rattle_make"),
                      ("passed through rim plane", "cross_make"),
                      ("lateral-centered crossing", "cross_y_override"),
                      ("centered crossing", "bounceback_override"),
                      ("smooth descent to", "smooth_make"),
                      ("RIM-OUT", "rim_out"), ("rim-bounce", "bounce_back"),
                      ("crossed rim plane", "cross_too_wide"),
                      ("rattle rejected", "y_reject_rattle"),
                      ("cross rejected", "y_reject_cross"),
                      ("smooth descent rejected", "y_reject_smooth"),
                      ("NR-rebound", "nr_rebound"),
                      ("no clear make signal", "rule6_default"),
                      ("CLEAN", "clean_far"),
                      ("UNDECIDED", "undecided")):
        if key in v:
            return rule
    return "other"


def shot_features(G: Path, name: str) -> dict:
    feats = dict(n_l1=0, n_hr=0, n_used=0, y_off_stop=-1.0, cross_r=-1.0,
                 cross_y=-1.0, z_min=-1.0, bounce_cm=-1.0, apex_r=-1.0,
                 apex_z=-1.0, n_post=-1, tri_verdict="UNDECIDED")
    srcs = {}
    for tag, d in (("l1", "results_sam3"), ("hr", "results_hires_sam3")):
        p = G / d / f"{name}.json"
        if p.exists():
            srcs[tag] = json.loads(p.read_text()).get("samples", [])
    feats["n_l1"] = len(srcs.get("l1", []))
    feats["n_hr"] = len(srcs.get("hr", []))
    # merge precedence identical to the pipeline: hr if decided, else l1
    chosen = None
    for tag in ("hr", "l1"):
        s = srcs.get(tag, [])
        if len(s) < 3:
            continue
        info, v = rescore_one(s)
        if not v.startswith("UNDECIDED"):
            chosen = (s, info, v)
            break
        if chosen is None:
            chosen = (s, info, v)
    if chosen is None:
        return feats
    samples, info, v = chosen
    feats["tri_verdict"] = v
    feats["n_used"] = len(samples)
    info = info or {}
    feats["y_off_stop"] = float(info.get("y_off_stop", -1.0))
    feats["cross_r"] = float(info.get("cross_r_cm", -1.0))
    cx = info.get("cross_xy")
    feats["cross_y"] = float(abs(cx[1] - RIM_Y)) if cx else -1.0
    feats["z_min"] = float(info.get("z_min", -1.0))
    feats["bounce_cm"] = float(info.get("bounce_cm", -1.0))
    feats["n_post"] = int(info.get("n_post", -1))
    zs = [s["X_cm"][2] for s in samples]
    ai = int(np.argmax(zs))
    ax, ay, az = samples[ai]["X_cm"]
    feats["apex_r"] = float(np.hypot(ax - RIM_X, ay - RIM_Y))
    feats["apex_z"] = float(az)
    return feats


def main() -> int:
    import pandas as pd  # only available in the conda python
    df = pd.read_parquet(ROOT / "data/p3_fresh_predictions.parquet")
    df["g8"] = df["game_id"].astype(str).str[:8]

    rows = []
    for g8 in sorted(df["g8"].unique()):
        G = ROOT / f"data/client_report/triangulation_test/val_{g8}"
        if not (G / "shots_right.json").exists():
            continue
        shots = {s["play_id"]: s for s in
                 json.loads((G / "shots_right.json").read_text())}
        sub = df[df["g8"] == g8]
        n = 0
        for _, r in sub.iterrows():
            pid = str(r["play_id"])
            if pid not in shots:
                continue
            s = shots[pid]
            feats = shot_features(G, s["name"])
            v = feats.pop("tri_verdict")
            rows.append(dict(
                gid=g8, play_id=pid, name=s["name"],
                gt_make=bool(s["gt"] in MAKE),
                gt_class=s["gt"],
                fus_make=bool(int(r["pred"]) == 1),
                fus_prob=float(r["prob"]),
                tri_verdict=v, tri_rule=tri_rule(v),
                tri_make=v.startswith("MAKE"),
                tri_und=v.startswith("UNDECIDED"),
                **feats))
            n += 1
        print(f"{g8}: {n} overlap shots")

    out = ROOT / "data/client_report/triangulation_test/arbiter_dataset.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"total {len(rows)} rows -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
