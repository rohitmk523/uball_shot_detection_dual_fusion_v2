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
from triangulate_shot import (  # noqa: E402
    RIM_X, RIM_Y, RIM_Z, nr_rebound_check, sample_in_court,
)

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
                 apex_z=-1.0, n_post=-1, tri_verdict="UNDECIDED",
                 # ---- tri-confidence features ----
                 src_agree_miss=False,    # l1 AND hr both decide MISS
                 src_conflict=False,      # l1/hr decide opposite verdicts
                 desc_n=0,                # in-court samples in descent window
                 desc_max_gap=-1.0,       # max time gap in descent window (s)
                 cross_dt=-1.0,           # bracket dt of rim-plane crossing
                 nr_rebound=False,        # NR pixel rebound corroboration
                 hr_available=False)
    srcs = {}
    for tag, dirs in (("l1", ("results_sam3",)),
                      ("hr", ("results_hires_arbiter", "results_hires_sam3"))):
        for d in dirs:
            p = G / d / f"{name}.json"
            if p.exists():
                srcs[tag] = json.loads(p.read_text()).get("samples", [])
                break
    feats["n_l1"] = len(srcs.get("l1", []))
    feats["n_hr"] = len(srcs.get("hr", []))
    feats["hr_available"] = feats["n_hr"] >= 3
    # verdicts of BOTH sources (cross-source agreement is a confidence cue)
    verdicts = {}
    chosen = None
    for tag in ("hr", "l1"):
        s = srcs.get(tag, [])
        if len(s) < 3:
            continue
        info, v = rescore_one(s)
        verdicts[tag] = v
        if chosen is None and not v.startswith("UNDECIDED"):
            chosen = (s, info, v)
    if chosen is None:
        for tag in ("hr", "l1"):
            s = srcs.get(tag, [])
            if len(s) >= 3:
                info, v = rescore_one(s)
                chosen = (s, info, v)
                break
    v_l1, v_hr = verdicts.get("l1", ""), verdicts.get("hr", "")
    if v_l1 and v_hr:
        l1_miss, hr_miss = v_l1.startswith("MISS"), v_hr.startswith("MISS")
        l1_make, hr_make = v_l1.startswith("MAKE"), v_hr.startswith("MAKE")
        feats["src_agree_miss"] = bool(l1_miss and hr_miss)
        feats["src_conflict"] = bool((l1_miss and hr_make)
                                     or (l1_make and hr_miss))
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
    court = [s for s in samples if sample_in_court(s)]
    zs = [s["X_cm"][2] for s in court] or [0]
    ai = int(np.argmax(zs))
    ax, ay, az = court[ai]["X_cm"] if court else (0, 0, 0)
    feats["apex_r"] = float(np.hypot(ax - RIM_X, ay - RIM_Y))
    feats["apex_z"] = float(az)
    if court:
        apex_t = court[ai]["t"]
        desc = [s for s in court if apex_t <= s["t"] <= apex_t + 1.2]
        feats["desc_n"] = len(desc)
        if len(desc) >= 2:
            ts = [s["t"] for s in desc]
            feats["desc_max_gap"] = float(max(b - a for a, b
                                              in zip(ts, ts[1:])))
        # crossing bracket dt: gap between the two samples the rim-plane
        # crossing was interpolated across (large = unreliable crossing)
        for i in range(ai, len(court) - 1):
            if court[i]["X_cm"][2] >= RIM_Z > court[i + 1]["X_cm"][2]:
                feats["cross_dt"] = float(court[i + 1]["t"] - court[i]["t"])
                break
        try:
            reb, *_ = nr_rebound_check(court, ai)
            feats["nr_rebound"] = bool(reb)
        except Exception:
            pass
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
