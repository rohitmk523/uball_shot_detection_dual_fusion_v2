#!/usr/bin/env python3
"""PRODUCTION HYBRID ROUTER — fusion + triangulation, validated 2026-06-11.

Decision policy (frozen; dev 95.55%, frozen-test 92.80% vs fusion alone
94.21% / 89.83%):

    1. fusion decides every shot (angle-aware HGB, real-time).
    2. ESCALATE only if fusion says MAKE with prob < 0.99  (~10% of shots):
         - triangulate the shot clip at hi-res (imgsz=1280, conf=0.05)
           with the production flag set
           (DESCENT_BOUNDS/SKIP_NOISY/MAX_SKIPS=8/Y_GUARD)
         - per-game sync from clip-audio cluster (clip_audio_sync.py),
           per-game SAM3 calibration (calibrate_june_sam3.py)
    3. VETO (flip MAKE -> MISS) only if the hi-res trajectory verdict is a
       confident MISS (rim-out bounce / lateral-off-center / passed-wide /
       rim-bounce-back / NR-rebound — i.e. NOT rule6-default, NOT UND).
    4. Everything else keeps fusion's verdict. fusion-MISS is never touched
       (99.6% correct).

Notes:
  - Trained stacker (HGB over tri features, 221 dev rows) was TESTED and
    LOST to these rules (test 91.53% vs 92.80%) — revisit only after the
    training-data expansion (23 fusion-era games) lands.
  - Audio rim-clang cue deferred (Rohit 2026-06-11).

This module provides the offline adjudication given fusion predictions and
triangulation results; the escalation orchestration reuses the existing
per-game chain (extract_val_clips -> clip_audio_sync -> calibrate_june_sam3
-> triangulate_shot).

Usage (offline eval shape):
  python pipeline/hybrid_router.py --fusion-csv <preds.csv> \
      --tri-root data/client_report/triangulation_test --prefix test_
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROB_GATE = 0.99
CONFIDENT_MISS = ("MISS (RIM-OUT", "MISS (rim-bounce", "MISS (crossed rim plane",
                  "MISS (rattle rejected", "MISS (cross rejected",
                  "MISS (smooth descent rejected", "MISS (NR-rebound",
                  "MISS (CLEAN")


def tri_verdict_for(G: Path, name: str) -> str:
    """Hi-res-first verdict (the v2 escalation policy)."""
    for d in ("results_hires_arbiter", "results_hires_sam3", "results_sam3"):
        p = G / d / f"{name}.json"
        if p.exists():
            v = json.loads(p.read_text()).get("verdict", "")
            if v and not v.startswith("UNDECIDED"):
                return v
    return "UNDECIDED"


def adjudicate(fus_make: bool, fus_prob: float, tri_verdict: str) -> bool:
    """Return final make verdict under the frozen policy."""
    if not fus_make:
        return False                       # never touch fusion-MISS
    if fus_prob >= PROB_GATE:
        return True                        # fusion certain -> keep MAKE
    if any(tri_verdict.startswith(c) for c in CONFIDENT_MISS):
        return False                       # confident trajectory veto
    return True                            # tri silent/ambiguous -> fusion


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-csv", required=True,
                    help="per-shot fusion preds (play_id, model_call/pred, prob)")
    ap.add_argument("--tri-root", default="data/client_report/triangulation_test")
    ap.add_argument("--prefix", default="test_")
    args = ap.parse_args()
    fus = pd.read_csv(args.fusion_csv)
    fus["g8"] = fus["game_id"].astype(str).str[:8]
    n = ok = base_ok = 0
    for g8, sub in fus.groupby("g8"):
        G = ROOT / args.tri_root / f"{args.prefix}{g8}"
        mp = G / "shots_right.json"
        if not mp.exists():
            continue
        shots = {s["play_id"]: s for s in json.loads(mp.read_text())}
        for _, r in sub.iterrows():
            pid = str(r["play_id"])
            if pid not in shots:
                continue
            gt = (r.get("gt_outcome") or r.get("label")) in ("MAKE", 1)
            fm = (r.get("model_call") or r.get("pred")) in ("MAKE", 1)
            tv = tri_verdict_for(G, shots[pid]["name"])
            final = adjudicate(fm, float(r["prob"]), tv)
            n += 1
            ok += (final == gt)
            base_ok += (fm == gt)
    print(f"n={n}  fusion {100*base_ok/n:.2f}%  hybrid {100*ok/n:.2f}%")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
