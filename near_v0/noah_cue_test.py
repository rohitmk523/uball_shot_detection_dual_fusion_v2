#!/usr/bin/env python3
"""Does the old fusion's Noah depth cue (ball_w/rim_w at the rim crossing)
separate make/miss on OUR detector's boxes? Computed offline from the stride-1
spotter caches + GT. If it separates, it's a cheap post-hoc gate to catch the
depth-illusion false-makes (clip #04) on top of the CNN's 0.93.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/near_detector/spotter_cache"
GAMES = {
    "72c08cb7": "data/client_report/triangulation_test/june_72c08cb7/shots_right.json",
    "9eb51980": "data/client_report/triangulation_test/train_9eb51980/shots_right.json",
}


def main():
    makes, misses = [], []
    per_game = {}
    for g, mpath in GAMES.items():
        cache = json.loads((CACHE / f"{g}.json").read_text())
        rim = cache["rim"]
        fps = cache["fps"]
        cxr, cyr = (rim[0]+rim[2])/2, (rim[1]+rim[3])/2
        rimw = rim[2]-rim[0]
        rimh = rim[3]-rim[1]
        man = json.loads((REPO / mpath).read_text())
        shots = [{"t0": float(s["t_start"]), "t1": float(s["t_end"]),
                  "make": s["gt"].endswith("MAKE")} for s in man
                 if cache["t0"] <= float(s["t_start"]) <= cache["t1"]]
        gm = gs = 0
        for s in shots:
            # frames within the shot window, ball near rim center
            best = None
            for fr in cache["frames"]:
                t = fr["f"]/fps
                if not (s["t0"]-1.0 <= t <= s["t1"]+2.0):
                    continue
                for bb in fr["balls"]:
                    bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                    # at-cross: horizontally inside rim AND near rim level
                    if abs(bx-cxr) < 0.6*rimw and abs(by-cyr) < 0.7*rimh:
                        bw = bb[2]-bb[0]
                        d = abs(by-cyr)
                        if best is None or d < best[0]:
                            best = (d, bw/(rimw+1e-6))
            if best is None:
                continue
            ratio = best[1]
            (makes if s["make"] else misses).append(ratio)
            if s["make"]: gm += 1
            else: gs += 1
        per_game[g] = (gm, gs)

    makes, misses = np.array(makes), np.array(misses)
    print(f"at-cross ratio measured: {len(makes)} makes, {len(misses)} misses")
    print(f"  MAKE  median={np.median(makes):.3f} (old fusion saw 0.355)")
    print(f"  MISS  median={np.median(misses):.3f} (old fusion saw 0.286)")
    # AUC (does bigger ratio -> make?)
    allv = np.concatenate([makes, misses])
    lab = np.concatenate([np.ones(len(makes)), np.zeros(len(misses))])
    order = np.argsort(allv)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(allv))
    n1 = lab.sum(); n0 = len(lab)-n1
    auc = (ranks[lab == 1].sum() - n1*(n1-1)/2) / (n1*n0)
    print(f"  single-feature AUC (ratio->make) = {auc:.3f} (old fusion saw ~0.60)")
    # how well does a low-ratio flag catch misses? (gate use-case)
    for thr in [0.28, 0.30, 0.32]:
        miss_caught = (misses < thr).mean()
        make_lost = (makes < thr).mean()
        print(f"  gate ratio<{thr}: flags {miss_caught:.0%} of misses, "
              f"would wrongly flag {make_lost:.0%} of makes")


if __name__ == "__main__":
    main()
