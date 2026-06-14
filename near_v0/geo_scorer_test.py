#!/usr/bin/env python3
"""Test the GEOMETRIC make/miss approach (Rohit's idea) on our detector boxes:
  - ball passes THROUGH the rim: above-rim -> inside -> below-rim (time-ordered)
  - dwell: # frames ball center inside the rim box
  - depth: ball_w/rim_w at the rim crossing
Measure how well each signal + a combined rule separates make/miss on the
stride-1 caches (per-frame ball+rim + GT), vs the CNN's 0.93. The crux is
whether 30fps per-frame ball recall is dense enough to see the through-passage.
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


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(scores))
    n1 = labels.sum(); n0 = len(labels) - n1
    return (ranks[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def shot_features(frames, fps, rim, t0, t1):
    cxr, cyr = (rim[0]+rim[2])/2, (rim[1]+rim[3])/2
    rimw, rimh = rim[2]-rim[0], rim[3]-rim[1]
    seq = []   # (t, dy_rel, inside_horiz, inside_box, ratio)
    for fr in frames:
        t = fr["f"]/fps
        if not (t0-1.0 <= t <= t1+2.0):
            continue
        # ball closest to rim center this frame
        best = None
        for bb in fr["balls"]:
            bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            d = (bx-cxr)**2 + (by-cyr)**2
            if best is None or d < best[0]:
                best = (d, bx, by, bb[2]-bb[0])
        if best is None:
            continue
        _, bx, by, bw = best
        dy = (by - cyr) / rimh                    # rim-heights below center
        ih = abs(bx - cxr) < 0.6 * rimw
        ibox = (rim[0] <= bx <= rim[2]) and (rim[1] <= by <= rim[3])
        seq.append((t, dy, ih, ibox, bw/(rimw+1e-6)))
    if not seq:
        return None
    seq.sort()
    dys = [s[1] for s in seq]
    n_inside = sum(s[3] for s in seq)
    # time-ordered through-passage: an above-frame (dy<-0.4) over the hoop,
    # then later a below-frame (dy>0.4) over the hoop
    above_t = [s[0] for s in seq if s[2] and s[1] < -0.4]
    below_t = [s[0] for s in seq if s[2] and s[1] > 0.4]
    passed = int(bool(above_t) and bool(below_t) and min(above_t) < max(below_t))
    went_below = int(bool(below_t))
    # depth ratio at the crossing (min |dy| frame, over hoop)
    over = [s for s in seq if s[2]]
    ratio = over[int(np.argmin([abs(s[1]) for s in over]))][4] if over else np.nan
    return dict(n_inside=n_inside, passed=passed, went_below=went_below,
                ratio=ratio, vspan=(max(dys)-min(dys)))


def main():
    feats, labels = [], []
    for g, mp in GAMES.items():
        cache = json.loads((CACHE / f"{g}.json").read_text())
        rim, fps = cache["rim"], cache["fps"]
        man = json.loads((REPO / mp).read_text())
        for s in man:
            t0, t1 = float(s["t_start"]), float(s["t_end"])
            if not (cache["t0"] <= t0 <= cache["t1"]):
                continue
            f = shot_features(cache["frames"], fps, rim, t0, t1)
            if f is None:
                continue
            f["make"] = s["gt"].endswith("MAKE")
            feats.append(f); labels.append(int(f["make"]))
    labels = np.array(labels)
    n = len(labels); nmk = labels.sum()
    print(f"shots with detections: {n} ({nmk} make / {n-nmk} miss)\n")

    # how often does each geometric signal even FIRE on makes vs misses?
    mk = [f for f in feats if f["make"]]; ms = [f for f in feats if not f["make"]]
    print("=== signal behaviour (the crux = does it fire when it should?) ===")
    print(f"passed-through (above->below) fires on: "
          f"{np.mean([f['passed'] for f in mk]):.0%} of MAKES, "
          f"{np.mean([f['passed'] for f in ms]):.0%} of MISSES")
    print(f"went-below-rim fires on:          "
          f"{np.mean([f['went_below'] for f in mk]):.0%} of MAKES, "
          f"{np.mean([f['went_below'] for f in ms]):.0%} of MISSES")
    print(f"mean frames-inside-rim-box:        "
          f"MAKE {np.mean([f['n_inside'] for f in mk]):.1f}, "
          f"MISS {np.mean([f['n_inside'] for f in ms]):.1f}")
    print()
    print("=== single-feature AUC (->make) ===")
    for key in ["passed", "went_below", "n_inside", "ratio", "vspan"]:
        print(f"  {key:12} AUC={auc([f[key] for f in feats], labels):.3f}")

    # simple geometric rule accuracy: make if passed-through
    print("\n=== rule accuracy vs CNN 0.93 ===")
    pred_passed = np.array([f["passed"] for f in feats])
    print(f"  rule 'make = passed-through':  acc={np.mean(pred_passed==labels):.3f} "
          f"(makes recalled {np.mean(pred_passed[labels==1]):.0%}, "
          f"false-makes {np.mean(pred_passed[labels==0]):.0%})")
    pred_below = np.array([f["went_below"] for f in feats])
    print(f"  rule 'make = went-below-rim':  acc={np.mean(pred_below==labels):.3f} "
          f"(makes recalled {np.mean(pred_below[labels==1]):.0%}, "
          f"false-makes {np.mean(pred_below[labels==0]):.0%})")


if __name__ == "__main__":
    main()
