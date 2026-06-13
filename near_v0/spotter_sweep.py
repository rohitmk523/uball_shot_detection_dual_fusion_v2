#!/usr/bin/env python3
"""Sweep spotter configs over cached detector detections (offline, instant).
Reports recall / precision / false-per-min per config, averaged across the
tuning games. Event TIME stays rim-crossing (locked from v0.1); we tune
DETECTION density + precision filters around it.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/near_detector/spotter_cache"

# tuning games (held out from the 4 frozen FINAL test games)
TUNE = {
    "72c08cb7": "data/client_report/triangulation_test/june_72c08cb7/shots_right.json",
    "9eb51980": "data/client_report/triangulation_test/train_9eb51980/shots_right.json",
}


def in_zone(bb, rim, zone_above):
    bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    x1, y1, x2, y2 = rim
    h, w = y2 - y1, x2 - x1
    return (x1 - 0.2 * w <= bx <= x2 + 0.2 * w) and (y1 - zone_above * h <= by <= y2)


def spot(cache, gt, stride, zone_above, gap_s, min_frames, reach_frac, downward):
    rim = cache["rim"]
    fps = cache["fps"]
    f0 = int(cache["t0"] * fps)
    cxr, cyr = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
    rimw = rim[2] - rim[0]
    hits = []
    for fr in cache["frames"]:
        if ((fr["f"] - f0) // cache["stride"]) % stride != 0:
            continue
        inz = [bb for bb in fr["balls"] if in_zone(bb, rim, zone_above)]
        if not inz:
            continue
        best = min(inz, key=lambda bb: ((bb[0]+bb[2])/2-cxr)**2 + ((bb[1]+bb[3])/2-cyr)**2)
        by = (best[1] + best[3]) / 2
        dist = (((best[0]+best[2])/2-cxr)**2 + (by-cyr)**2) ** 0.5
        hits.append((fr["f"] / fps, fr["f"], dist, by))
    # cluster
    events = []
    for h in hits:
        if not events or h[0] - events[-1][-1][0] > gap_s:
            events.append([h])
        else:
            events[-1].append(h)
    events = [e for e in events if len(e) >= min_frames]
    ev = []
    for e in events:
        mind = min(x[2] for x in e)
        if reach_frac is not None and mind > reach_frac * rimw:
            continue
        if downward:                      # ball descends: y rises into the crossing
            cidx = min(range(len(e)), key=lambda i: e[i][2])
            pre = [x[3] for x in e[:cidx + 1]]
            if len(pre) >= 2 and pre[-1] < pre[0]:   # ended higher than started -> not descending
                continue
        ev.append(min(e, key=lambda x: x[2])[0])
    # match
    matched_gt = set()
    matched_ev = 0
    for et in ev:
        hit = False
        for gi, g in enumerate(gt):
            if g["t0"] - 2.0 <= et <= g["t1"] + 2.5:
                matched_gt.add(gi); hit = True
        matched_ev += hit
    recall = len(matched_gt) / max(1, len(gt))
    prec = matched_ev / max(1, len(ev))
    dur_min = (cache["t1"] - cache["t0"]) / 60 if cache["t1"] < 1e8 else 20
    false_pm = (len(ev) - matched_ev) / max(dur_min, 1)
    return recall, prec, false_pm, len(ev)


def main():
    caches, gts = {}, {}
    for g, mpath in TUNE.items():
        cp = CACHE / f"{g}.json"
        if not cp.exists():
            print(f"missing cache {cp}"); return
        caches[g] = json.loads(cp.read_text())
        man = json.loads((REPO / mpath).read_text())
        t0, t1 = caches[g]["t0"], caches[g]["t1"]
        gts[g] = [{"t0": float(s["t_start"]), "t1": float(s["t_end"])}
                  for s in man if t0 <= float(s["t_start"]) <= (t1 if t1 < 1e8 else 1e9)]

    grid = {
        "stride": [1, 2, 3],
        "zone_above": [1.2, 1.8, 2.5],
        "gap_s": [2.0],
        "min_frames": [1, 2],
        "reach_frac": [None, 0.6, 0.8],
        "downward": [False, True],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        rs, ps, fs = [], [], []
        for g in caches:
            r, p, f, n = spot(caches[g], gts[g], **cfg)
            rs.append(r); ps.append(p); fs.append(f)
        results.append((sum(rs)/len(rs), sum(ps)/len(ps), sum(fs)/len(fs), cfg))
    # rank: prioritise recall, then precision, penalise false/min
    results.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    print(f"swept {len(results)} configs on {list(caches)} "
          f"(GT: {[len(gts[g]) for g in caches]} shots)\n")
    print(f"{'recall':>7} {'prec':>6} {'false/m':>8}  config")
    for r, p, f, cfg in results[:12]:
        print(f"{r:7.3f} {p:6.3f} {f:8.2f}  {cfg}")
    # also show the v0.1 baseline config explicitly
    base = dict(stride=3, zone_above=1.2, gap_s=2.0, min_frames=2,
                reach_frac=None, downward=False)
    rs = [spot(caches[g], gts[g], **base) for g in caches]
    print(f"\nv0.1 baseline {base}:")
    print(f"  recall={sum(x[0] for x in rs)/len(rs):.3f} "
          f"prec={sum(x[1] for x in rs)/len(rs):.3f} "
          f"false/m={sum(x[2] for x in rs)/len(rs):.2f}")


if __name__ == "__main__":
    main()
