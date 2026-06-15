#!/usr/bin/env python3
"""Noah-method port test: fit a parabola to the ball's APPROACH (where it IS
detected) and EXTRAPOLATE the rim-plane crossing -- so we don't need the ball
detected AT the crossing (which failed: through-passage fired on only 31% of
makes). Measures: (a) does extrapolation 'fire' on more makes than 31%? (b) does
the extrapolated crossing (L-R over-hoop) + ball-size depth separate make/miss?
Compared to through-passage 0.544, ball-size 0.64, CNN 0.93.
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
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def shot_track(frames, fps, rim, t0, t1):
    cxr, cyr = (rim[0]+rim[2])/2, (rim[1]+rim[3])/2
    rimw, rimh = rim[2]-rim[0], rim[3]-rim[1]
    pts = []   # (t, cx, cy, bw) ball nearest rim center per frame
    for fr in frames:
        t = fr["f"]/fps
        if not (t0-1.5 <= t <= t1+2.0):
            continue
        best = None
        for bb in fr["balls"]:
            bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            # only consider balls roughly over/above the hoop column
            if abs(bx-cxr) > 2.0*rimw or by > cyr + 2.0*rimh:
                continue
            d = (bx-cxr)**2 + (by-cyr)**2
            if best is None or d < best[0]:
                best = (d, t, bx, by, bb[2]-bb[0])
        if best:
            pts.append(best[1:])
    return np.array(pts), (cxr, cyr, rimw, rimh)


def features(pts, geom):
    cxr, cyr, rimw, rimh = geom
    out = dict(crossing=0, lr=np.nan, depth=np.nan, n=len(pts))
    if len(pts) < 5:
        return out
    t, cx, cy, bw = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    t = t - t[0]
    # fit cy(t) quadratic (gravity), cx(t) linear
    cyf = np.polyfit(t, cy, 2)
    cxf = np.polyfit(t, cx, 1)
    if cyf[0] <= 0:                      # must open downward in image (accel +y)
        return out
    # solve cy(t*) = cyr on the descending branch (dcy/dt>0)
    a, b, c = cyf; roots = np.roots([a, b, c - cyr])
    roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    desc = [r for r in roots if (2*a*r + b) > 0]
    if not desc:
        return out
    tstar = min(desc, key=lambda r: abs(r - t[-1]))   # nearest the observed end
    cx_star = np.polyval(cxf, tstar)
    out["crossing"] = 1
    out["lr"] = abs(cx_star - cxr) / (rimw/2 + 1e-6)   # 0=center, 1=rim edge
    # depth = ball size near the crossing (nearest detected frame)
    k = int(np.argmin(np.abs(t - tstar)))
    out["depth"] = bw[k] / (rimw + 1e-6)
    return out


def main():
    rows = []
    for g, mp in GAMES.items():
        cache = json.loads((CACHE / f"{g}.json").read_text())
        rim, fps = cache["rim"], cache["fps"]
        man = json.loads((REPO / mp).read_text())
        for s in man:
            t0, t1 = float(s["t_start"]), float(s["t_end"])
            if not (cache["t0"] <= t0 <= cache["t1"]):
                continue
            pts, geom = shot_track(cache["frames"], fps, rim, t0, t1)
            f = features(pts, geom)
            f["make"] = s["gt"].endswith("MAKE")
            rows.append(f)

    mk = [r for r in rows if r["make"]]; ms = [r for r in rows if not r["make"]]
    y = np.array([int(r["make"]) for r in rows])
    print(f"shots: {len(rows)} ({len(mk)} make / {len(ms)} miss)\n")
    print("=== (a) ROBUSTNESS: does parabola extrapolation FIRE on more makes? ===")
    print(f"  parabola crossing found on: {np.mean([r['crossing'] for r in mk]):.0%} "
          f"of MAKES, {np.mean([r['crossing'] for r in ms]):.0%} of MISSES")
    print(f"  (vs through-passage which fired on only 31% of makes)\n")

    print("=== (b) SEPARATION (make/miss) ===")
    # over-hoop signal: small lr offset -> ball crossed over the rim opening
    lr_score = [-(r["lr"] if not np.isnan(r["lr"]) else 2.0) for r in rows]
    depth_score = [r["depth"] for r in rows]
    print(f"  L-R over-hoop (parabola)  AUC={auc(lr_score, y):.3f}")
    print(f"  ball-size depth           AUC={auc(depth_score, y):.3f}")
    # combined: normalize + sum (only where crossing found)
    comb = []
    for r in rows:
        if r["crossing"] and not np.isnan(r["depth"]):
            comb.append((1.0 - min(r["lr"], 1.5)/1.5) + (r["depth"]/0.5))
        else:
            comb.append(np.nan)
    print(f"  combined (over-hoop+depth) AUC={auc(comb, y):.3f}")
    print(f"\n  reference: through-passage 0.544 | CNN 0.93")

    # simple rule accuracy: make if crossing found AND over hoop AND depth>=0.30
    pred = np.array([int(r["crossing"] and not np.isnan(r["lr"]) and r["lr"] < 1.0
                         and (r["depth"] or 0) >= 0.30) for r in rows])
    print(f"\n  rule(make = parabola-crosses-over-hoop & depth>=0.30): "
          f"acc={np.mean(pred==y):.3f} "
          f"(makes recalled {np.mean(pred[y==1]):.0%}, false {np.mean(pred[y==0]):.0%})")


if __name__ == "__main__":
    main()
