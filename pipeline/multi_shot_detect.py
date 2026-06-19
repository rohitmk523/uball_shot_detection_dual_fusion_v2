#!/usr/bin/env python3
"""Multi-shot detector.

Scans the triangulated samples for multiple parabolic arcs (shot attempts)
in the same Supabase play window, and reports a verdict per attempt.

A "shot apex" is a local maximum in z(t) where:
  - z >= 250 cm (real shot height, not a dribble)
  - separated from any other detected apex by >= 0.5 s in time
  - separated from any other detected apex by a z-dip of >= 50 cm
    (so the ball clearly descended between apexes — not a single noisy plateau)

For each detected apex, we slice the trajectory around it (-0.8 s before,
+1.5 s after) and run the existing descent_verdict logic.

The final report includes ALL attempts. Two output modes for the merged
verdict layer:
  - "first": match what most human annotators label (first attempt outcome)
  - "last":  match what actually scored (final outcome)
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from triangulate_shot import (   # noqa: E402
    descent_verdict, nr_rebound_check, RIM_X, RIM_Y, RIM_Z,
)

FG = ROOT / "data/client_report/triangulation_test/full_game"
MAIN_RESULTS = FG / "results"

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def find_shot_apexes(samples: list[dict],
                     min_apex_z: float = 300.0,
                     min_sep_s: float = 1.0,
                     min_dip_cm: float = 50.0,
                     min_apex_x: float = 1500.0,
                     max_apex_x: float = 2200.0,
                     min_rise_cm: float = 50.0,
                     min_fall_cm: float = 50.0,
                     window_frames: int = 25) -> list[int]:
    """Return indices of local maxima in z that look like real shot apexes.

    Constraints:
      * z >= min_apex_z (skips dribbles / ball-in-hands height)
      * apex must be near the rim X (the shooter aiming at this rim)
      * if two candidates are within min_sep_s seconds OR not separated
        by a dip of at least min_dip_cm, keep only the higher one.
    """
    if len(samples) < 6:
        return []
    ts = np.array([s['t'] for s in samples])
    zs = np.array([s['X_cm'][2] for s in samples])
    xs = np.array([s['X_cm'][0] for s in samples])

    # Smooth z to avoid spurious peaks (3-frame moving avg)
    k = 3
    pad = np.r_[zs[:k], zs, zs[-k:]]
    zs_sm = np.convolve(pad, np.ones(k)/k, mode='same')[k:-k]

    candidates: list[int] = []
    for i in range(1, len(zs_sm) - 1):
        if zs_sm[i] < min_apex_z: continue
        if not (min_apex_x <= xs[i] <= max_apex_x): continue
        if not (zs_sm[i] >= zs_sm[i-1] and zs_sm[i] >= zs_sm[i+1]):
            continue
        # Require sustained rise BEFORE this apex
        lo = max(0, i - window_frames)
        rise = zs_sm[i] - zs_sm[lo:i].min() if i > 0 else 0
        if rise < min_rise_cm: continue
        # Require sustained fall AFTER this apex
        hi = min(len(zs_sm), i + window_frames)
        fall = zs_sm[i] - zs_sm[i+1:hi].min() if i < len(zs_sm)-1 else 0
        if fall < min_fall_cm: continue
        candidates.append(i)

    if not candidates:
        return []

    # Sort by height (descending)
    candidates.sort(key=lambda i: -zs_sm[i])

    # Greedy selection: keep highest, drop any too-close-in-time OR not
    # separated by a sufficient dip.
    kept: list[int] = []
    for c in candidates:
        ok = True
        for k_idx in kept:
            if abs(ts[c] - ts[k_idx]) < min_sep_s:
                ok = False; break
            # check dip between the two
            lo, hi = sorted([c, k_idx])
            min_between = float(zs_sm[lo:hi+1].min())
            higher = max(zs_sm[c], zs_sm[k_idx])
            if higher - min_between < min_dip_cm:
                ok = False; break
        if ok:
            kept.append(c)

    return sorted(kept)   # in time order


def verdict_for_arc(samples: list[dict], apex_idx: int,
                    window_before: float = 0.8,
                    window_after:  float = 1.5) -> tuple[dict | None, str]:
    """Slice samples around a single apex and run the existing descent
    verdict logic on that arc."""
    if not samples:
        return None, "UNDECIDED (no samples)"
    apex_t = float(samples[apex_idx]['t'])
    arc_samples = [s for s in samples
                   if apex_t - window_before <= s['t'] <= apex_t + window_after]
    if len(arc_samples) < 5:
        return None, "UNDECIDED (arc window too few samples)"

    apex_x, apex_y, apex_z = samples[apex_idx]['X_cm']
    apex_dxy = float(np.hypot(apex_x - RIM_X, apex_y - RIM_Y))
    apex_idx_in_arc = min(range(len(arc_samples)),
                          key=lambda i: abs(arc_samples[i]['t'] - apex_t))

    descent_info, descent_v = descent_verdict(arc_samples, apex_idx_in_arc,
                                              apex_dxy, apex_z)
    # NR-rebound override (same logic as main pipeline)
    if descent_v.startswith("MAKE"):
        post_apex_z_min = min(
            (arc_samples[i]['X_cm'][2]
             for i in range(apex_idx_in_arc, min(apex_idx_in_arc+30, len(arc_samples)))),
            default=9999)
        rebound, max_cy, min_cy_after, dt_reb = nr_rebound_check(
            arc_samples, apex_idx_in_arc)
        if rebound and post_apex_z_min > 150:
            descent_v = (f"MISS (NR-rebound: cy {max_cy:.0f}→{min_cy_after:.0f}, "
                         f"Δ={max_cy-min_cy_after:.0f}px)")

    info = dict(apex_t=apex_t, apex_z=apex_z, apex_dxy=apex_dxy,
                arc_n=len(arc_samples), descent=descent_info or {})
    return info, descent_v


def analyze_shot(samples: list[dict]) -> dict:
    """Run multi-shot analysis on one shot's samples."""
    apex_indices = find_shot_apexes(samples)
    attempts = []
    for ai in apex_indices:
        info, v = verdict_for_arc(samples, ai)
        attempts.append(dict(apex_idx=ai, info=info, verdict=v))
    return dict(n_attempts=len(attempts), apex_indices=apex_indices,
                attempts=attempts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-sep shot names")
    ap.add_argument("--out", default=str(FG / "multi_shot_results.json"))
    args = ap.parse_args()

    manifest = json.loads((FG / "shots_88.json").read_text())
    gt_map = {s['name']: s['gt'] for s in manifest}
    sel = set(args.only.split(",")) if args.only else None

    rows = []
    for f in sorted(MAIN_RESULTS.glob("*.json")):
        if f.name == "summary.json": continue
        d = json.loads(f.read_text())
        name = d['name']
        if sel and name not in sel: continue
        samples = d.get('samples', [])
        analysis = analyze_shot(samples)
        gt = gt_map.get(name, "?")
        orig_v = d.get('verdict', '')
        first = analysis['attempts'][0] if analysis['attempts'] else None
        last  = analysis['attempts'][-1] if analysis['attempts'] else None
        first_v = first['verdict'] if first else "UNDECIDED"
        last_v  = last['verdict']  if last  else "UNDECIDED"
        rows.append(dict(name=name, gt=gt, n_attempts=analysis['n_attempts'],
                         original=orig_v, first=first_v, last=last_v,
                         apex_indices=analysis['apex_indices']))
        if analysis['n_attempts'] >= 2:
            print(f"  ★ MULTI-SHOT  {name:14s}  GT={gt:18s}  "
                  f"original={orig_v[:30]:30s}  first={first_v[:25]:25s}  "
                  f"last={last_v[:25]:25s}")

    Path(args.out).write_text(json.dumps(rows, indent=2))

    # Quick stats
    multi = [r for r in rows if r['n_attempts'] >= 2]
    print(f"\n  {len(multi)} of {len(rows)} shots had multiple shot-apexes")

    # Three accuracy views
    def cat(g, v):
        if v.startswith("UNDECIDED"): return "UND"
        if g in MAKE_LABELS and v.startswith("MAKE"): return "TP"
        if g in MISS_LABELS and v.startswith("MISS"): return "TN"
        if g in MISS_LABELS and v.startswith("MAKE"): return "FP"
        if g in MAKE_LABELS and v.startswith("MISS"): return "FN"
        return "?"

    for label, getter in [("ORIG verdict (current pipeline)",  "original"),
                          ("FIRST attempt (matches GT scope)", "first"),
                          ("LAST attempt (final outcome)",     "last")]:
        roll = defaultdict(int)
        for r in rows:
            roll[cat(r['gt'], r[getter])] += 1
        n = sum(roll.values()); dec = roll['TP']+roll['TN']+roll['FP']+roll['FN']
        acc = 100*(roll['TP']+roll['TN'])/dec if dec else 0
        print(f"\n  {label}:")
        print(f"    TP={roll['TP']}  TN={roll['TN']}  FP={roll['FP']}  "
              f"FN={roll['FN']}  UND={roll['UND']}  decided_acc={acc:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
