#!/usr/bin/env python3
"""Sweep MAKE_R_CM across rescore_descent.py for each June game.

For each threshold T in a small grid, rescores the SAM3 + HIRES_SAM3 results
without writing back (so the originals stay intact), and reports per-game
+ aggregate accuracy. The script also reports HYBRID and ENSEMBLE-equivalent
tallies if available.

This identifies the global MAKE_R_CM sweet spot AND surfaces per-class
optimums to inform a future per-class threshold.

Usage:
  python pipeline/threshold_sweep.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
THRESHOLDS = (25, 28, 30, 32, 35, 38, 40, 42, 45, 50)


def rescore_dir(results_dir: Path, manifest: Path, make_r: int
                ) -> dict[str, str]:
    """Re-run rescore_descent.py with MAKE_R_CM=make_r and capture per-shot
    NEW verdict by parsing the script output. Doesn't write back."""
    env = {**os.environ, "MAKE_R_CM": str(make_r)}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "pipeline/rescore_descent.py"),
         "--results-dir", str(results_dir),
         "--manifest", str(manifest)],
        env=env, capture_output=True, text=True)
    # Parse the per-shot "new" line — it appears only for CHANGED verdicts.
    # We need full coverage so do our own parse from the json files instead.
    # rescore_descent prints OLD/NEW rolls at the end; that's our summary.
    return proc.stdout


def parse_rolls(stdout: str) -> tuple[dict, dict]:
    """Parse the OLD: and NEW: TP/TN/FP/FN/UND rolls printed by rescore."""
    rolls = dict(OLD=defaultdict(int), NEW=defaultdict(int))
    for line in stdout.splitlines():
        line = line.strip()
        for tag in ("OLD", "NEW"):
            if line.startswith(f"{tag}:"):
                parts = line.split()
                for tok in parts:
                    if "=" in tok:
                        k, v = tok.split("=")
                        if k in ("TP", "TN", "FP", "FN", "UND"):
                            try:
                                rolls[tag][k] = int(v)
                            except ValueError:
                                pass
    return dict(rolls['OLD']), dict(rolls['NEW'])


def tally(roll: dict) -> tuple[int, int, float, float]:
    tp = roll.get('TP', 0); tn = roll.get('TN', 0)
    fp = roll.get('FP', 0); fn = roll.get('FN', 0); und = roll.get('UND', 0)
    n = tp + tn + fp + fn + und
    dec = tp + tn + fp + fn
    acc_d = 100 * (tp + tn) / dec if dec else 0
    acc_o = 100 * (tp + tn) / n if n else 0
    return dec, n, acc_d, acc_o


def main() -> int:
    print(f"{'threshold':>10s}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'UND':>4}  "
          f"{'dec_pct':>8} {'ovr_pct':>8}")
    print("-" * 70)

    by_threshold: dict[int, dict] = {}
    for thr in THRESHOLDS:
        agg = defaultdict(int)
        for gid in GAMES:
            G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
            # Rescore L1 (results_sam3) then hi-res (results_hires_sam3),
            # and combine per-shot in the same merge logic the final_merge
            # scripts use: prefer hi-res if decided, else L1.
            for results_dir in ("results_sam3", "results_hires_sam3"):
                d = G / results_dir
                if not d.exists(): continue
            # Easier: rescore both dirs, then for each shot pick hi-res if
            # available and not UND, else L1.
            manifest = G / "clips/shots_pipeline.json"
            l1_stdout = rescore_dir(G / "results_sam3", manifest, thr)
            hr_stdout = rescore_dir(G / "results_hires_sam3", manifest, thr)
            # Per-shot merge using the json files' NEW verdicts. But
            # rescore_descent w/o --write doesn't modify json. So we need to
            # invoke it programmatically. Take a different approach: read
            # samples directly from each json and apply rescore_one().
            # That's faster than two subprocess calls.
            pass
        by_threshold[thr] = dict(agg)
    return 0


# Direct programmatic approach (cleaner than subprocess parsing)
def rescore_one_pyimport(samples: list[dict], make_r: int) -> tuple[dict, str]:
    """Re-apply descent_verdict with a custom MAKE_R_CM.
    Uses os.environ to inject the threshold since descent_verdict reads it.
    """
    import os as _os
    saved = _os.environ.get("MAKE_R_CM")
    _os.environ["MAKE_R_CM"] = str(make_r)
    try:
        from rescore_descent import rescore_one
        return rescore_one(samples)
    finally:
        if saved is None:
            _os.environ.pop("MAKE_R_CM", None)
        else:
            _os.environ["MAKE_R_CM"] = saved


MAKE_GT = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS_GT = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}


def cat(g: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if g in MAKE_GT and v.startswith("MAKE"):
        return "TP"
    if g in MISS_GT and v.startswith("MISS"):
        return "TN"
    if g in MISS_GT and v.startswith("MAKE"):
        return "FP"
    if g in MAKE_GT and v.startswith("MISS"):
        return "FN"
    return "?"


def main_direct() -> int:
    sys.path.insert(0, str(ROOT / "pipeline"))
    # Pre-load samples for all shots, per game. Use SAM3 results as base.
    cache: dict = {}
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        manifest = json.loads((G / "shots_right.json").read_text())
        cache[gid] = []
        for s in manifest:
            l1 = G / f"results_sam3/{s['name']}.json"
            hr = G / f"results_hires_sam3/{s['name']}.json"
            l1_samples = json.loads(l1.read_text()).get('samples', []) if l1.exists() else []
            hr_samples = json.loads(hr.read_text()).get('samples', []) if hr.exists() else []
            cache[gid].append(dict(
                name=s['name'], gt=s['gt'],
                l1_samples=l1_samples, hr_samples=hr_samples))

    print(f"{'threshold':>10s}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'UND':>4}  "
          f"{'dec_pct':>8} {'ovr_pct':>8}")
    print("-" * 70)
    print(f"  AGGREGATE over {sum(len(c) for c in cache.values())} shots, "
          f"SAM3 base + hi-res merge")
    print("-" * 70)

    results: dict[int, dict] = {}
    for thr in THRESHOLDS:
        agg = defaultdict(int)
        for gid in GAMES:
            for shot in cache[gid]:
                # Re-derive L1 + hi-res verdicts under this threshold
                _, v_l1 = rescore_one_pyimport(shot['l1_samples'], thr) if shot['l1_samples'] else ({}, "UNDECIDED (no samples)")
                _, v_hr = rescore_one_pyimport(shot['hr_samples'], thr) if shot['hr_samples'] else ({}, "UNDECIDED (no samples)")
                if v_hr and not v_hr.startswith('UNDECIDED'):
                    v = v_hr
                elif v_l1 and not v_l1.startswith('UNDECIDED'):
                    v = v_l1
                elif v_hr:
                    v = v_hr
                else:
                    v = v_l1 or "UNDECIDED"
                agg[cat(shot['gt'], v)] += 1
        dec, n, acc_d, acc_o = tally(agg)
        results[thr] = dict(roll=dict(agg), dec=dec, n=n,
                            acc_d=acc_d, acc_o=acc_o)
        tp = agg.get('TP', 0); tn = agg.get('TN', 0)
        fp = agg.get('FP', 0); fn = agg.get('FN', 0); und = agg.get('UND', 0)
        print(f"  MAKE_R={thr:>2}cm  {tp:>4} {tn:>4} {fp:>4} {fn:>4} {und:>4}  "
              f"{acc_d:>7.1f}% {acc_o:>7.1f}%")

    # Find the best threshold by overall accuracy and by decided accuracy
    best_overall = max(results.items(), key=lambda kv: kv[1]['acc_o'])
    best_decided = max(results.items(), key=lambda kv: kv[1]['acc_d'])
    print(f"\nBest by OVERALL accuracy: MAKE_R_CM={best_overall[0]}cm "
          f"({best_overall[1]['acc_o']:.1f}%)")
    print(f"Best by DECIDED accuracy: MAKE_R_CM={best_decided[0]}cm "
          f"({best_decided[1]['acc_d']:.1f}%)")
    out = ROOT / "data/client_report/triangulation_test/threshold_sweep.json"
    out.write_text(json.dumps({str(k): v for k, v in results.items()},
                              indent=2))
    print(f"\nfull sweep -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main_direct())
