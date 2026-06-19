#!/usr/bin/env python3
"""Joint sweep: apply MAKE_R_CM threshold T to BOTH SAM3 and HYBRID samples,
ensemble the resulting verdicts, then measure accuracy.

This stacks two known improvements: per-shot ensemble of two calibrations
(+1.4 pts decided over SAM3) and a tighter MAKE_R threshold (+1.7 pts
decided on SAM3 alone at 35cm). Does combining help further?

Usage:
  python pipeline/threshold_sweep_ensemble.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rescore_descent import rescore_one  # noqa: E402
from ensemble_sam3_hybrid import ensemble_pair, MAKE_GT, MISS_GT, cat  # noqa: E402

GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")
THRESHOLDS = (25, 28, 30, 32, 35, 38, 40, 42, 45)


def rescore_with_thr(samples: list[dict], thr: int) -> str:
    if not samples or len(samples) < 3:
        return "UNDECIDED (no samples)"
    saved = os.environ.get("MAKE_R_CM")
    os.environ["MAKE_R_CM"] = str(thr)
    try:
        _, v = rescore_one(samples)
    finally:
        if saved is None:
            os.environ.pop("MAKE_R_CM", None)
        else:
            os.environ["MAKE_R_CM"] = saved
    return v


def merge_l1_hr(v_l1: str, v_hr: str) -> str:
    if v_hr and not v_hr.startswith('UNDECIDED'):
        return v_hr
    if v_l1 and not v_l1.startswith('UNDECIDED'):
        return v_l1
    return v_hr or v_l1 or "UNDECIDED"


def main() -> int:
    print("loading samples...")
    cache: dict = {}
    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        shots = json.loads((G / "shots_right.json").read_text())
        cache[gid] = []
        for s in shots:
            entry = dict(name=s['name'], gt=s['gt'])
            for variant, l1_dir, hr_dir in [
                ("sam3", "results_sam3", "results_hires_sam3"),
                ("hybrid", "results_hybrid", "results_hires_hybrid"),
            ]:
                for tag, d in (("l1", l1_dir), ("hr", hr_dir)):
                    p = G / d / f"{s['name']}.json"
                    entry[f"{variant}_{tag}"] = (
                        json.loads(p.read_text()).get('samples', [])
                        if p.exists() else [])
            cache[gid].append(entry)
    n_shots = sum(len(c) for c in cache.values())
    print(f"loaded {n_shots} shots")

    print()
    print(f"{'thr':>5s}  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'UND':>4}  "
          f"{'dec_pct':>8} {'ovr_pct':>8}    agree disagree fill-sam3 fill-hybrid both-und")
    print("-" * 110)

    results: dict[int, dict] = {}
    for thr in THRESHOLDS:
        agg = defaultdict(int)
        src = defaultdict(int)
        for gid in GAMES:
            for shot in cache[gid]:
                v_sam3 = merge_l1_hr(
                    rescore_with_thr(shot['sam3_l1'], thr),
                    rescore_with_thr(shot['sam3_hr'], thr))
                v_hybrid = merge_l1_hr(
                    rescore_with_thr(shot['hybrid_l1'], thr),
                    rescore_with_thr(shot['hybrid_hr'], thr))
                v_ens, source = ensemble_pair(v_sam3, v_hybrid)
                agg[cat(shot['gt'], v_ens)] += 1
                src[source] += 1
        tp = agg.get('TP', 0); tn = agg.get('TN', 0)
        fp = agg.get('FP', 0); fn = agg.get('FN', 0); und = agg.get('UND', 0)
        n = tp + tn + fp + fn + und
        dec = tp + tn + fp + fn
        acc_d = 100 * (tp + tn) / dec if dec else 0
        acc_o = 100 * (tp + tn) / n if n else 0
        results[thr] = dict(roll=dict(agg), src=dict(src),
                            dec_pct=acc_d, ovr_pct=acc_o)
        print(f"  {thr:>3}cm  {tp:>4} {tn:>4} {fp:>4} {fn:>4} {und:>4}  "
              f"{acc_d:>7.1f}% {acc_o:>7.1f}%    "
              f"{src['agree']:>3} {src['disagree']:>4} {src['filled-by-sam3']:>5} "
              f"{src['filled-by-hybrid']:>6} {src['both-und']:>5}")

    best_overall = max(results.items(), key=lambda kv: kv[1]['ovr_pct'])
    best_decided = max(results.items(), key=lambda kv: kv[1]['dec_pct'])
    print()
    print(f"Best by OVERALL: MAKE_R={best_overall[0]}cm ({best_overall[1]['ovr_pct']:.1f}%)")
    print(f"Best by DECIDED: MAKE_R={best_decided[0]}cm ({best_decided[1]['dec_pct']:.1f}%)")
    out = ROOT / "data/client_report/triangulation_test/threshold_sweep_ensemble.json"
    out.write_text(json.dumps({str(k): v for k, v in results.items()},
                              indent=2))
    print(f"\nfull sweep -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
