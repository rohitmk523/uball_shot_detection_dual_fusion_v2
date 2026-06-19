#!/usr/bin/env python3
"""Verdict-level ensemble of SAM3 and HYBRID calibrations.

Each calibration produces a per-shot verdict file. We merge them shot-by-shot:

  * Both decided + AGREE              -> their shared verdict (high confidence)
  * Both decided + DISAGREE           -> UNDECIDED (ensemble flags ambiguity)
  * One decided, one UND              -> the decided one (fill in UND gaps)
  * Both UND                          -> UND

Then re-tally per game and aggregate. Compares against SAM3 baseline,
HYBRID variant, and the ensemble.

Inputs: final_sam3.json + final_hybrid.json per game (already exist).
Output: final_ensemble.json + console summary.

Usage:
  python pipeline/ensemble_sam3_hybrid.py
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def verdict_class(v: str) -> str:
    """Reduce a long verdict string to {MAKE, MISS, UND}."""
    if v.startswith("UNDECIDED"):
        return "UND"
    if v.startswith("MAKE"):
        return "MAKE"
    if v.startswith("MISS"):
        return "MISS"
    return "UND"


def ensemble_pair(v_sam3: str, v_hybrid: str) -> tuple[str, str]:
    """Combine two verdict strings; return (ensemble_verdict, source_tag)."""
    c_s = verdict_class(v_sam3)
    c_h = verdict_class(v_hybrid)
    if c_s == "UND" and c_h == "UND":
        return v_sam3 or v_hybrid, "both-und"
    if c_s == "UND":
        return v_hybrid, "filled-by-hybrid"
    if c_h == "UND":
        return v_sam3, "filled-by-sam3"
    if c_s == c_h:
        # Both decided and agree (MAKE vs MAKE or MISS vs MISS).
        # Prefer SAM3 string since its rim-center cross-check is tighter
        # (more trustworthy descent_verdict text).
        return v_sam3, "agree"
    # Disagreement on a decided verdict: surface as undecided.
    return "UNDECIDED (sam3-hybrid disagree)", "disagree"


MAKE_GT = {"FREE_THROW_MAKE", "FG_MAKE", "3PT_MAKE", "4PT_MAKE"}
MISS_GT = {"FREE_THROW_MISS", "FG_MISS", "3PT_MISS", "4PT_MISS"}


def cat(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"):
        return "UND"
    if gt in MAKE_GT and v.startswith("MAKE"):
        return "TP"
    if gt in MISS_GT and v.startswith("MISS"):
        return "TN"
    if gt in MISS_GT and v.startswith("MAKE"):
        return "FP"
    if gt in MAKE_GT and v.startswith("MISS"):
        return "FN"
    return "?"


def roll(rows: list[dict]) -> dict:
    out = defaultdict(int)
    for r in rows:
        out[r['cat']] += 1
    return dict(out)


def print_row(label: str, r: dict, total: int = 0) -> tuple[float, float]:
    tp = r.get('TP', 0); tn = r.get('TN', 0)
    fp = r.get('FP', 0); fn = r.get('FN', 0); und = r.get('UND', 0)
    n = tp + tn + fp + fn + und if total == 0 else total
    dec = tp + tn + fp + fn
    acc_d = 100 * (tp + tn) / dec if dec else 0
    acc_o = 100 * (tp + tn) / n if n else 0
    print(f"  {label:>18s}  {tp:>3} {tn:>3} {fp:>3} {fn:>3} {und:>3}  "
          f"{acc_d:>6.1f}%  {acc_o:>6.1f}%")
    return acc_d, acc_o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Write final_ensemble.json files per game")
    args = ap.parse_args()

    print(f"{'game':>20s}  {'variant':>18s}  "
          f"{'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} {'UND':>3}  "
          f"{'dec_pct':>8}  {'ovr_pct':>8}")
    print("-" * 96)

    agg = {variant: defaultdict(int) for variant in
           ("SAM3", "HYBRID", "ENSEMBLE")}
    agg_total = 0
    ensemble_source_counts = defaultdict(int)

    for gid in GAMES:
        G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
        sam3_rows = json.loads((G / "final_sam3.json").read_text())
        hybrid_rows = json.loads((G / "final_hybrid.json").read_text())

        sam3_by_name = {r['name']: r for r in sam3_rows}
        hybrid_by_name = {r['name']: r for r in hybrid_rows}
        assert set(sam3_by_name) == set(hybrid_by_name), \
            f"{gid}: shot-name mismatch SAM3 vs HYBRID"

        ensemble_rows = []
        for name, sam3 in sam3_by_name.items():
            hybrid = hybrid_by_name[name]
            v_ensemble, source = ensemble_pair(sam3['verdict'], hybrid['verdict'])
            ensemble_source_counts[source] += 1
            ensemble_rows.append(dict(
                name=name, gt=sam3['gt'],
                verdict=v_ensemble,
                source=source,
                verdict_sam3=sam3['verdict'],
                verdict_hybrid=hybrid['verdict'],
                cat=cat(sam3['gt'], v_ensemble),
            ))
        if args.write:
            (G / "final_ensemble.json").write_text(
                json.dumps(ensemble_rows, indent=2))

        total = len(sam3_rows)
        agg_total += total
        print(f"  {gid:>20s}")
        for variant, rows in [("SAM3", sam3_rows),
                                ("HYBRID", hybrid_rows),
                                ("ENSEMBLE", ensemble_rows)]:
            r = roll(rows)
            for k in ("TP", "TN", "FP", "FN", "UND"):
                agg[variant][k] += r.get(k, 0)
            print_row(variant, r, total)
        print()

    print("-" * 96)
    print(f"  AGGREGATE (n={agg_total})")
    for variant in ("SAM3", "HYBRID", "ENSEMBLE"):
        print_row(variant, agg[variant], agg_total)

    print()
    print(f"  ensemble source distribution: "
          f"agree={ensemble_source_counts['agree']} "
          f"disagree={ensemble_source_counts['disagree']} "
          f"filled-by-sam3={ensemble_source_counts['filled-by-sam3']} "
          f"filled-by-hybrid={ensemble_source_counts['filled-by-hybrid']} "
          f"both-und={ensemble_source_counts['both-und']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
