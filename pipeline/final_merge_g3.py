#!/usr/bin/env python3
"""Game-3 tiered final merge: L1 triangulation -> L2 UND-rerun -> L3 ensemble
-> L5 multi-shot override. (No L4 hi-res yet; game-3 doesn't have a hi-res
results directory.)
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from merge_und_results import is_high_conf   # noqa: E402

G3 = ROOT / "data/client_report/triangulation_test/game3_3398befc"
MAIN = G3 / "results"
RERUN = G3 / "results_und_conf10"
HIRES = G3 / "results_hires"
ENS_PATH = G3 / "ensemble_results.json"
MULTI_PATH = G3 / "multi_shot_results.json"

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def classify(gt: str, v: str) -> str:
    if v.startswith("UNDECIDED"): return "UND"
    if gt in MAKE_LABELS and v.startswith("MAKE"):  return "TP"
    if gt in MISS_LABELS and v.startswith("MISS"):  return "TN"
    if gt in MISS_LABELS and v.startswith("MAKE"):  return "FP"
    if gt in MAKE_LABELS and v.startswith("MISS"):  return "FN"
    return "?"


def main() -> int:
    manifest = json.loads((G3 / "shots_right.json").read_text())
    gt_map = {s["name"]: s["gt"] for s in manifest}

    final: dict[str, str] = {}
    layer: dict[str, str] = {}

    # L1 triangulation (RIGHT-75 only)
    for f in MAIN.glob("*.json"):
        if f.name == "summary.json": continue
        try:
            d = json.loads(f.read_text())
            if not isinstance(d, dict) or 'name' not in d: continue
            if d["name"] not in gt_map: continue   # skip LEFT shots
            final[d["name"]] = d.get("verdict", "")
            layer[d["name"]] = "tri"
        except Exception:
            continue

    # L2 UND-rerun (conf=0.10)
    for name, v in list(final.items()):
        if v.startswith("UNDECIDED"):
            rp = RERUN / f"{name}.json"
            if rp.exists():
                rv = json.loads(rp.read_text()).get("verdict", "")
                if is_high_conf(rv):
                    final[name] = rv; layer[name] = "L2-und-rerun"

    # L3 per-camera ensemble
    if ENS_PATH.exists():
        ens_rows = json.loads(ENS_PATH.read_text())
        for r in ens_rows:
            name = r["name"]
            if name not in gt_map: continue   # RIGHT-only
            ev = r["final"]
            old = final.get(name, "")
            if ev != old:
                final[name] = ev; layer[name] = "L3-ensemble"
    else:
        print("[warn] ensemble_results.json not present, skipping L3")

    # L4 hi-res YOLO on known-error shots (FPs + FNs targeted with imgsz=1280,
    # conf=0.05). Override final verdict with hi-res whenever it differs from
    # current final and is not blank.
    if HIRES.exists():
        for hf in HIRES.glob("*.json"):
            if hf.name == "summary.json": continue
            try:
                d = json.loads(hf.read_text())
                if not isinstance(d, dict) or 'name' not in d: continue
                name = d["name"]
                if name not in gt_map: continue
                hv = d.get("verdict", "")
                if hv and hv != final.get(name, ""):
                    final[name] = hv; layer[name] = "L4-hires"
            except Exception:
                continue

    # L5 multi-shot override (first attempt verdict)
    multi_overrides: list[tuple] = []
    if MULTI_PATH.exists():
        for r in json.loads(MULTI_PATH.read_text()):
            if r["n_attempts"] < 2: continue
            first_v = r.get("first") or ""
            if not first_v or first_v.startswith("UNDECIDED"): continue
            name = r["name"]
            if name not in gt_map: continue   # RIGHT-only
            old = final.get(name, "")
            if first_v != old:
                multi_overrides.append((name, gt_map[name], old, first_v,
                                        classify(gt_map[name], old),
                                        classify(gt_map[name], first_v)))
                final[name] = first_v + "  [MULTI-SHOT: 1st attempt]"
                layer[name] = "L5-multishot"

    print(f"=== L5 multi-shot OVERRIDES ({len(multi_overrides)}) ===")
    for name, gt, old, new, oc, nc in multi_overrides:
        marker = ("+" if oc in ("FP","FN","UND") and nc in ("TP","TN")
                  else "-" if oc in ("TP","TN") and nc in ("FP","FN") else ".")
        print(f"  {marker} {name:18s} {gt:18s} {oc:3s}->{nc:3s}  "
              f"old: {old[:35]:35s}  new: {new[:35]:35s}")

    # Tally
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    roll: dict[str, int] = defaultdict(int)
    for name, v in final.items():
        gt = gt_map[name]; cat = classify(gt, v)
        by_class[gt][cat] += 1; roll[cat] += 1

    print(f"\n=== GAME-3 TIERED FINAL ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]; n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp+tn+fp+fn; acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-"*60)
    tp,tn,fp,fn,und = (roll['TP'], roll['TN'], roll['FP'], roll['FN'], roll['UND'])
    n = tp+tn+fp+fn+und; decided = tp+tn+fp+fn
    acc_dec = 100*(tp+tn)/decided if decided else 0
    acc_all = 100*(tp+tn)/n if n else 0
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc_dec:>6.1f}")
    print(f"\n  Decided accuracy: {tp+tn}/{decided} = {acc_dec:.1f}%")
    print(f"  Overall accuracy: {tp+tn}/{n} = {acc_all:.1f}%")
    print(f"\n  By layer:")
    lc: dict[str, int] = defaultdict(int)
    for v in layer.values(): lc[v] += 1
    for k, c in sorted(lc.items()):
        print(f"    {k}: {c}")

    out_rows = [dict(name=name, gt=gt_map[name], verdict=final[name],
                     layer=layer[name], cat=classify(gt_map[name], final[name]))
                for name in sorted(final)]
    (G3 / "final_v3.json").write_text(json.dumps(out_rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
