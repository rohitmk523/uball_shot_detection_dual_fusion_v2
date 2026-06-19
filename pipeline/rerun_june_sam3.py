#!/usr/bin/env python3
"""Re-run L1 + hi-res-on-UND on one June game using its SAM3 calibration.

Skips clip extraction (clips already exist from baseline run). Saves L1
results to results_sam3/ and hi-res to results_hires_sam3/, then writes
final_sam3.json + prints per-class accuracy.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    clips = G / "clips"
    pipeline_manifest = clips / "shots_pipeline.json"
    if not pipeline_manifest.exists():
        print(f"  ERROR: missing {pipeline_manifest}"); return 1

    sam3_calib = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_sam3.json"
    if not sam3_calib.exists():
        print(f"  ERROR: missing SAM3 calibration {sam3_calib.name}"); return 1

    results_sam3 = G / "results_sam3"; results_sam3.mkdir(exist_ok=True)
    results_hires_sam3 = G / "results_hires_sam3"; results_hires_sam3.mkdir(exist_ok=True)
    for d in (results_sam3, results_hires_sam3):
        for f in d.glob("*.json"): f.unlink()

    env = {**os.environ,
           "CALIB_JUNE_SAM3": "1",
           "CALIB_JUNE_JSON": str(sam3_calib)}

    print(f"[L1-sam3] running on {gid}")
    r = subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                        "--shots-json", str(pipeline_manifest),
                        "--clips-dir", str(clips),
                        "--out-dir", str(results_sam3),
                        "--conf", "0.20", "--imgsz", "640"], env=env)
    print(f"  L1 exit: {r.returncode}")

    # Identify UND shots
    shots = json.loads((G / "shots_right.json").read_text())
    und_targets = []
    for s in shots:
        rp = results_sam3 / f"{s['name']}.json"
        if not rp.exists(): continue
        d = json.loads(rp.read_text())
        if d.get('verdict','').startswith('UNDECIDED'):
            und_targets.append(s['name'])
    print(f"[hi-res-sam3] {len(und_targets)} UND shots")

    if und_targets:
        r = subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                            "--shots-json", str(pipeline_manifest),
                            "--clips-dir", str(clips),
                            "--out-dir", str(results_hires_sam3),
                            "--only", ",".join(und_targets),
                            "--conf", "0.05", "--imgsz", "1280"], env=env)
        print(f"  hi-res exit: {r.returncode}")

    # Final tally
    MAKE = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
    MISS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}
    def cat(g, v):
        if v.startswith("UNDECIDED"): return "UND"
        if g in MAKE and v.startswith("MAKE"): return "TP"
        if g in MISS and v.startswith("MISS"): return "TN"
        if g in MISS and v.startswith("MAKE"): return "FP"
        if g in MAKE and v.startswith("MISS"): return "FN"
        return "?"

    rows = []
    roll = defaultdict(int); by_class = defaultdict(lambda: defaultdict(int))
    for s in shots:
        v_l1 = v_hr = ""
        rp = results_sam3 / f"{s['name']}.json"
        if rp.exists(): v_l1 = json.loads(rp.read_text()).get('verdict','')
        rph = results_hires_sam3 / f"{s['name']}.json"
        if rph.exists(): v_hr = json.loads(rph.read_text()).get('verdict','')
        if v_hr and not v_hr.startswith('UNDECIDED'):
            v, layer = v_hr, "hires"
        elif v_l1 and not v_l1.startswith('UNDECIDED'):
            v, layer = v_l1, "L1"
        elif v_hr:
            v, layer = v_hr, "hires-und"
        else:
            v, layer = v_l1 or "UNDECIDED", "L1-und"
        c = cat(s['gt'], v)
        roll[c] += 1; by_class[s['gt']][c] += 1
        rows.append(dict(name=s['name'], gt=s['gt'], verdict=v, layer=layer, cat=c))
    (G / "final_sam3.json").write_text(json.dumps(rows, indent=2))

    tp,tn,fp,fn,und = roll['TP'],roll['TN'],roll['FP'],roll['FN'],roll['UND']
    n=tp+tn+fp+fn+und; dec=tp+tn+fp+fn
    acc_d = 100*(tp+tn)/dec if dec else 0
    acc_o = 100*(tp+tn)/n if n else 0
    print(f"\n=== {gid} SAM3 FINAL ===")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn} UND={und}")
    print(f"  Decided: {tp+tn}/{dec} = {acc_d:.1f}%")
    print(f"  Overall: {tp+tn}/{n} = {acc_o:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
