#!/usr/bin/env python3
"""Re-extract NR clips with corrected sync offset, then rerun L1 + hi-res
with SAM3 calibration. Writes results_sync/, results_hires_sync/,
final_sync.json.

Usage:
  python pipeline/rerun_june_sync.py --game-id 454da9cf --sync-frames 16
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPS = 29.97
PAD_BEFORE = 1.0
PAD_AFTER = 2.5


def is_good(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 100_000: return False
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=nk=1:nw=1",str(p)], capture_output=True,
                       text=True, timeout=30)
    try: return r.returncode == 0 and float(r.stdout.strip()) > 0.5
    except Exception: return False


def extract_nr(src: Path, t0: float, dur: float, out: Path) -> bool:
    out.unlink(missing_ok=True)
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-ss",f"{t0:.3f}","-i",str(src),
           "-t",f"{dur:.3f}","-c","copy",str(out)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return is_good(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--sync-frames", type=float, required=True)
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    nr_video = G / f"{gid}_NR_full.mp4"
    clips = G / "clips"
    shots_right = json.loads((G / "shots_right.json").read_text())
    nr_shift_s = args.sync_frames / FPS
    print(f"[{gid}] re-extracting NR clips with offset = +{args.sync_frames} frames "
          f"({nr_shift_s*1000:.0f} ms)")

    # Re-extract NR only (FR clips stay)
    ok = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {}
        for s in shots_right:
            t0 = max(0.0, s["t_start"] - PAD_BEFORE)
            dur = (s["t_end"] + PAD_AFTER) - t0
            nr_out = clips / f"{s['name']}_NR.mp4"
            futs[ex.submit(extract_nr, nr_video, t0 + nr_shift_s, dur, nr_out)] = s['name']
        for f in as_completed(futs):
            if f.result(): ok += 1
    print(f"  {ok}/{len(shots_right)} NR clips re-extracted")

    sam3_calib = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_sam3.json"
    if not sam3_calib.exists():
        print(f"  ERROR: SAM3 calibration missing"); return 1

    results_sync = G / "results_sync"; results_sync.mkdir(exist_ok=True)
    results_hires_sync = G / "results_hires_sync"; results_hires_sync.mkdir(exist_ok=True)
    for d in (results_sync, results_hires_sync):
        for f in d.glob("*.json"): f.unlink()

    env = {**os.environ, "CALIB_JUNE_SAM3":"1", "CALIB_JUNE_JSON":str(sam3_calib)}
    pipeline_manifest = clips / "shots_pipeline.json"

    print(f"[L1] running with corrected sync...")
    subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                    "--shots-json", str(pipeline_manifest),
                    "--clips-dir", str(clips),
                    "--out-dir", str(results_sync),
                    "--conf", "0.20", "--imgsz", "640"], env=env)

    # Identify UND
    und = []
    for s in shots_right:
        rp = results_sync / f"{s['name']}.json"
        if not rp.exists(): continue
        v = json.loads(rp.read_text()).get('verdict','')
        if v.startswith('UNDECIDED'): und.append(s['name'])
    print(f"[hi-res] {len(und)} UND shots")

    if und:
        subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                        "--shots-json", str(pipeline_manifest),
                        "--clips-dir", str(clips),
                        "--out-dir", str(results_hires_sync),
                        "--only", ",".join(und),
                        "--conf", "0.05", "--imgsz", "1280"], env=env)

    # Tally
    MAKE = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
    MISS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}
    def cat(g, v):
        if v.startswith("UNDECIDED"): return "UND"
        if g in MAKE and v.startswith("MAKE"): return "TP"
        if g in MISS and v.startswith("MISS"): return "TN"
        if g in MISS and v.startswith("MAKE"): return "FP"
        if g in MAKE and v.startswith("MISS"): return "FN"
        return "?"

    rows = []; roll = defaultdict(int)
    for s in shots_right:
        v_l1 = v_hr = ""
        rp = results_sync / f"{s['name']}.json"
        if rp.exists(): v_l1 = json.loads(rp.read_text()).get('verdict','')
        rph = results_hires_sync / f"{s['name']}.json"
        if rph.exists(): v_hr = json.loads(rph.read_text()).get('verdict','')
        if v_hr and not v_hr.startswith('UNDECIDED'): v=v_hr; layer="hires"
        elif v_l1 and not v_l1.startswith('UNDECIDED'): v=v_l1; layer="L1"
        elif v_hr: v=v_hr; layer="hires-und"
        else: v=v_l1 or "UNDECIDED"; layer="L1-und"
        c = cat(s['gt'], v); roll[c] += 1
        rows.append(dict(name=s['name'], gt=s['gt'], verdict=v, layer=layer, cat=c))
    (G / "final_sync.json").write_text(json.dumps(rows, indent=2))

    tp,tn,fp,fn,und_cnt = roll['TP'],roll['TN'],roll['FP'],roll['FN'],roll['UND']
    n=tp+tn+fp+fn+und_cnt; dec=tp+tn+fp+fn
    acc_d=100*(tp+tn)/dec if dec else 0
    print(f"\n=== {gid} SYNC-CORRECTED FINAL ===")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn} UND={und_cnt}")
    print(f"  Decided: {tp+tn}/{dec} = {acc_d:.1f}%")
    print(f"  Overall: {tp+tn}/{n} = {100*(tp+tn)/n if n else 0:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
