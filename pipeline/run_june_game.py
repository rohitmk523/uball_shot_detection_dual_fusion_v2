#!/usr/bin/env python3
"""End-to-end pipeline for one June game.

Workflow:
  1. Extract per-shot clips (FR + NR) with sync offset baked in
  2. Build clip-relative shots_pipeline.json
  3. Run L1 triangulation with the per-game calibration
  4. Run hi-res YOLO on UNDs (the ceiling-lift trick we discovered)
  5. Tally accuracy + write final_v3.json + summary

Sync offset estimation:
  Trend: G1=+1f (May ~25), G2=+7f (May 29), G3=+10.5f (May 30).
  Default for June: +13 frames (extrapolated). Override with --sync-frames.

Usage:
  python pipeline/run_june_game.py --game-id 72c08cb7
"""
from __future__ import annotations
import argparse, json, subprocess, sys, os
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


def extract_one(src: Path, t_start: float, dur: float, out: Path) -> bool:
    if is_good(out): return True
    out.unlink(missing_ok=True)
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-ss",f"{t_start:.3f}","-i",str(src),
           "-t",f"{dur:.3f}","-c","copy",str(out)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return is_good(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--sync-frames", type=float, default=13.0,
                    help="NR ahead of FR by this many frames (default 13)")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-l1", action="store_true")
    ap.add_argument("--skip-hires", action="store_true")
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    fr_video = G / f"{gid}_FR_full.mp4"
    nr_video = G / f"{gid}_NR_full.mp4"
    clips = G / "clips"
    clips.mkdir(exist_ok=True)
    results = G / "results"
    results.mkdir(exist_ok=True)
    results_hires = G / "results_hires"
    results_hires.mkdir(exist_ok=True)

    shots = json.loads((G / "shots_right.json").read_text())
    nr_shift_s = args.sync_frames / FPS
    print(f"[setup] {gid}: {len(shots)} RIGHT-angle shots, NR shift = +{nr_shift_s*1000:.0f}ms")

    if not args.skip_extract:
        print(f"[extract] clipping {len(shots)} pairs...")
        ok_count = 0
        with ProcessPoolExecutor(max_workers=6) as ex:
            futs = {}
            for s in shots:
                t0 = max(0.0, s["t_start"] - PAD_BEFORE)
                dur = (s["t_end"] + PAD_AFTER) - t0
                fr_out = clips / f"{s['name']}_FR.mp4"
                nr_out = clips / f"{s['name']}_NR.mp4"
                futs[ex.submit(extract_one, fr_video, t0, dur, fr_out)] = ("FR", s['name'])
                futs[ex.submit(extract_one, nr_video, t0 + nr_shift_s, dur, nr_out)] = ("NR", s['name'])
            for f in as_completed(futs):
                ok = f.result()
                if ok: ok_count += 1
        print(f"  {ok_count}/{2*len(shots)} clips extracted")

    # Build clip-relative manifest
    pipeline_manifest = []
    for s in shots:
        dur = s["t_end"] - s["t_start"]
        pipeline_manifest.append(dict(name=s["name"], gt=s["gt"],
                                       t_start=PAD_BEFORE, t_end=PAD_BEFORE+dur))
    (clips / "shots_pipeline.json").write_text(json.dumps(pipeline_manifest, indent=2))

    # Set calibration env var for this game
    calib_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}.json"
    os.environ["CALIB_JUNE_JSON"] = str(calib_path)

    if not args.skip_l1:
        print(f"[L1] running triangulate_shot.py (conf=0.20, imgsz=640)")
        r = subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                            "--shots-json", str(clips / "shots_pipeline.json"),
                            "--clips-dir", str(clips),
                            "--out-dir", str(results),
                            "--conf", "0.20", "--imgsz", "640"],
                           env={**os.environ, "CALIB_JUNE_JSON": str(calib_path),
                                "CALIB_JUNE": "1"})
        print(f"  L1 exit: {r.returncode}")

    # Identify UND shots for hi-res rerun
    und_targets = []
    for s in shots:
        rp = results / f"{s['name']}.json"
        if not rp.exists(): continue
        d = json.loads(rp.read_text())
        if d.get('verdict','').startswith('UNDECIDED'):
            und_targets.append(s['name'])
    print(f"[hi-res] {len(und_targets)} UND shots to rerun at imgsz=1280")

    if not args.skip_hires and und_targets:
        r = subprocess.run([sys.executable, "pipeline/triangulate_shot.py",
                            "--shots-json", str(clips / "shots_pipeline.json"),
                            "--clips-dir", str(clips),
                            "--out-dir", str(results_hires),
                            "--only", ",".join(und_targets),
                            "--conf", "0.05", "--imgsz", "1280"],
                           env={**os.environ, "CALIB_JUNE_JSON": str(calib_path),
                                "CALIB_JUNE": "1"})
        print(f"  hi-res exit: {r.returncode}")

    # Compute final accuracy: take hi-res verdict if present, else L1
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
    roll = defaultdict(int)
    by_class = defaultdict(lambda: defaultdict(int))
    for s in shots:
        v_l1 = ""; v_hr = ""
        rp = results / f"{s['name']}.json"
        if rp.exists():
            v_l1 = json.loads(rp.read_text()).get('verdict','')
        rph = results_hires / f"{s['name']}.json"
        if rph.exists():
            v_hr = json.loads(rph.read_text()).get('verdict','')
        # Use hi-res verdict if present and decisive
        if v_hr and not v_hr.startswith('UNDECIDED'):
            v = v_hr; layer = "hires"
        elif v_l1 and not v_l1.startswith('UNDECIDED'):
            v = v_l1; layer = "L1"
        elif v_hr:
            v = v_hr; layer = "hires-und"
        else:
            v = v_l1 or "UNDECIDED"; layer = "L1-und"
        c = cat(s['gt'], v)
        roll[c] += 1; by_class[s['gt']][c] += 1
        rows.append(dict(name=s['name'], gt=s['gt'], verdict=v, layer=layer, cat=c))

    (G / "final.json").write_text(json.dumps(rows, indent=2))

    print(f"\n=== {gid} FINAL ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s}")
    for cls in sorted(by_class):
        d = by_class[cls]
        tp,tn,fp,fn,und = d['TP'],d['TN'],d['FP'],d['FN'],d['UND']
        n = tp+tn+fp+fn+und
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d}")
    tp,tn,fp,fn,und = roll['TP'],roll['TN'],roll['FP'],roll['FN'],roll['UND']
    n = tp+tn+fp+fn+und; dec = tp+tn+fp+fn
    acc_d = 100*(tp+tn)/dec if dec else 0
    acc_o = 100*(tp+tn)/n if n else 0
    print("-"*55)
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d}")
    print(f"\nDecided: {tp+tn}/{dec} = {acc_d:.1f}%")
    print(f"Overall: {tp+tn}/{n} = {acc_o:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
