#!/usr/bin/env python3
"""Download just the per-shot windows for game 5a5f1aae directly from S3 via
signed URL + ffmpeg seek. ~13 MB per camera per shot vs 10 GB whole game.

For each shot we extract t_start-1.0 .. t_end+2.5 to give a small lead-in and
the post-release descent buffer the triangulation pipeline already adds.
"""
from __future__ import annotations
import json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data/client_report/triangulation_test/full_game/clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data/client_report/triangulation_test/full_game/shots_88.json"

FR_S3 = "s3://uball-videos-production/court-a/2026-05-28/5a5f1aae-06f0-4a6e-80d1/2026-05-28_5a5f1aae-06f0-4a6e-80d1_FR.mp4"
NR_S3 = "s3://uball-videos-production/court-a/2026-05-28/5a5f1aae-06f0-4a6e-80d1/2026-05-28_5a5f1aae-06f0-4a6e-80d1_NR.mp4"

# Sync offset: NR_frame = FR_frame + 1. We'll bake that into the extracted
# NR clips by shifting NR's start by +1 frame = +0.033 sec, so both clips
# share a t=0 origin downstream (the pipeline then uses sync_offset=0).
NR_SHIFT_S = 1.0 / 29.97

PAD_BEFORE = 1.0
PAD_AFTER  = 2.5


def presign(s3_url: str, expires: int = 7200) -> str:
    r = subprocess.run(["aws", "s3", "presign", s3_url,
                        "--expires-in", str(expires)],
                       check=True, capture_output=True, text=True)
    return r.stdout.strip()


def extract(url: str, t_start: float, dur: float, out_path: Path) -> dict:
    if out_path.exists() and out_path.stat().st_size > 100_000:
        return {"path": str(out_path), "skipped": True}
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{t_start:.3f}", "-i", url,
           "-t", f"{dur:.3f}", "-c", "copy", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return {"path": str(out_path), "ok": r.returncode == 0,
            "size_kb": out_path.stat().st_size // 1024 if out_path.exists() else 0,
            "stderr": r.stderr[-300:] if r.returncode != 0 else ""}


def extract_pair(args):
    shot, fr_url, nr_url = args
    t0 = max(0.0, shot['t_start'] - PAD_BEFORE)
    dur = (shot['t_end'] + PAD_AFTER) - t0
    fr_out = CLIPS_DIR / f"{shot['name']}_FR.mp4"
    nr_out = CLIPS_DIR / f"{shot['name']}_NR.mp4"
    fr_res = extract(fr_url, t0,                dur, fr_out)
    nr_res = extract(nr_url, t0 + NR_SHIFT_S,   dur, nr_out)
    return dict(name=shot['name'], gt=shot['gt'],
                t_start_orig=shot['t_start'], t_end_orig=shot['t_end'],
                t_start_clip=PAD_BEFORE,
                t_end_clip=shot['t_end'] - shot['t_start'] + PAD_BEFORE,
                fr=fr_res, nr=nr_res)


def main():
    shots = json.loads(MANIFEST.read_text())
    print(f"[setup] {len(shots)} shots; signing S3 URLs ...")
    fr_url = presign(FR_S3); nr_url = presign(NR_S3)
    tasks = [(s, fr_url, nr_url) for s in shots]
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(extract_pair, t): t[0]['name'] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            fr_ok = "✓" if not r['fr'].get('skipped') and r['fr'].get('ok') else \
                    ("S" if r['fr'].get('skipped') else "✗")
            nr_ok = "✓" if not r['nr'].get('skipped') and r['nr'].get('ok') else \
                    ("S" if r['nr'].get('skipped') else "✗")
            print(f"  [{done:>3}/{len(shots)}] {r['name']:14s}  FR{fr_ok}({r['fr'].get('size_kb',0):>5}kB)  "
                  f"NR{nr_ok}({r['nr'].get('size_kb',0):>5}kB)")
    # save extraction manifest + a pipeline-compatible shots manifest
    (CLIPS_DIR / "extraction_manifest.json").write_text(
        json.dumps(results, indent=2))
    pipeline_shots = [
        {"name": r['name'], "gt": r['gt'],
         "t_start": r['t_start_clip'], "t_end": r['t_end_clip']}
        for r in results
        if (r['fr'].get('ok') or r['fr'].get('skipped'))
        and (r['nr'].get('ok') or r['nr'].get('skipped'))
    ]
    (CLIPS_DIR / "shots_pipeline.json").write_text(
        json.dumps(pipeline_shots, indent=2))
    print(f"\n  {len(pipeline_shots)}/{len(results)} shot-clips ready in {CLIPS_DIR}")
    print(f"  wrote shots_pipeline.json with {len(pipeline_shots)} entries")


if __name__ == "__main__":
    sys.exit(main())
