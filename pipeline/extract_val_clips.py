#!/usr/bin/env python3
"""Extract per-shot FR/NR clip pairs for the May validation games directly
from S3 (presigned URL + ffmpeg seek), same convention as extract_shot_clips:

  - window = t_start - 1.0 .. t_end + 2.5  (lead-in + descent buffer)
  - NR start shifted by SYNC_FRAMES/29.97 so both clips share t=0
    (June auto-sync default = +13 frames; re-run with --sync-frames N --nr-only
     when measured offsets arrive)
  - writes clips/shots_pipeline.json with clip-relative t_start/t_end

Usage:
  python pipeline/extract_val_clips.py --game-id 77715f25 [--sync-frames 13]
  python pipeline/extract_val_clips.py --all
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

S3_GAMES = {
    "77715f25": ("2026-05-19", "77715f25-bf6e-4986-824b"),
    "cc1710c4": ("2026-05-19", "cc1710c4-ddac-44ef-9e78"),
    "fb677f72": ("2026-05-22", "fb677f72-9f2c-4f43-9c4f"),
    "b3c1f62c": ("2026-05-16", "b3c1f62c-1a02-47c9-8d2a"),
    "cc5deb39": ("2026-05-19", "cc5deb39-71de-4ea9-b3a2"),
    "f3e7b25a": ("2026-05-16", "f3e7b25a-430a-4b37-aa80"),
    "ee8745f1": ("2026-04-16", "ee8745f1-863f-47cf-a43d"),
    "6d601c99": ("2026-04-18", "6d601c99-9173-445f-a647"),
    "c2a354fe": ("2026-03-19", "c2a354fe-eb34-4980-af00"),
}
FPS = 29.97
PAD_BEFORE = 1.0
PAD_AFTER = 2.5


def presign(s3_url: str, expires: int = 7200) -> str:
    r = subprocess.run(["aws", "s3", "presign", s3_url,
                        "--expires-in", str(expires)],
                       check=True, capture_output=True, text=True)
    return r.stdout.strip()


def is_good(p: Path) -> bool:
    """Size check is not enough: an interrupted -c copy leaves a big file
    with no moov atom. Validate with ffprobe like extract_local_clips."""
    if not p.exists() or p.stat().st_size < 100_000:
        return False
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(p)],
        capture_output=True, text=True, timeout=30)
    try:
        return r.returncode == 0 and float(r.stdout.strip()) > 0.5
    except Exception:
        return False


def extract(url: str, t_start: float, dur: float, out_path: Path) -> bool:
    try:
        if is_good(out_path):
            return True
        out_path.unlink(missing_ok=True)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{t_start:.3f}", "-i", url,
               "-t", f"{dur:.3f}", "-c", "copy", str(out_path)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return is_good(out_path)
    except Exception as e:                       # timeout / network hiccup
        print(f"  EXC {out_path.name}: {type(e).__name__}")
        return False


def run_game(gid8: str, sync_frames: int, nr_only: bool, workers: int,
             prefix: str = "val_") -> int:
    date, gid = S3_GAMES[gid8]
    G = ROOT / f"data/client_report/triangulation_test/{prefix}{gid8}"
    clips = G / "clips"
    clips.mkdir(exist_ok=True)
    shots = json.loads((G / "shots_right.json").read_text())
    base = f"s3://uball-videos-production/court-a/{date}/{gid}/{date}_{gid}"
    fr_url = presign(f"{base}_FR.mp4")
    nr_url = presign(f"{base}_NR.mp4")
    nr_shift = sync_frames / FPS

    jobs = []
    manifest = []
    for s in shots:
        t0 = max(0.0, s["t_start"] - PAD_BEFORE)
        dur = (s["t_end"] + PAD_AFTER) - t0
        if not nr_only:
            jobs.append((fr_url, t0, dur, clips / f"{s['name']}_FR.mp4"))
        jobs.append((nr_url, t0 + nr_shift, dur, clips / f"{s['name']}_NR.mp4"))
        manifest.append(dict(name=s["name"], gt=s["gt"],
                             t_start=round(s["t_start"] - t0, 3),
                             t_end=round(s["t_end"] - t0, 3)))

    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract, *j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            done += 1
            if not f.result():
                fails += 1
                print(f"  FAIL {futs[f][3].name}")
            if done % 40 == 0:
                print(f"  [{gid8}] {done}/{len(jobs)}")
    (clips / "shots_pipeline.json").write_text(json.dumps(manifest, indent=1))
    print(f"[{gid8}] done: {len(jobs)-fails}/{len(jobs)} clips ok, "
          f"sync_frames={sync_frames}, manifest n={len(manifest)}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sync-frames", type=int, default=13)
    ap.add_argument("--nr-only", action="store_true",
                    help="re-extract only NR clips (after measured sync)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dir-prefix", default="val_")
    args = ap.parse_args()
    targets = list(S3_GAMES) if args.all else [args.game_id]
    total_fails = 0
    for gid8 in targets:
        total_fails += run_game(gid8, args.sync_frames, args.nr_only,
                                args.workers, args.dir_prefix)
    return 1 if total_fails else 0


if __name__ == "__main__":
    sys.exit(main())
