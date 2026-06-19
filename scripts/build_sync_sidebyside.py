#!/usr/bin/env python3
"""Build a 2-minute side-by-side FR | NR clip with absolute frame numbers
overlaid on each pane, so the user can pick a matching frame in each
camera and report the sync offset back.

Usage:
  python scripts/build_sync_sidebyside.py \
      --fr <path>.mp4 --nr <path>.mp4 --out <path>.mp4 \
      [--start 0] [--dur 120]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    return float(r.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr", required=True)
    ap.add_argument("--nr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0,
                    help="seek time (seconds)")
    ap.add_argument("--dur", type=float, default=120.0,
                    help="clip length (seconds)")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--width", type=int, default=960,
                    help="per-pane output width")
    args = ap.parse_args()

    fr = Path(args.fr); nr = Path(args.nr)
    for p in (fr, nr):
        if not p.exists():
            print(f"missing: {p}"); return 1

    fr_dur = ffprobe_duration(fr)
    nr_dur = ffprobe_duration(nr)
    print(f"FR duration: {fr_dur:.1f}s ({fr_dur/60:.1f}m)")
    print(f"NR duration: {nr_dur:.1f}s ({nr_dur/60:.1f}m)")
    print(f"Diff       : {fr_dur - nr_dur:+.1f}s  "
          f"(positive = FR longer; negative = NR longer)")

    # Compute starting frame number per camera (assumes both videos
    # start at recording-start; the sync offset is what we're measuring).
    start_frame = int(args.start * args.fps)
    h = int(args.width * 9 / 16)  # 16:9
    # drawtext start_number resets frame counter at this offset so the
    # overlay matches absolute frame number in the source clip.
    drawtext = (
        f"drawtext=fontfile=/System/Library/Fonts/Menlo.ttc:"
        f"text='%{{frame_num}}':start_number={start_frame}:"
        f"x=20:y=20:fontsize=42:fontcolor=yellow:"
        f"box=1:boxcolor=black@0.6:boxborderw=8"
    )
    filter_complex = (
        f"[0:v]trim=start={args.start}:duration={args.dur},setpts=PTS-STARTPTS,"
        f"scale={args.width}:{h},{drawtext},"
        f"drawtext=fontfile=/System/Library/Fonts/Menlo.ttc:text='FR':"
        f"x=w-tw-20:y=20:fontsize=36:fontcolor=cyan:box=1:boxcolor=black@0.6"
        f"[fr];"
        f"[1:v]trim=start={args.start}:duration={args.dur},setpts=PTS-STARTPTS,"
        f"scale={args.width}:{h},{drawtext},"
        f"drawtext=fontfile=/System/Library/Fonts/Menlo.ttc:text='NR':"
        f"x=w-tw-20:y=20:fontsize=36:fontcolor=cyan:box=1:boxcolor=black@0.6"
        f"[nr];"
        f"[fr][nr]hstack=inputs=2[v]"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(fr), "-i", str(nr),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", str(args.fps), "-an",
        str(out),
    ]
    print(f"\n[ffmpeg] writing {out}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    print(f"OK ({out.stat().st_size/1024/1024:.1f} MB)")
    print(f"\nPick a matching frame in FR and NR, then tell me 'FR:X = NR:Y'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
