#!/usr/bin/env python3
"""Extract missing per-shot clips from local full-game FR + NR mp4s.

Far more reliable than ffmpeg-seek over HTTP-signed S3 URLs — local seeks
hit a single moov atom. Used to backfill the 31 shots that produced
corrupt clips during the S3-streaming extraction phase.
"""
from __future__ import annotations
import json, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FG = ROOT / "data/client_report/triangulation_test/full_game"
CLIPS = FG / "clips"
FR_FULL = FG / "5a5f1aae_FR_full.mp4"
NR_FULL = FG / "5a5f1aae_NR_full.mp4"

PAD_BEFORE = 1.0
PAD_AFTER = 2.5
NR_SHIFT_S = 1.0 / 29.97   # bake sync into NR clip


def is_good(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100_000:
        return False
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return r.returncode == 0 and float(r.stdout.strip()) > 0.5
    except Exception:
        return False


def extract(src: Path, t_start: float, dur: float, out: Path) -> dict:
    if is_good(out):
        return {"path": str(out), "skipped": True}
    out.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{t_start:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-c", "copy", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = is_good(out)
    return {"path": str(out), "ok": ok,
            "size_kb": out.stat().st_size // 1024 if out.exists() else 0,
            "stderr": r.stderr[-200:] if not ok else ""}


def do_pair(shot):
    t0 = max(0.0, shot["t_start"] - PAD_BEFORE)
    dur = (shot["t_end"] + PAD_AFTER) - t0
    fr_out = CLIPS / f"{shot['name']}_FR.mp4"
    nr_out = CLIPS / f"{shot['name']}_NR.mp4"
    fr = extract(FR_FULL, t0, dur, fr_out)
    nr = extract(NR_FULL, t0 + NR_SHIFT_S, dur, nr_out)
    return dict(name=shot["name"], fr=fr, nr=nr)


def main():
    manifest = json.loads((FG / "shots_88.json").read_text())
    missing = [s for s in manifest
               if not is_good(CLIPS / f"{s['name']}_FR.mp4")
               or not is_good(CLIPS / f"{s['name']}_NR.mp4")]
    print(f"[setup] {len(missing)} of 88 shots have missing/corrupt clips")
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(do_pair, s): s["name"] for s in missing}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            fr_ok = "✓" if (r['fr'].get('ok') or r['fr'].get('skipped')) else "✗"
            nr_ok = "✓" if (r['nr'].get('ok') or r['nr'].get('skipped')) else "✗"
            print(f"  [{done:>2}/{len(missing)}] {r['name']:14s}  "
                  f"FR{fr_ok}({r['fr'].get('size_kb',0):>5}kB)  "
                  f"NR{nr_ok}({r['nr'].get('size_kb',0):>5}kB)")
    # report full ready count
    ready = sum(1 for s in manifest
                if is_good(CLIPS / f"{s['name']}_FR.mp4")
                and is_good(CLIPS / f"{s['name']}_NR.mp4"))
    print(f"\n  total ready pairs: {ready} / {len(manifest)}")


if __name__ == "__main__":
    main()
