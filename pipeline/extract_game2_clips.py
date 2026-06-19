#!/usr/bin/env python3
"""Extract per-shot clips for game-2 (dc5f199e) with the +7 frame NR offset
baked in. NR_SHIFT_S = 7 / 29.97 seconds.

This mirrors extract_local_clips.py but for game-2.
"""
from __future__ import annotations
import json, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G2 = ROOT / "data/client_report/triangulation_test/game2_dc5f199e"
CLIPS = G2 / "clips"
FR_FULL = G2 / "dc5f199e_FR_full.mp4"
NR_FULL = G2 / "dc5f199e_NR_full.mp4"

PAD_BEFORE = 1.0
PAD_AFTER = 2.5
NR_SHIFT_S = 7.0 / 29.97   # bake +7 frame offset into NR clip start


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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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
    # NR shifted by +7 frames so the extracted clips are physically aligned
    nr = extract(NR_FULL, t0 + NR_SHIFT_S, dur, nr_out)
    return dict(name=shot["name"], fr=fr, nr=nr)


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((G2 / "shots_usable.json").read_text())
    print(f"[setup] {len(manifest)} shots to extract")
    print(f"[setup] NR offset: +{NR_SHIFT_S*1000:.0f} ms (+7 frames)")

    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(do_pair, s): s["name"] for s in manifest}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            fr_ok = "✓" if (r['fr'].get('ok') or r['fr'].get('skipped')) else "✗"
            nr_ok = "✓" if (r['nr'].get('ok') or r['nr'].get('skipped')) else "✗"
            print(f"  [{done:>2}/{len(manifest)}] {r['name']:18s}  "
                  f"FR{fr_ok}({r['fr'].get('size_kb',0):>5}kB)  "
                  f"NR{nr_ok}({r['nr'].get('size_kb',0):>5}kB)")
    # report ready
    ready = sum(1 for s in manifest
                if is_good(CLIPS / f"{s['name']}_FR.mp4")
                and is_good(CLIPS / f"{s['name']}_NR.mp4"))
    print(f"\n  Ready pairs: {ready} / {len(manifest)}")


if __name__ == "__main__":
    main()
