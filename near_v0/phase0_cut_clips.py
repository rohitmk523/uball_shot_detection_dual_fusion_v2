#!/usr/bin/env python3
"""Phase 0 (NEAR_ANGLE_PLAN.md): cut NR shot clips from local June full videos.

Stratified sample across shot classes from the 3 June DEV games
(e74164e6 is frozen test -- excluded). Window = t_start-1.0 .. t_end+2.5.
Re-encode (not -c copy) for frame-accurate starts. ffprobe-validate output.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRI = REPO / "data/client_report/triangulation_test"
OUT = REPO / "data/client_report/near_angle/phase0/clips"

DEV_GAMES = {
    "4692eb2b": TRI / "june_4692eb2b/4692eb2b_NR_full.mp4",
    "454da9cf": TRI / "june_454da9cf/454da9cf_NR_full.mp4",
    "72c08cb7": TRI / "june_72c08cb7/72c08cb7_NR_full.mp4",
}
# per-class total targets across all games
TARGETS = {
    "FREE_THROW_MAKE": 3, "FREE_THROW_MISS": 3,
    "FG_MAKE": 4, "FG_MISS": 4,
    "3PT_MAKE": 3, "3PT_MISS": 3,
    "4PT_MAKE": 2, "4PT_MISS": 2,
}
PAD_BEFORE, PAD_AFTER = 1.0, 2.5


def ffprobe_ok(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and float(r.stdout.strip()) > 0.5
    except Exception:
        return False


def pick_shots():
    """Round-robin across games per class so no single game dominates."""
    by_class = defaultdict(list)
    for gid, _video in DEV_GAMES.items():
        shots = json.loads((TRI / f"june_{gid}/shots_right.json").read_text())
        for s in shots:
            by_class[s["gt"]].append({**s, "game": gid})
    picked = []
    for cls, want in TARGETS.items():
        pool = by_class.get(cls, [])
        per_game = defaultdict(list)
        for s in pool:
            per_game[s["game"]].append(s)
        # spread picks: alternate games, take shots spaced through the game
        games = sorted(per_game)
        i = 0
        while sum(1 for p in picked if p["gt"] == cls) < want and i < 50:
            g = games[i % len(games)]
            if per_game[g]:
                # take from middle of remaining list for time diversity
                picked.append(per_game[g].pop(len(per_game[g]) // 2))
            i += 1
    return picked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    picked = pick_shots()
    manifest = []
    for s in picked:
        t0 = max(0.0, s["t_start"] - PAD_BEFORE)
        dur = (s["t_end"] + PAD_AFTER) - t0
        name = f"{s['game']}_{s['name']}_{s['gt']}"
        out = OUT / f"{name}.mp4"
        if not (out.exists() and ffprobe_ok(out)):
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-ss", f"{t0:.3f}", "-i", str(DEV_GAMES[s["game"]]),
                   "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-an", str(out)]
            subprocess.run(cmd, timeout=300)
        ok = ffprobe_ok(out)
        manifest.append({**s, "clip": out.name, "t0_game": round(t0, 3),
                         "t_start_clip": round(s["t_start"] - t0, 3),
                         "t_end_clip": round(s["t_end"] - t0, 3), "ok": ok})
        print(f"{'OK ' if ok else 'BAD'} {name}  dur={dur:.1f}s")
    (OUT.parent / "phase0_manifest.json").write_text(json.dumps(manifest, indent=1))
    n_ok = sum(1 for m in manifest if m["ok"])
    print(f"\n{n_ok}/{len(manifest)} clips valid -> {OUT}")
    return 0 if n_ok == len(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
