#!/usr/bin/env python3
"""Mine training frames for the near-angle ball+hoop detector retrain.

Build mode (default): turn data/near_detector/plays_all.json into a sampling
manifest -- per game/angle, <=25 stratified shots x 6 timestamps each
(rim-moment-heavy) + 12 no-shot gap frames.

Extract mode (--game GID8 --angle NR|NL --video PATH_OR_URL): cut the sampled
frames for one game/angle into data/near_detector/frames/. Works with a local
file (fast seeks) or a presigned S3 URL (use on AWS, in-region).

Frozen test games are excluded at build time (NEAR_ANGLE_PLAN.md 8.1).
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ND = REPO / "data/near_detector"
FROZEN = {"c2a354fe", "6d601c99", "ee8745f1", "0fa23810", "49b3873e", "e74164e6"}

MAX_SHOTS = 25          # per game/angle
NOSHOT_FRAMES = 12      # per game/angle
SHOT_OFFSETS = [(-0.8, "rim"), (-0.4, "rim"), (0.0, "rim"),
                (0.4, "rim"), (0.8, "rim")]   # relative to t1 (play end)
APPROACH_OFFSET = 0.3   # relative to t0


def fam(cls: str) -> str:
    return cls.rsplit("_", 1)[0]


def build_manifest() -> dict:
    plays = json.loads((ND / "plays_all.json").read_text())
    plays = [p for p in plays if p["gid8"] not in FROZEN]
    for p in plays:  # postgres numeric serializes as string
        p["t0"], p["t1"] = float(p["t0"]), float(p["t1"])
    by_ga = defaultdict(list)
    for p in plays:
        by_ga[(p["gid8"], p["angle"])].append(p)

    manifest = {}
    for (gid, angle), pool in sorted(by_ga.items()):
        pool.sort(key=lambda p: p["t0"])
        # stratify: round-robin over (family, make/miss) buckets
        buckets = defaultdict(list)
        for p in pool:
            buckets[(fam(p["cls"]), p["cls"].endswith("MAKE"))].append(p)
        picked, i = [], 0
        keys = sorted(buckets, key=lambda k: -len(buckets[k]))
        while len(picked) < min(MAX_SHOTS, len(pool)) and i < 200:
            k = keys[i % len(keys)]
            if buckets[k]:
                picked.append(buckets[k].pop(len(buckets[k]) // 2))
            i += 1
        picked.sort(key=lambda p: p["t0"])

        frames = []
        for p in picked:
            for off, kind in SHOT_OFFSETS:
                frames.append({"t": round(p["t1"] + off, 2), "kind": kind,
                               "pid8": p["pid8"]})
            frames.append({"t": round(p["t0"] + APPROACH_OFFSET, 2),
                           "kind": "approach", "pid8": p["pid8"]})
        # no-shot frames from gaps between consecutive plays (ball in play
        # elsewhere or dead ball -- hard negatives for heads/jerseys)
        gaps = []
        for a, b in zip(pool, pool[1:]):
            if b["t0"] - a["t1"] > 8.0:
                gaps.append((a["t1"] + 3.0, b["t0"] - 3.0))
        g = 0
        while len([f for f in frames if f["kind"] == "noshot"]) < NOSHOT_FRAMES and gaps:
            lo, hi = gaps[g % len(gaps)]
            t = lo + ((g * 7919) % max(1, int(hi - lo)))  # deterministic spread
            frames.append({"t": round(float(t), 2), "kind": "noshot", "pid8": "none"})
            g += 1
        manifest[f"{gid}_{ 'NR' if angle == 'RIGHT' else 'NL' }"] = frames
    return manifest


def extract(gid: str, cam: str, video: str, out_dir: Path, frames: list) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for f in frames:
        name = f"{gid}_{cam}_t{f['t']:08.1f}_{f['pid8']}_{f['kind']}.jpg"
        out = out_dir / name
        if out.exists() and out.stat().st_size > 30_000:
            n_ok += 1
            continue
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{f['t']:.2f}",
             "-i", video, "-frames:v", "1", "-q:v", "2", str(out)],
            capture_output=True, timeout=120)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 30_000:
            n_ok += 1
        else:
            out.unlink(missing_ok=True)
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--angle", choices=["NR", "NL"])
    ap.add_argument("--video", help="local path or presigned URL")
    ap.add_argument("--out", default=str(ND / "frames"))
    args = ap.parse_args()

    mpath = ND / "sampling_manifest.json"
    if not mpath.exists():
        manifest = build_manifest()
        mpath.write_text(json.dumps(manifest, indent=0))
        total = sum(len(v) for v in manifest.values())
        print(f"manifest: {len(manifest)} game/angle sets, {total} frames")
    else:
        manifest = json.loads(mpath.read_text())

    if args.game and args.video:
        key = f"{args.game}_{args.angle}"
        frames = manifest.get(key)
        if not frames:
            print(f"no frames for {key}", file=sys.stderr)
            return 1
        n = extract(args.game, args.angle, args.video, Path(args.out), frames)
        print(f"{key}: {n}/{len(frames)} frames extracted")
        return 0 if n >= len(frames) * 0.9 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
