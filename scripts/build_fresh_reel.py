#!/usr/bin/env python3
"""Build an error-review reel for the FRESH out-of-sample games.

For every shot the production model got wrong (data/p3_fresh_predictions.parquet,
correct==False), cut the shot window from the correct court-side cameras and
show FAR | NEAR side-by-side with a subtitle stating truth vs model + the reason
hint. 41/42 errors are MISS called MAKE (depth illusion) — seeing FAR (true miss)
next to NEAR (looks like a make) is the point.

Clips are pulled straight from S3 via presigned-URL HTTP range seeks (no full
video download). Local only. Usage:
    python3 scripts/build_fresh_reel.py [--limit N] [--workers K]
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "client_report" / "fresh_error_reel"
TMP = OUT / "_clips"
GT = Path("/tmp/gt_fresh.json")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
PAD = 1.5           # seconds of lead/trail around the GT window
TIMEOUT = 240       # per-error ffmpeg ceiling


PLAYS_META = Path("/tmp/plays_meta.json")   # authoritative angle+timestamps from public.plays


def load_jobs(limit: int | None):
    manifest = {g["game_id"]: g
                for g in json.loads((ROOT / "data" / "games_manifest.json")
                                    .read_text())["games"]}
    # PRIMARY: precise angle + start/end from the uball.ai plays table (dumped
    # to /tmp/plays_meta.json). FALLBACK: the rounded GT export.
    gt = json.loads(GT.read_text())["games"]
    meta = {s["play_id"]: dict(t0=s["start_timestamp"], t1=s["end_timestamp"],
                               angle=s["angle"], cls=s["classification"])
            for gid, shots in gt.items() for s in shots}
    if PLAYS_META.exists():
        for pid, p in json.loads(PLAYS_META.read_text()).items():
            meta[pid] = dict(t0=p["t0"], t1=p["t1"], angle=p["angle"],
                             cls=p["cls"])
        print(f"[reel] using plays-table timestamps for "
              f"{len(json.loads(PLAYS_META.read_text()))} plays")
    pred = pd.read_parquet(ROOT / "data" / "p3_fresh_predictions.parquet")
    err = pred[pred.correct == False].copy()                       # noqa: E712
    err = err.sort_values(["game_id", "prob"], ascending=[True, False])
    jobs = []
    for _, r in err.iterrows():
        m = meta[r.play_id]
        jobs.append(dict(gid=r.game_id, play=r.play_id, m=m,
                         label=int(r.label), pred=int(r.pred),
                         prob=float(r.prob), manifest=manifest[r.game_id]))
    return jobs if limit is None else jobs[:limit]


def uri(g, cam):
    return (f"s3://uball-videos-production/{g['s3_prefix']}"
            f"{g['date']}_{g['game_id'][:23]}_{cam}.mp4")


def presign(u):
    return subprocess.run(["aws", "s3", "presign", u, "--expires-in", "10800"],
                          capture_output=True, text=True, check=True).stdout.strip()


def esc(t: str) -> str:
    return t.replace(":", "\\:").replace("'", "")


def build_one(idx: int, j: dict) -> tuple[int, Path | None, str]:
    m = j["m"]
    far, near = ("FL", "NL") if m["angle"] == "LEFT" else ("FR", "NR")
    t0 = max(0.0, float(m["t0"]) - PAD)
    dur = (float(m["t1"]) - float(m["t0"])) + 2 * PAD
    g = j["manifest"]
    truth = "MAKE" if j["label"] == 1 else "MISS"
    model = "MAKE" if j["pred"] == 1 else "MISS"
    kind = ("FALSE POSITIVE (called MAKE, was MISS)" if (truth == "MISS" and model == "MAKE")
            else "FALSE NEGATIVE (called MISS, was MAKE)" if (truth == "MAKE" and model == "MISS")
            else "ERROR")
    cap = esc(f"#{idx+1}  {j['gid'][:8]}  {m['angle']} side  TRUTH={truth}  "
              f"MODEL={model} p={j['prob']:.2f}  {kind}  [{m['cls']}]")
    out = TMP / f"{idx:02d}.mp4"
    fc = (
        f"[0:v]scale=900:506,drawtext=fontfile={FONT}:text='FAR {far}':"
        f"x=12:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
        f"[1:v]scale=900:506,drawtext=fontfile={FONT}:text='NEAR {near}':"
        f"x=12:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6[b];"
        f"[a][b]hstack=inputs=2[h];"
        f"[h]pad=iw:ih+56:0:0:black,drawtext=fontfile={FONT}:text='{cap}':"
        f"x=(w-text_w)/2:y=h-42:fontsize=24:fontcolor=white[v]"
    )
    try:
        fu, nu = presign(uri(g, far)), presign(uri(g, near))
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t0:.2f}", "-i", fu, "-ss", f"{t0:.2f}", "-i", nu,
             "-t", f"{dur:.2f}", "-filter_complex", fc, "-map", "[v]",
             "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", str(out)],
            check=True, timeout=TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return idx, out, "ok"
    except subprocess.TimeoutExpired:
        return idx, None, "TIMEOUT"
    except subprocess.CalledProcessError as e:
        return idx, None, (e.stderr or b"").decode()[-200:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(a.limit)
    print(f"[reel] building {len(jobs)} error clips with {a.workers} workers")
    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, i, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futs):
            idx, path, msg = f.result()
            print(f"  clip {idx:02d}: {'OK' if path else 'FAIL'} {msg if msg!='ok' else ''}")
            if path:
                results[idx] = path
    if not results:
        print("[reel] no clips built; aborting"); sys.exit(1)
    ordered = [results[i].resolve() for i in sorted(results)]
    concat = TMP / "_concat.txt"
    concat.write_text("\n".join(f"file '{c}'" for c in ordered))
    final = OUT / "fresh_error_reel.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", str(final)], check=True)
    print(f"[reel] {len(ordered)}/{len(jobs)} clips -> {final}")
    print(f"[reel] size {final.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
