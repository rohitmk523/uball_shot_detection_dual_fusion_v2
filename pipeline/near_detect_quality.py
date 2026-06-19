#!/usr/bin/env python3
"""Near-angle detection-quality probe on the 5 fresh games.

Answers: does the NEAR detector need more training, or is near's weakness the
camera ANGLE (geometry), not detection? For each shot we look at the play's
near camera (LEFT->NL, RIGHT->NR) and measure how well it sees the ball + rim,
especially in the frames where the ball is AT the rim (where a Noah-style
ball-size depth cue would have to operate). Far (far_v16) shown for contrast.
Local CPU only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRESH = Path("/tmp/p1tracks_fresh")
PLAYS = json.loads(Path("/tmp/plays_meta.json").read_text()) if Path("/tmp/plays_meta.json").exists() else {}
GT = json.loads(Path("/tmp/gt_fresh.json").read_text())["games"]

# play_id -> side (RIGHT/LEFT). plays table first, GT fallback.
SIDE = {pid: p["angle"] for pid, p in PLAYS.items()}
for gid, shots in GT.items():
    for s in shots:
        SIDE.setdefault(s["play_id"], s.get("angle", "LEFT"))


def near_cam(side: str) -> str:
    return "NL" if side == "LEFT" else "NR"


def far_cam(side: str) -> str:
    return "FL" if side == "LEFT" else "FR"


def cxy(g, p):
    return g[f"{p}_x"] + g[f"{p}_w"] / 2, g[f"{p}_y"] + g[f"{p}_h"] / 2


def shot_stats(g: pd.DataFrame) -> dict:
    """g = one (play, camera) frame slice sorted by frame_idx."""
    n = len(g)
    rim_ok = g.rim_x.notna()
    ball_ok = g.ball_x.notna()
    out = dict(n=n, rim_rate=float(rim_ok.mean()) if n else 0.0,
               ball_rate=float(ball_ok.mean()) if n else 0.0,
               ball_near_rim=0, ball_seen_at_rim=False)
    both = g[rim_ok & ball_ok]
    if len(both):
        bcx, bcy = cxy(both, "ball")
        rcx, rcy = cxy(both, "rim")
        near = (np.abs(bcx - rcx) < 1.5 * both.rim_w) & \
               (np.abs(bcy - rcy) < 2.0 * both.rim_h)
        out["ball_near_rim"] = int(near.sum())
        out["ball_seen_at_rim"] = bool(near.sum() >= 2)
    return out


def main():
    rows = []
    for pq in sorted(FRESH.glob("*.parquet")):
        gid = pq.stem
        df = pd.read_parquet(pq)
        for pid, gp in df.groupby("play_id"):
            side = SIDE.get(pid, "LEFT")
            ncam, fcam = near_cam(side), far_cam(side)
            ns = shot_stats(gp[gp.angle == ncam].sort_values("frame_idx"))
            fs = shot_stats(gp[gp.angle == fcam].sort_values("frame_idx"))
            rows.append(dict(game=gid[:8], play=pid[:8], side=side,
                             near_ball=ns["ball_rate"], near_rim=ns["rim_rate"],
                             near_at_rim=ns["ball_seen_at_rim"],
                             far_ball=fs["ball_rate"], far_rim=fs["rim_rate"],
                             far_at_rim=fs["ball_seen_at_rim"]))
    d = pd.DataFrame(rows)
    print(f"shots: {len(d)}\n")
    print("=== per-game NEAR vs FAR detection (means over shots) ===")
    print(f"{'game':<10}{'shots':>6}{'nearBall':>9}{'nearRim':>8}{'near@rim%':>10}"
          f"{'farBall':>9}{'farRim':>8}{'far@rim%':>9}")
    for gid, g in d.groupby("game"):
        print(f"{gid:<10}{len(g):>6}{g.near_ball.mean():>9.2f}{g.near_rim.mean():>8.2f}"
              f"{100*g.near_at_rim.mean():>9.0f}%{g.far_ball.mean():>9.2f}"
              f"{g.far_rim.mean():>8.2f}{100*g.far_at_rim.mean():>8.0f}%")
    print("-" * 69)
    print(f"{'OVERALL':<10}{len(d):>6}{d.near_ball.mean():>9.2f}{d.near_rim.mean():>8.2f}"
          f"{100*d.near_at_rim.mean():>9.0f}%{d.far_ball.mean():>9.2f}"
          f"{d.far_rim.mean():>8.2f}{100*d.far_at_rim.mean():>8.0f}%")
    print()
    blind = d[~d.near_at_rim]
    print(f"shots where NEAR never sees the ball at the rim: {len(blind)}/{len(d)} "
          f"({100*len(blind)/len(d):.0f}%)")
    print(f"  of those, FAR DOES see ball at rim: {int(blind.far_at_rim.sum())} "
          f"-> near-detection-limited (a better near detector could help)")
    print(f"  near AND far both blind at rim: {int((~blind.far_at_rim).sum())} "
          f"-> not fixable by near detector alone")
    d.to_parquet(ROOT / "data" / "near_detect_quality_fresh.parquet", index=False)
    print(f"\nsaved data/near_detect_quality_fresh.parquet")


if __name__ == "__main__":
    main()
