#!/usr/bin/env python3
"""Auto-detect NR sync offset per game by trying offsets 5..25 frames and
picking the one that minimises triangulation residual on a few representative
shots.

How it works:
  1. Pick 5-8 representative shots per game (a mix of classes/distances)
  2. For each shot: run YOLO on FR clip and NR clip ONCE (cache detections)
  3. Sweep candidate offset X in [5, 25]:
     - The clips were extracted with an assumed +13 frame baked-in offset.
       skew = X - 13 means: pair FR[f] with NR[f + skew].
     - Triangulate each frame-pair; collect ball positions.
     - Score = n_valid_in_bounds - 0.5 * RANSAC RMS residual (cm)
  4. Best X per shot; median across shots = the game's true offset.

Then re-extract clips with corrected offset and re-run pipeline.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from triangulate_shot import (   # noqa: E402
    detect_ball_hoop, triangulate, ransac_parabola, extract_arc, FPS,
)
from ultralytics import YOLO   # noqa: E402

FR_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt")
NR_W = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt")
BAKED_OFFSET = 13  # frames assumed when clips were extracted


def yolo_detections(model: YOLO, clip: Path, conf: float = 0.15
                    ) -> dict[int, tuple[float, float]]:
    """Return {frame_idx: (cx, cy)} for every frame where the model finds a ball."""
    cap = cv2.VideoCapture(str(clip))
    out: dict[int, tuple[float, float]] = {}
    f = 0
    while True:
        ok, im = cap.read()
        if not ok: break
        ball, _ = detect_ball_hoop(model, im, conf=conf, imgsz=640)
        if ball:
            out[f] = (float(ball['cx']), float(ball['cy']))
        f += 1
    cap.release()
    return out


def score_offset(fr_det: dict, nr_det: dict, skew: int, fr_P, nr_P
                 ) -> tuple[float, int, float]:
    """Triangulate paired frames at given skew. Return (score, n_inbounds, rms_cm)."""
    samples = []
    for f, (cx, cy) in fr_det.items():
        nr_f = f + skew
        if nr_f not in nr_det: continue
        ncx, ncy = nr_det[nr_f]
        X = triangulate(fr_P, nr_P, np.array([cx, cy], float),
                                       np.array([ncx, ncy], float))
        samples.append((f, X))
    if len(samples) < 6:
        return -999, 0, 999.0
    # In-bounds filter
    in_bounds = []
    for f, X in samples:
        if not (0 < X[0] < 2400 and -200 < X[1] < 1700 and -50 < X[2] < 800):
            continue
        in_bounds.append((f, X))
    if len(in_bounds) < 6:
        return -100, len(in_bounds), 999.0

    ts = np.array([f/FPS for f,_ in in_bounds])
    xs = np.array([X[0] for _,X in in_bounds])
    ys = np.array([X[1] for _,X in in_bounds])
    zs = np.array([X[2] for _,X in in_bounds])

    # Filter to arc (drop ball-in-hands pre-shot)
    try:
        arc = extract_arc(ts, zs, xs, min_apex_cm=240, t_before=0.8, t_after=0.15)
    except Exception:
        return -50, len(in_bounds), 999.0
    if len(arc) < 6:
        return -50, len(in_bounds), 999.0
    parab, inl = ransac_parabola(ts[arc], xs[arc], ys[arc], zs[arc],
                                   n_iter=200, inlier_cm=70)
    rms = parab['rms_cm']
    n_in = int(inl.sum())
    # Score = many inliers, low RMS
    score = float(n_in) - 0.5 * rms
    return score, n_in, float(rms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--n-shots", type=int, default=6,
                    help="number of representative shots to use")
    ap.add_argument("--sweep-min", type=int, default=5)
    ap.add_argument("--sweep-max", type=int, default=25)
    args = ap.parse_args()
    gid = args.game_id

    G = ROOT / f"data/client_report/triangulation_test/june_{gid}"
    clips = G / "clips"
    shots = json.loads((G / "shots_right.json").read_text())

    # Load calibration to get camera matrices
    calib = json.loads((ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_sam3.json").read_text())
    def _pack(c):
        K = np.array(c['K']); rvec = np.array(c['rvec'])
        t = np.array(c['tvec']).reshape(3,1)
        R, _ = cv2.Rodrigues(rvec)
        return K @ np.hstack([R, t])
    fr_P = _pack(calib['FR'])
    nr_P = _pack(calib['NR'])

    # Pick representative shots: 2 FG_MAKE, 1 3PT, 1 4PT, 1 FT, 1 random
    picks = []
    by_class = {"FG_MAKE": [], "3PT_MAKE": [], "4PT_MAKE": [], "FREE_THROW_MAKE": []}
    for s in shots:
        if s['gt'] in by_class: by_class[s['gt']].append(s)
    if by_class["FG_MAKE"]: picks.extend(by_class["FG_MAKE"][:2])
    if by_class["3PT_MAKE"]: picks.append(by_class["3PT_MAKE"][0])
    if by_class["4PT_MAKE"]: picks.append(by_class["4PT_MAKE"][0])
    if by_class["FREE_THROW_MAKE"]: picks.append(by_class["FREE_THROW_MAKE"][0])
    # Fill to n_shots
    while len(picks) < args.n_shots and len(picks) < len(shots):
        for s in shots:
            if s not in picks:
                picks.append(s)
                if len(picks) >= args.n_shots: break
    picks = picks[:args.n_shots]
    print(f"[{gid}] picked {len(picks)} representative shots:")
    for p in picks: print(f"  {p['name']:18s} {p['gt']}")

    fr_model = YOLO(str(FR_W)); nr_model = YOLO(str(NR_W))

    sweep_offsets = list(range(args.sweep_min, args.sweep_max+1))
    print(f"\n[{gid}] sweeping NR offset {args.sweep_min}..{args.sweep_max} frames")
    print(f"  (clips were extracted assuming +{BAKED_OFFSET}; skew = offset - {BAKED_OFFSET})")

    per_shot_best = []
    for p in picks:
        fr_clip = clips / f"{p['name']}_FR.mp4"
        nr_clip = clips / f"{p['name']}_NR.mp4"
        if not (fr_clip.exists() and nr_clip.exists()):
            print(f"  {p['name']}: missing clips"); continue
        print(f"\n  shot {p['name']} ({p['gt']}): running YOLO once on each clip...")
        fr_det = yolo_detections(fr_model, fr_clip, conf=0.15)
        nr_det = yolo_detections(nr_model, nr_clip, conf=0.15)
        print(f"    FR detections: {len(fr_det)}  NR detections: {len(nr_det)}")
        best = None
        for X in sweep_offsets:
            skew = X - BAKED_OFFSET
            score, n_in, rms = score_offset(fr_det, nr_det, skew, fr_P, nr_P)
            if best is None or score > best[1]:
                best = (X, score, n_in, rms)
            print(f"    offset={X:>2d}  skew={skew:+d}  n_in={n_in:>3d}  rms={rms:>6.1f}  score={score:+7.1f}")
        print(f"    BEST: offset={best[0]} (score={best[1]:.1f}, n={best[2]}, rms={best[3]:.1f})")
        per_shot_best.append(best[0])

    print(f"\n[{gid}] per-shot best offsets: {per_shot_best}")
    if per_shot_best:
        med = int(np.median(per_shot_best))
        print(f"\n=== {gid} median best offset = {med} frames ===")

    # Save result
    out = dict(game_id=gid, baked_offset=BAKED_OFFSET,
               per_shot_best=per_shot_best,
               median_best=int(np.median(per_shot_best)) if per_shot_best else None,
               sweep_range=[args.sweep_min, args.sweep_max])
    (G / "sync_detect.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
