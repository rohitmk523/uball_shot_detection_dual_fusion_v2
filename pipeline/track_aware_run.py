#!/usr/bin/env python3
"""Run triangulation with ByteTrack-aware ball detection.

For each shot clip:
  1. Run YOLO + ByteTrack on FR and NR independently.
  2. Identify the "shot ball" tracks per camera — the union of tracks that
     produce a sensible high-arc trajectory in the shot window.
  3. For each frame, prefer the shot-ball detection (drop spurious player /
     ref / floor-ball detections).
  4. Triangulate the filtered detections and apply the v4 verdict pipeline.

This addresses the failure mode in `8b2ffd4b_FR` where raw YOLO swapped
between the real shot ball and another basketball during the rim-passing
phase, producing impossible XY jumps that broke the rim-plane crossing
interpolation.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from triangulate_shot import (   # noqa: E402
    calibrate, triangulate, extract_arc, fit_parabola, ransac_parabola,
    descent_verdict, nr_rebound_check, FPS, SYNC_OFFSET_NR,
    FR_WEIGHTS, NR_WEIGHTS, RIM_X, RIM_Y, RIM_Z,
)

FG = ROOT / "data/client_report/triangulation_test/full_game"
CLIPS = FG / "clips"

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}


def tracked_balls(model: YOLO, clip: Path, conf: float = 0.10) -> list[dict]:
    """Stream tracking; return one dict per frame with all ball detections
    + their persistent track IDs."""
    per_frame = []
    for r in model.track(source=str(clip), conf=conf, persist=True,
                         tracker="bytetrack.yaml", stream=True,
                         verbose=False, classes=[0]):
        frame_balls = []
        if r.boxes.id is not None:
            for j in range(len(r.boxes)):
                if int(r.boxes.cls[j]) != 0: continue
                cnf = float(r.boxes.conf[j])
                tid = int(r.boxes.id[j])
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                frame_balls.append(dict(
                    id=tid, conf=cnf,
                    cx=float((x1+x2)/2), cy=float((y1+y2)/2),
                    w=float(x2-x1), h=float(y2-y1)))
        per_frame.append(frame_balls)
    return per_frame


def identify_shot_ball_tracks(per_frame: list[list[dict]],
                              shot_high_in_image: bool) -> set[int]:
    """Pick the track IDs that are the most likely "the shot ball".

    Heuristic: the shot ball has the LARGEST cy span in the clip — it
    travels from low (camera view of release height) to high (descent into
    bottom of frame on NR) or vice versa. For FR the shot ball passes
    through low cy (top of image = near rim) at apex.

    We return tracks whose cy-range is at least 60% of the max cy-range
    observed (catches the shot ball if it's split across IDs by occlusion).
    """
    tracks = defaultdict(list)   # id -> list of (frame, cx, cy, conf)
    for fi, balls in enumerate(per_frame):
        for b in balls:
            tracks[b['id']].append((fi, b['cx'], b['cy'], b['conf']))

    if not tracks:
        return set()

    # Score each track by cy span (and require minimum length)
    scores = {}
    for tid, pts in tracks.items():
        if len(pts) < 3: continue
        cys = [p[2] for p in pts]
        cy_span = max(cys) - min(cys)
        # bonus for tracks reaching extreme cy (near rim level)
        extreme = (min(cys) if shot_high_in_image else max(cys))
        scores[tid] = (cy_span, len(pts), extreme)

    if not scores:
        return set()

    # Best score = largest cy_span
    max_span = max(s[0] for s in scores.values())
    if max_span < 50:
        # No strong arc; fall back to all tracks (don't filter)
        return set(tracks.keys())
    keep = {tid for tid, s in scores.items() if s[0] >= 0.6 * max_span}

    # Also include tracks that overlap in time AND space with the kept tracks
    # (the shot ball can be split into multiple tracks by short occlusions)
    kept_frames = set()
    kept_xy = []
    for tid in keep:
        for f, cx, cy, _ in tracks[tid]:
            kept_frames.add(f); kept_xy.append((f, cx, cy))
    if kept_xy:
        for tid, pts in tracks.items():
            if tid in keep: continue
            for f, cx, cy, _ in pts:
                # if temporally close (±5 frames) to a kept detection AND
                # spatially within 150 px, include this track
                for kf, kx, ky in kept_xy:
                    if abs(f - kf) <= 5 and np.hypot(cx-kx, cy-ky) < 150:
                        keep.add(tid); break
                if tid in keep: break
    return keep


def select_ball_for_frame(balls: list[dict], shot_ids: set[int],
                          prev_xy: tuple[float, float] | None) -> dict | None:
    """Pick the single ball detection that's most likely the shot ball.

    Strategy: do not strictly filter by track ID (that's too aggressive
    when the shot ball gets split by occlusions). Instead, when multiple
    detections exist in a frame, pick the one closest to the previous
    selected position — that gives continuity through frames without
    locking out the real ball when it briefly changes track IDs.
    """
    if not balls: return None
    if prev_xy is None:
        # First frame: prefer a detection whose track ID is in the broader
        # shot-ball set (some ID-based filtering on the very first pick),
        # then by confidence.
        cands = [b for b in balls if b['id'] in shot_ids] or balls
        return max(cands, key=lambda b: b['conf'])
    px, py = prev_xy
    # Continuity-based selection. Discard candidates >300 px from previous —
    # those are different basketballs entirely.
    near = [b for b in balls if np.hypot(b['cx']-px, b['cy']-py) <= 300]
    if not near: return None
    return min(near, key=lambda b: np.hypot(b['cx']-px, b['cy']-py))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots-json", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    shots = json.loads(Path(args.shots_json).read_text())
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir)

    print("[setup] loading YOLO models")
    fr_model = YOLO(str(FR_WEIGHTS))
    nr_model = YOLO(str(NR_WEIGHTS))

    print("[setup] calibrating")
    fr_cal, nr_cal = calibrate()

    summary = []
    for s in shots:
        name = s['name']; gt = s['gt']
        t_start = s['t_start']; t_end = s['t_end']
        fr_clip = clips_dir / f"{name}_FR.mp4"
        nr_clip = clips_dir / f"{name}_NR.mp4"
        if not (fr_clip.exists() and nr_clip.exists()):
            print(f"  [skip] {name}"); continue

        print(f"\n=== {name}  GT={gt}  clip t={t_start:.1f}–{t_end:.1f}s ===")

        # Pass 1: track each camera
        fr_per_frame = tracked_balls(fr_model, fr_clip, conf=args.conf)
        nr_per_frame = tracked_balls(nr_model, nr_clip, conf=args.conf)
        # Sync: NR shifted by 1 frame already (baked into the clips by
        # extract_local_clips.py) — but cv2.VideoCapture is 0-indexed and
        # the tracker returns frames in clip order. SYNC_OFFSET is 0 here.

        # Pass 2: identify shot-ball tracks per camera
        # In FR the shot ball reaches HIGH in image (cy small at apex)
        # In NR the shot ball reaches DEEP in image (cy large at apex)
        fr_shot_ids = identify_shot_ball_tracks(fr_per_frame, shot_high_in_image=True)
        nr_shot_ids = identify_shot_ball_tracks(nr_per_frame, shot_high_in_image=False)
        print(f"  shot-ball track IDs:  FR={sorted(fr_shot_ids)[:6]}{'...' if len(fr_shot_ids)>6 else ''}  "
              f"NR={sorted(nr_shot_ids)[:6]}{'...' if len(nr_shot_ids)>6 else ''}")

        # Pass 3: per-frame triangulation using filtered detections
        n_frames = min(len(fr_per_frame), len(nr_per_frame))
        samples = []
        prev_fr = prev_nr = None
        for fi in range(n_frames):
            ball_fr = select_ball_for_frame(fr_per_frame[fi], fr_shot_ids, prev_fr)
            ball_nr = select_ball_for_frame(nr_per_frame[fi], nr_shot_ids, prev_nr)
            if ball_fr and ball_nr:
                X = triangulate(fr_cal['P'], nr_cal['P'],
                                np.array([ball_fr['cx'], ball_fr['cy']]),
                                np.array([ball_nr['cx'], ball_nr['cy']]))
                samples.append(dict(
                    frame=fi, t=fi/FPS,
                    fr_px=(ball_fr['cx'], ball_fr['cy']),
                    fr_conf=ball_fr['conf'], fr_id=ball_fr['id'],
                    nr_px=(ball_nr['cx'], ball_nr['cy']),
                    nr_conf=ball_nr['conf'], nr_id=ball_nr['id'],
                    X_cm=X.tolist()))
                prev_fr = (ball_fr['cx'], ball_fr['cy'])
                prev_nr = (ball_nr['cx'], ball_nr['cy'])

        print(f"  triangulated samples (tracked): {len(samples)}")

        if len(samples) < 6:
            verdict = "UNDECIDED (too few tracked samples)"
        else:
            ts = np.array([s['t']    for s in samples])
            xs = np.array([s['X_cm'][0] for s in samples])
            ys = np.array([s['X_cm'][1] for s in samples])
            zs = np.array([s['X_cm'][2] for s in samples])
            bounds = (xs > 0) & (xs < 2400) & (ys > -200) & (ys < 1700) & \
                     (zs > -50) & (zs < 800)
            if bounds.sum() < 6:
                verdict = "UNDECIDED (post-bounds too few)"
            else:
                ts, xs, ys, zs = ts[bounds], xs[bounds], ys[bounds], zs[bounds]
                arc = extract_arc(ts, zs, xs, min_apex_cm=240,
                                  t_before=0.8, t_after=0.15)
                if len(arc) < 3:
                    verdict = "UNDECIDED (arc too short)"
                else:
                    apex_idx = int(np.argmax(zs[arc]))
                    apex = dict(x=float(xs[arc][apex_idx]),
                                y=float(ys[arc][apex_idx]),
                                z=float(zs[arc][apex_idx]),
                                t=float(ts[arc][apex_idx]))
                    apex['dxy_to_rim'] = float(np.hypot(
                        apex['x']-RIM_X, apex['y']-RIM_Y))
                    apex_dxy = apex['dxy_to_rim']
                    apex_t = apex['t']
                    apex_sample_idx = min(range(len(samples)),
                                          key=lambda i: abs(samples[i]['t']-apex_t))
                    descent_info, descent_v = descent_verdict(
                        samples, apex_sample_idx, apex_dxy, apex['z'])
                    verdict = f"{descent_v}  [apex r={apex_dxy:.0f}cm, z_peak={apex['z']:.0f}]"
                    # NR-rebound override
                    if descent_v.startswith("MAKE"):
                        post_apex_z_min = min(
                            (samples[i]['X_cm'][2]
                             for i in range(apex_sample_idx,
                                            min(apex_sample_idx+30, len(samples)))),
                            default=9999)
                        rebound, max_cy, min_cy_after, dt_reb = nr_rebound_check(
                            samples, apex_sample_idx)
                        if rebound and post_apex_z_min > 150:
                            verdict = (f"MISS (NR-rebound override: cy {max_cy:.0f}→"
                                       f"{min_cy_after:.0f} in {dt_reb:.2f}s, "
                                       f"Δ={max_cy-min_cy_after:.0f}px)  "
                                       f"[was: {descent_v[:50]}]")

        print(f"  VERDICT: {verdict}  |  GT: {gt}")
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(dict(
            name=name, gt=gt, verdict=verdict,
            n_samples=len(samples), samples=samples,
            fr_shot_ids=sorted(fr_shot_ids), nr_shot_ids=sorted(nr_shot_ids),
        ), indent=2, default=str))
        summary.append(dict(name=name, gt=gt, verdict=verdict,
                            n_samples=len(samples)))

    # Summary
    print("\n=== TRACK-AWARE SUMMARY ===")
    tp = tn = fp = fn = und = 0
    for s in summary:
        gt = s['gt']; v = s['verdict']
        if v.startswith("UNDECIDED"): und += 1; mark = "UND"
        elif gt in MAKE_LABELS and v.startswith("MAKE"): tp += 1; mark = "✓TP"
        elif gt in MISS_LABELS and v.startswith("MISS"): tn += 1; mark = "✓TN"
        elif gt in MISS_LABELS and v.startswith("MAKE"): fp += 1; mark = "✗FP"
        elif gt in MAKE_LABELS and v.startswith("MISS"): fn += 1; mark = "✗FN"
        else: mark = "?"
        print(f"  {s['name']:14s}  GT={gt:18s} -> {v[:70]:70s}  [{mark}]")
    decided = tp+tn+fp+fn
    if decided:
        print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}  UND={und}  "
              f"Acc={100*(tp+tn)/decided:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
