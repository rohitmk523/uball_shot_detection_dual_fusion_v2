#!/usr/bin/env python3
"""
P1 track extraction for ONE game.

For a game_id: pull its 4 angle videos from S3, run the FROZEN v1 detector
(imported from the verified bundle — YOLO is reused, never reimplemented),
and for every annotated shot window (from the same Supabase GT v1 uses)
emit per-frame ball/rim records.

Output (immutable; written once per game):
  s3://<work>/dual-fusion-v2/tracks/<game_id>/tracks.parquet   (or .jsonl)
  s3://<work>/dual-fusion-v2/tracks/<game_id>/tracks_meta.json

Idempotent: if the meta object already exists the game is skipped unless
--force is given. Inputs (videos, bundle) are never mutated.

Usage:
  python extract_tracks.py --game-id <uuid> [--force]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from common import (
    ANGLE_TO_DETECTOR, ANGLES, Game, eprint, git_sha, load_gt_shots,
    load_manifest, s3_cp, s3_exists, video_s3_uri,
)
from frozen_bundle import (
    fetch_and_verify_bundle, import_v1, weight_paths,
)


def _work_prefix() -> str:
    base = os.environ.get(
        "S3_WORK_PREFIX", "s3://uball-cv-results/cv-results/dual-fusion-v2"
    ).rstrip("/")
    return base


def tracks_dir(game_id: str) -> str:
    return f"{_work_prefix()}/tracks/{game_id}"


def _det_to_record(det) -> Dict[str, float]:
    """v1 Detection -> xywh+conf (x,y = top-left; w,h = box size)."""
    if det is None:
        return {"x": None, "y": None, "w": None, "h": None, "conf": None}
    x1, y1, x2, y2 = det.bbox
    return {
        "x": float(x1), "y": float(y1),
        "w": float(x2 - x1), "h": float(y2 - y1),
        "conf": float(det.confidence),
    }


def _emit_rows(records: List[dict], out_path: Path) -> str:
    """Write parquet if pyarrow is importable, else jsonl. Returns format."""
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa

        cols: Dict[str, list] = {k: [] for k in records[0].keys()} if records else {}
        for r in records:
            for k, v in r.items():
                cols[k].append(v)
        table = pa.table(cols) if records else pa.table({})
        pq.write_table(table, out_path)
        return "parquet"
    except Exception as e:  # pyarrow missing or write failed -> jsonl fallback
        eprint(f"[extract] parquet unavailable ({e}); writing jsonl")
        jl = out_path.with_suffix(".jsonl")
        with open(jl, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return "jsonl"


def extract_game(game: Game, force: bool = False) -> dict:
    out_dir = tracks_dir(game.game_id)
    meta_uri = f"{out_dir}/tracks_meta.json"

    if not force and s3_exists(meta_uri):
        eprint(f"[extract] {game.game_id}: output exists, skipping (use --force)")
        return {"game_id": game.game_id, "status": "skipped",
                "s3_dir": out_dir}

    bundle_dir, bundle_sha = fetch_and_verify_bundle()
    v1_config, VideoProcessor, EnhancedShotDetector = import_v1(bundle_dir)
    v1_config.VERBOSE_LOGGING = False
    weights = weight_paths(bundle_dir)

    # Frozen GT windows (same source/semantics as v1).
    gt_shots = load_gt_shots(game.game_id)
    if not gt_shots:
        raise RuntimeError(
            f"no ground-truth shot windows for game {game.game_id}; aborting"
        )

    bucket = os.environ.get("UPLOAD_BUCKET", "uball-videos-production")

    # Lazily-built detectors, one per detector kind (near/far) — frozen.
    detectors: Dict[str, object] = {}

    def detector_for(angle: str):
        kind = ANGLE_TO_DETECTOR[angle]
        if kind not in detectors:
            detectors[kind] = EnhancedShotDetector(weights[kind], kind)
        return detectors[kind]

    records: List[dict] = []
    per_angle_meta: Dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="p1vid_") as td:
        for angle in ANGLES:
            src = video_s3_uri(game, angle, bucket)
            local = Path(td) / f"{game.game23}_{angle}.mp4"
            if not s3_exists(src):
                raise RuntimeError(f"missing source video: {src}")
            eprint(f"[extract] {game.game_id} {angle}: downloading video")
            s3_cp(src, str(local))

            vp = VideoProcessor(str(local), offset=0.0,
                                angle=ANGLE_TO_DETECTOR[angle])
            det = detector_for(angle)
            fps = float(vp.fps)
            windows = []
            n_frames_angle = 0

            try:
                for shot in gt_shots:
                    w_start, w_end = shot.buffered_window()
                    frames = list(vp.extract_frames_in_window(
                        w_start, w_end, frame_skip=1))
                    windows.append({
                        "play_id": shot.play_id,
                        "classification": shot.classification,
                        "gt_angle": shot.angle,
                        "window_start_s": round(w_start, 4),
                        "window_end_s": round(w_end, 4),
                        "frames": len(frames),
                    })
                    if not frames:
                        continue
                    fdets = det.detect_in_frames(frames)
                    for fd, fr in zip(fdets, frames):
                        ball = fd.get_primary_ball()
                        rim = fd.get_primary_hoop()
                        b = _det_to_record(ball)
                        r = _det_to_record(rim)
                        records.append({
                            "game_id": game.game_id,
                            "play_id": shot.play_id,
                            "classification": shot.classification,
                            "angle": angle,
                            "detector": ANGLE_TO_DETECTOR[angle],
                            "frame_idx": int(fr.frame_number),
                            "timestamp": round(float(fr.timestamp), 6),
                            "ball_x": b["x"], "ball_y": b["y"],
                            "ball_w": b["w"], "ball_h": b["h"],
                            "ball_conf": b["conf"],
                            "rim_x": r["x"], "rim_y": r["y"],
                            "rim_w": r["w"], "rim_h": r["h"],
                            "rim_conf": r["conf"],
                        })
                        n_frames_angle += 1
            finally:
                vp.release()

            per_angle_meta[angle] = {
                "fps": fps,
                "total_frames": int(vp.total_frames),
                "width": int(vp.width),
                "height": int(vp.height),
                "detector": ANGLE_TO_DETECTOR[angle],
                "emitted_frames": n_frames_angle,
                "shot_windows": windows,
            }

        # Write outputs to a temp file, then push immutably to S3.
        out_local = Path(td) / "tracks.parquet"
        fmt = _emit_rows(records, out_local)
        produced = out_local if fmt == "parquet" else out_local.with_suffix(".jsonl")
        out_name = "tracks.parquet" if fmt == "parquet" else "tracks.jsonl"
        out_uri = f"{out_dir}/{out_name}"

        meta = {
            "game_id": game.game_id,
            "split": game.split,
            "format": fmt,
            "tracks_s3_key": out_uri,
            "n_records": len(records),
            "n_shot_windows": len(gt_shots),
            "frozen_bundle_s3": (
                "s3://uball-cv-results/cv-results/dual-fusion-v2/"
                "frozen_detector_v16/frozen_detector_v16.tar.gz"
            ),
            "frozen_bundle_sha256": bundle_sha,
            "git_sha": git_sha(),
            "timestamp_buffer_seconds": 2.0,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "per_angle": per_angle_meta,
        }
        meta_local = Path(td) / "tracks_meta.json"
        meta_local.write_text(json.dumps(meta, indent=2))

        s3_cp(str(produced), out_uri)
        s3_cp(str(meta_local), meta_uri)

    eprint(f"[extract] {game.game_id}: wrote {len(records)} rows -> {out_uri}")
    return {"game_id": game.game_id, "status": "done",
            "s3_dir": out_dir, "tracks_s3_key": out_uri,
            "n_records": len(records), "meta": meta}


def _find_game(game_id: str) -> Game:
    for g in load_manifest():
        if g.game_id == game_id:
            return g
    raise SystemExit(f"game_id {game_id!r} not in data/games_manifest.json")


def main(argv: Optional[List[str]] = None) -> int:
    from common import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(description="P1 per-game track extraction")
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if output already exists")
    args = ap.parse_args(argv)

    game = _find_game(args.game_id)
    result = extract_game(game, force=args.force)
    print(json.dumps(result.get("meta", result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
