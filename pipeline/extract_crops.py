#!/usr/bin/env python3
"""
P1c rim-region crop extraction for ONE game.

Hypothesis under test: a PIXEL model on rim-region crops can separate a swish
from a rattle-out (and make from miss) better than a box-feature model, which
only ever sees ball/rim centre trajectories. This pass produces the crops that
feed p3_cropmodel.py.

Like extract_netmotion.py, it reuses the per-frame rim box already produced by
P1 (s3://.../tracks/<game_id>/tracks.parquet) — it never re-runs YOLO for the
crops themselves — and decodes the video ONLY for pixels inside a rim-anchored
ROI, with a forward-only sequential cursor (per-frame cap.set is pathologically
slow on H.264).

For every annotated shot, in every one of the 4 angles, it:
  1. stabilises the rim box (median filter, exactly as extract_netmotion does),
  2. defines a crop ROI centred on the rim covering rim + net + approach zone,
  3. samples a FIXED-LENGTH window of T frames centred on the ball's
     closest-approach-to-rim frame (fallback: centre of the GT window),
  4. resizes each crop to 64x64 grayscale and stacks -> uint8 [T,64,64].

Output (immutable; written once per game):
  s3://<work>/dual-fusion-v2/crops/<game_id>/crops.npz       (compressed)
  s3://<work>/dual-fusion-v2/crops/<game_id>/crops_meta.json

  crops.npz keys: f"{play_id}_{angle}" -> uint8 [T,64,64]
  crops_meta.json: play_id -> {label, classification, split, angles:[...]}

Idempotent: if crops_meta.json already exists the game is skipped unless
--force. Inputs (videos, tracks) are never mutated.

Optional (#2 secondary): --redetect-conf 0.10 re-runs the frozen YOLO at the
lower confidence over the SAME windows and writes denser ball boxes to
crops/<game_id>/redetect.parquet. Non-fatal if torch/ultralytics/weights are
missing.

Usage:
  python extract_crops.py --game-id <uuid> [--force] [--redetect-conf 0.10]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import (
    ANGLES, Game, SHOT_MAKE_CLASSES, TIMESTAMP_BUFFER_SECONDS, eprint,
    git_sha, load_manifest, s3_cp, s3_exists, video_s3_uri,
)
# Reuse the EXACT rim-stabilisation + sequential-decode helpers from the
# net-motion pass so the two passes share one geometry/decoding contract.
from extract_netmotion import _download_tracks, _group_windows, stabilized_rim_boxes

# ---------------------------------------------------------------------------
# Crop-ROI geometry constants. The crop is centred on the rim and covers the
# rim + the net hanging below + an approach zone above the iron (where the
# ball comes in). All multipliers are relative to the rim box width/height so
# the ROI scales with apparent rim size per angle.
#
#   rim_cx     = rim_x + rim_w/2          (rim box centre x)
#   rim_top    = rim_y                    (rim box top edge y)
#   rim_bottom = rim_y + rim_h            (rim box bottom edge y)
#   roi_x0     = rim_cx - CROP_HALF_W_RIM_W * rim_w
#   roi_x1     = rim_cx + CROP_HALF_W_RIM_W * rim_w
#   roi_y0     = rim_top    - CROP_TOP_ABOVE_RIM_H    * rim_h   (approach zone)
#   roi_y1     = rim_bottom + CROP_BOTTOM_BELOW_RIM_H * rim_h   (net zone)
# clipped to the frame.
#
# width  ~= 3.0 * rim_w  (2 * 1.5)
# height ~= 1.5*rim_h above rim top + rim_h + 2.5*rim_h below rim bottom
#         = 5.0 * rim_h
# ---------------------------------------------------------------------------
CROP_HALF_W_RIM_W = 1.5         # half-width = 1.5 * rim_w each side -> 3.0*rim_w
CROP_TOP_ABOVE_RIM_H = 1.5      # extend 1.5 * rim_h above the rim TOP (approach)
CROP_BOTTOM_BELOW_RIM_H = 2.5   # extend 2.5 * rim_h below the rim BOTTOM (net)

# Fixed temporal window length and output spatial size.
CROP_T = 16
CROP_SIZE = 64

# Median smoothing window (frames) for the rim box — same default as the
# net-motion pass (imported stabilized_rim_boxes uses its own constant).

_CROPS_VERSION = "p1c-crops-1"


def _work_prefix() -> str:
    base = os.environ.get(
        "S3_WORK_PREFIX", "s3://uball-cv-results/cv-results/dual-fusion-v2"
    ).rstrip("/")
    return base


def crops_dir(game_id: str) -> str:
    return f"{_work_prefix()}/crops/{game_id}"


# ---------------------------------------------------------------------------
# Crop-ROI geometry. Pure + deterministic (no numpy / no cv2). Mirrors
# extract_netmotion.net_roi_from_rim's clip-and-validate contract.
# ---------------------------------------------------------------------------

def crop_roi_from_rim(
    rim: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Map a (rim_x, rim_y, rim_w, rim_h) box to an integer
    (roi_x, roi_y, roi_w, roi_h) crop covering rim + approach + net, clipped
    to the frame. Returns None if the clipped ROI is degenerate (<2px)."""
    rim_x, rim_y, rim_w, rim_h = rim
    rim_cx = rim_x + rim_w / 2.0
    rim_top = rim_y
    rim_bottom = rim_y + rim_h

    x0 = rim_cx - CROP_HALF_W_RIM_W * rim_w
    x1 = rim_cx + CROP_HALF_W_RIM_W * rim_w
    y0 = rim_top - CROP_TOP_ABOVE_RIM_H * rim_h
    y1 = rim_bottom + CROP_BOTTOM_BELOW_RIM_H * rim_h

    ix0 = max(0, int(math.floor(x0)))
    iy0 = max(0, int(math.floor(y0)))
    ix1 = min(int(frame_w), int(math.ceil(x1)))
    iy1 = min(int(frame_h), int(math.ceil(y1)))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    return ix0, iy0, ix1 - ix0, iy1 - iy0


# ---------------------------------------------------------------------------
# Closest-approach detection + fixed-window sampling. Pure helpers operating
# on the parquet rows; no video needed (so they unit-test without GPU/AWS).
# ---------------------------------------------------------------------------

def _ball_rim_distance(row: dict) -> Optional[float]:
    """Centre-to-centre distance between the ball and rim boxes for a row,
    or None if either box is missing for that frame."""
    bx, by, bw, bh = (row.get("ball_x"), row.get("ball_y"),
                      row.get("ball_w"), row.get("ball_h"))
    rx, ry, rw, rh = (row.get("rim_x"), row.get("rim_y"),
                      row.get("rim_w"), row.get("rim_h"))
    if None in (bx, by, bw, bh, rx, ry, rw, rh):
        return None
    bcx = float(bx) + float(bw) / 2.0
    bcy = float(by) + float(bh) / 2.0
    rcx = float(rx) + float(rw) / 2.0
    rcy = float(ry) + float(rh) / 2.0
    return math.hypot(bcx - rcx, bcy - rcy)


def closest_approach_index(wrows: List[dict]) -> Tuple[int, bool]:
    """Index (into wrows) of the frame where the ball is closest to the rim
    centre. Returns (index, ball_detected). If the ball is never detected in
    the window, falls back to the centre frame and ball_detected=False."""
    n = len(wrows)
    if n == 0:
        return 0, False
    best_i: Optional[int] = None
    best_d = float("inf")
    for i, r in enumerate(wrows):
        d = _ball_rim_distance(r)
        if d is not None and d < best_d:
            best_d = d
            best_i = i
    if best_i is None:
        return n // 2, False
    return best_i, True


def sample_window_indices(center: int, n: int, t: int = CROP_T) -> List[int]:
    """Indices of T frames centred on `center` within [0, n). Clamps the
    window to the available range while keeping length exactly T (so the
    output tensor is always [T,...]). If n < t, indices are clamped into
    range (duplicates allowed) so the contract holds for short windows."""
    if n <= 0:
        return [0] * t
    half = t // 2
    start = center - half
    # Shift the window so it stays inside [0, n) when possible.
    if start < 0:
        start = 0
    if start + t > n:
        start = max(0, n - t)
    idxs = [start + k for k in range(t)]
    # For windows shorter than T, clamp out-of-range indices to the last frame.
    return [min(i, n - 1) for i in idxs]


def make_label(classification: Optional[str]) -> int:
    """make(1) if classification in SHOT_MAKE_CLASSES else miss(0)."""
    return 1 if classification in SHOT_MAKE_CLASSES else 0


# ---------------------------------------------------------------------------
# Crop decoding for one angle. cv2 only used here (kept out of the pure
# helpers above so the geometry/sampler tests run with no OpenCV).
# ---------------------------------------------------------------------------

def _to_gray_64(frame_bgr, roi: Tuple[int, int, int, int]):
    """Crop a BGR frame to `roi`, convert to grayscale, resize to 64x64.
    Returns a uint8 [64,64] array. Deterministic area interpolation."""
    import cv2
    import numpy as np

    rx, ry, rw, rh = roi
    sub = frame_bgr[ry:ry + rh, rx:rx + rw]
    if sub.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype="uint8")
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA
    )
    return small.astype("uint8")


def _process_angle(
    video_path: Path,
    angle: str,
    windows: Dict[Tuple[str, str], List[dict]],
) -> Tuple[Dict[str, "object"], dict]:
    """Decode ONE angle video and produce a [T,64,64] uint8 crop stack for
    every shot window of that angle. Returns (arrays, per_angle_meta) where
    arrays maps f"{play_id}_{angle}" -> np.ndarray[T,64,64] uint8."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    angle_windows = {k: v for k, v in windows.items() if k[1] == angle}
    arrays: Dict[str, object] = {}
    n_windows = 0
    n_ball_detected = 0

    # Forward-only sequential cursor — identical contract to extract_netmotion.
    _cur = 0

    def _read_at(target: int):
        nonlocal _cur
        if target < _cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            _cur = target
        while _cur < target:
            if not cap.grab():
                return False, None
            _cur += 1
        ok, frame = cap.read()
        _cur += 1
        return ok, frame

    try:
        # Order windows by frame index so the cursor advances monotonically.
        ordered = sorted(
            angle_windows.items(),
            key=lambda kv: int(kv[1][0]["frame_idx"]),
        )
        for (play_id, _ang), wrows in ordered:
            wrows = sorted(wrows, key=lambda r: int(r["frame_idx"]))
            n_windows += 1
            boxes, _rim_ok = stabilized_rim_boxes(wrows)

            center, ball_ok = closest_approach_index(wrows)
            if ball_ok:
                n_ball_detected += 1
            sample_idxs = sample_window_indices(center, len(wrows), CROP_T)

            stack = np.zeros((CROP_T, CROP_SIZE, CROP_SIZE), dtype="uint8")
            for t_pos, wi in enumerate(sample_idxs):
                r = wrows[wi]
                fidx = int(r["frame_idx"])
                box = boxes[wi]
                roi = (
                    crop_roi_from_rim(box, frame_w, frame_h)
                    if box is not None else None
                )
                ok, frame = _read_at(fidx)
                if ok and roi is not None:
                    stack[t_pos] = _to_gray_64(frame, roi)
                # else: leave zeros (missing rim or unreadable frame)
            arrays[f"{play_id}_{angle}"] = stack
    finally:
        cap.release()

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "width": frame_w,
        "height": frame_h,
        "n_windows": n_windows,
        "n_ball_detected": n_ball_detected,
    }
    return arrays, meta


# ---------------------------------------------------------------------------
# Optional (#2): denser re-detection at a lower confidence via the frozen
# YOLO. Entirely non-fatal — any import/weight/runtime failure logs and
# returns no rows. Schema is the tracks subset the caller asked for.
# ---------------------------------------------------------------------------

def _redetect_window(
    detector, frames, play_id: str, angle: str, conf: float,
) -> List[dict]:
    """Run the frozen detector over already-decoded frames at a lower conf,
    emit one row per frame with ball + rim boxes. Best-effort."""
    rows: List[dict] = []
    fdets = detector.detect_in_frames(frames)
    for fd, fr in zip(fdets, frames):
        ball = fd.get_primary_ball()
        rim = fd.get_primary_hoop()

        def _xywh(det):
            if det is None or float(getattr(det, "confidence", 0.0)) < conf:
                return (None, None, None, None, None)
            x1, y1, x2, y2 = det.bbox
            return (float(x1), float(y1), float(x2 - x1), float(y2 - y1),
                    float(det.confidence))

        bx, by, bw, bh, bc = _xywh(ball)
        rx, ry, rw, rh, rc = _xywh(rim)
        rows.append({
            "play_id": play_id, "angle": angle,
            "frame_idx": int(fr.frame_number),
            "ball_x": bx, "ball_y": by, "ball_w": bw, "ball_h": bh,
            "ball_conf": bc,
            "rim_x": rx, "rim_y": ry, "rim_w": rw, "rim_h": rh,
            "rim_conf": rc,
        })
    return rows


def _emit_redetect(records: List[dict], out_path: Path) -> Optional[str]:
    """Write redetect rows as parquet (best-effort jsonl fallback). Returns
    the format, or None if there is nothing to write."""
    if not records:
        return None
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa

        cols: Dict[str, list] = {k: [] for k in records[0].keys()}
        for r in records:
            for k, v in r.items():
                cols[k].append(v)
        pq.write_table(pa.table(cols), out_path)
        return "parquet"
    except Exception as e:
        eprint(f"[crops] redetect parquet unavailable ({e}); writing jsonl")
        jl = out_path.with_suffix(".jsonl")
        with open(jl, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return "jsonl"


def _try_redetect_all(
    game: Game,
    bucket: str,
    tdir: Path,
    windows: Dict[Tuple[str, str], List[dict]],
    conf: float,
) -> Tuple[List[dict], Optional[str]]:
    """Optional dense re-detection. Returns (records, error_or_None). Any
    failure (no torch/ultralytics, no weights, decode error) is swallowed
    into the returned error string so the main crop pass is never blocked."""
    try:
        from common import ANGLE_TO_DETECTOR
        from frozen_bundle import (
            fetch_and_verify_bundle, import_v1, weight_paths,
        )
    except Exception as e:  # pragma: no cover - import-time env dependent
        return [], f"redetect imports unavailable: {e}"

    try:
        bundle_dir, _sha = fetch_and_verify_bundle()
        _cfg, VideoProcessor, EnhancedShotDetector = import_v1(bundle_dir)
        _cfg.VERBOSE_LOGGING = False
        weights = weight_paths(bundle_dir)
    except Exception as e:
        return [], f"redetect bundle unavailable: {e}"

    records: List[dict] = []
    detectors: Dict[str, object] = {}

    def detector_for(angle: str):
        kind = ANGLE_TO_DETECTOR[angle]
        if kind not in detectors:
            detectors[kind] = EnhancedShotDetector(weights[kind], kind)
        return detectors[kind]

    try:
        for angle in ANGLES:
            angle_windows = {k: v for k, v in windows.items() if k[1] == angle}
            if not angle_windows:
                continue
            src = video_s3_uri(game, angle, bucket)
            local = tdir / f"redetect_{game.game23}_{angle}.mp4"
            if not s3_exists(src):
                eprint(f"[crops] redetect: missing video {src}, skipping angle")
                continue
            s3_cp(src, str(local))
            vp = VideoProcessor(str(local), offset=0.0,
                                angle=ANGLE_TO_DETECTOR[angle])
            det = detector_for(angle)
            try:
                for (play_id, _ang), wrows in angle_windows.items():
                    wrows = sorted(wrows, key=lambda r: int(r["frame_idx"]))
                    # Decode the same window span the tracks already covered.
                    start_s = float(wrows[0]["timestamp"])
                    end_s = float(wrows[-1]["timestamp"])
                    frames = list(vp.extract_frames_in_window(
                        start_s, end_s, frame_skip=1))
                    if not frames:
                        continue
                    records.extend(_redetect_window(
                        det, frames, play_id, angle, conf))
            finally:
                vp.release()
            try:
                local.unlink()
            except OSError:
                pass
    except Exception as e:
        return records, f"redetect runtime error: {e}"
    return records, None


# ---------------------------------------------------------------------------
# Per-game extraction.
# ---------------------------------------------------------------------------

def extract_game(
    game: Game, force: bool = False, redetect_conf: Optional[float] = None,
) -> dict:
    out_dir = crops_dir(game.game_id)
    meta_uri = f"{out_dir}/crops_meta.json"

    if not force and s3_exists(meta_uri):
        eprint(
            f"[crops] {game.game_id}: output exists, skipping (use --force)"
        )
        return {"game_id": game.game_id, "status": "skipped", "s3_dir": out_dir}

    import numpy as np

    bucket = os.environ.get("UPLOAD_BUCKET", "uball-videos-production")
    all_arrays: Dict[str, object] = {}
    per_angle_meta: Dict[str, dict] = {}
    play_meta: Dict[str, dict] = {}
    redetect_status: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="p1ccrop_") as td:
        tdir = Path(td)
        rows = _download_tracks(game.game_id, tdir)
        windows = _group_windows(rows)
        if not windows:
            raise RuntimeError(
                f"P1 tracks for {game.game_id} contain no shot windows"
            )

        # Build per-play metadata once (label/classification independent of
        # angle; the same play_id appears under every angle key).
        for (play_id, _angle), wrows in windows.items():
            cls = wrows[0].get("classification")
            if play_id not in play_meta:
                play_meta[play_id] = {
                    "label": make_label(cls),
                    "classification": cls,
                    "split": game.split,
                    "angles": [],
                }

        for angle in ANGLES:
            src = video_s3_uri(game, angle, bucket)
            local = tdir / f"{game.game23}_{angle}.mp4"
            if not s3_exists(src):
                raise RuntimeError(f"missing source video: {src}")
            eprint(f"[crops] {game.game_id} {angle}: downloading video")
            s3_cp(src, str(local))
            arrays, ameta = _process_angle(local, angle, windows)
            all_arrays.update(arrays)
            per_angle_meta[angle] = ameta
            for key in arrays:
                pid = key[: -(len(angle) + 1)]  # strip "_<angle>"
                if pid in play_meta and angle not in play_meta[pid]["angles"]:
                    play_meta[pid]["angles"].append(angle)
            try:
                local.unlink()
            except OSError:
                pass

        # Optional dense re-detection (best-effort, non-fatal).
        if redetect_conf is not None:
            eprint(
                f"[crops] {game.game_id}: redetect at conf={redetect_conf}"
            )
            rd_records, rd_err = _try_redetect_all(
                game, bucket, tdir, windows, float(redetect_conf))
            if rd_err:
                eprint(f"[crops] redetect skipped/partial: {rd_err}")
                redetect_status = rd_err
            if rd_records:
                rd_local = tdir / "redetect.parquet"
                rd_fmt = _emit_redetect(rd_records, rd_local)
                if rd_fmt:
                    rd_produced = (
                        rd_local if rd_fmt == "parquet"
                        else rd_local.with_suffix(".jsonl")
                    )
                    rd_name = (
                        "redetect.parquet" if rd_fmt == "parquet"
                        else "redetect.jsonl"
                    )
                    s3_cp(str(rd_produced), f"{out_dir}/{rd_name}")
                    redetect_status = (
                        f"ok:{len(rd_records)} rows"
                        if not redetect_status else redetect_status
                    )

        # Write the compressed npz + meta, then push immutably to S3.
        out_local = tdir / "crops.npz"
        np.savez_compressed(out_local, **all_arrays)

        meta = {
            "game_id": game.game_id,
            "split": game.split,
            "crops_version": _CROPS_VERSION,
            "crops_s3_key": f"{out_dir}/crops.npz",
            "tracks_s3_dir": f"{_work_prefix()}/tracks/{game.game_id}",
            "n_arrays": len(all_arrays),
            "n_plays": len(play_meta),
            "T": CROP_T,
            "crop_size": CROP_SIZE,
            "timestamp_buffer_seconds": TIMESTAMP_BUFFER_SECONDS,
            "roi_constants": {
                "crop_half_w_rim_w": CROP_HALF_W_RIM_W,
                "crop_top_above_rim_h": CROP_TOP_ABOVE_RIM_H,
                "crop_bottom_below_rim_h": CROP_BOTTOM_BELOW_RIM_H,
            },
            "redetect_conf": redetect_conf,
            "redetect_status": redetect_status,
            "git_sha": git_sha(),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "per_angle": per_angle_meta,
            "plays": play_meta,
        }
        meta_local = tdir / "crops_meta.json"
        meta_local.write_text(json.dumps(meta, indent=2))

        s3_cp(str(out_local), f"{out_dir}/crops.npz")
        s3_cp(str(meta_local), meta_uri)

    eprint(
        f"[crops] {game.game_id}: wrote {len(all_arrays)} crop stacks "
        f"({len(play_meta)} plays) -> {out_dir}/crops.npz"
    )
    return {"game_id": game.game_id, "status": "done", "s3_dir": out_dir,
            "crops_s3_key": f"{out_dir}/crops.npz",
            "n_arrays": len(all_arrays), "meta": meta}


def _find_game(game_id: str) -> Game:
    for g in load_manifest():
        if g.game_id == game_id:
            return g
    raise SystemExit(f"game_id {game_id!r} not in data/games_manifest.json")


def main(argv: Optional[List[str]] = None) -> int:
    from common import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="P1c per-game rim-region crop extraction"
    )
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if output already exists")
    ap.add_argument(
        "--redetect-conf", type=float, default=None,
        help="if set, re-run frozen YOLO at this conf over the same windows "
             "and write crops/<game_id>/redetect.parquet (best-effort)",
    )
    args = ap.parse_args(argv)

    game = _find_game(args.game_id)
    result = extract_game(
        game, force=args.force, redetect_conf=args.redetect_conf)
    print(json.dumps(result.get("meta", result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
