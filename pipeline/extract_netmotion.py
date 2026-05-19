#!/usr/bin/env python3
"""
P1b net-motion extraction for ONE game.

The make/miss ceiling (data/P3_RESULTS_v3.md) is a representation problem: a
box-CENTER trajectory cannot separate a swish from a rattle-out. The single
highest-impact extra signal is NET MOTION just below the iron in the frames
after closest approach.

This pass derives that signal WITHOUT re-running YOLO and WITHOUT a net
detector. It reuses the per-frame rim box already produced by P1 (in
s3://.../tracks/<game_id>/tracks.parquet) to place a rim-anchored ROI, and
only decodes the video for pixels inside that ROI.

For every annotated shot, in every one of the 4 angles, over the same
buffered window P1 used, it emits per-frame net-motion energy:
  (a) framediff_energy  mean |gray(t) - gray(t-1)| inside the ROI
  (b) flow_mag          Farneback optical-flow mean magnitude inside the ROI
  (c) roi_var           ROI grayscale temporal variance contribution

Output (immutable; written once per game):
  s3://<work>/dual-fusion-v2/netmotion/<game_id>/netmotion.parquet (or .jsonl)
  s3://<work>/dual-fusion-v2/netmotion/<game_id>/netmotion_meta.json

Idempotent: if the meta object already exists the game is skipped unless
--force. Inputs (videos, tracks) are never mutated.

Usage:
  python extract_netmotion.py --game-id <uuid> [--force]
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
    ANGLES, Game, TIMESTAMP_BUFFER_SECONDS, eprint, git_sha, load_manifest,
    s3_cp, s3_exists, video_s3_uri,
)

# ---------------------------------------------------------------------------
# Net-ROI geometry constants. The net hangs BELOW the iron, so the ROI is the
# band directly under the rim box. All multipliers are relative to the rim
# box width/height so the ROI scales with apparent rim size per angle.
#
#   rim_cx      = rim_x + rim_w/2          (rim box centre x)
#   rim_bottom  = rim_y + rim_h            (rim box bottom edge y)
#   roi_x0      = rim_cx - NET_ROI_HALF_W_RIM_W * rim_w
#   roi_x1      = rim_cx + NET_ROI_HALF_W_RIM_W * rim_w
#   roi_y0      = rim_bottom + NET_ROI_TOP_OFFSET_RIM_H * rim_h
#   roi_y1      = rim_bottom + NET_ROI_BOTTOM_OFFSET_RIM_H * rim_h
# clipped to the frame.
# ---------------------------------------------------------------------------
NET_ROI_HALF_W_RIM_W = 0.6        # half-width of ROI = 0.6 * rim_w each side
NET_ROI_TOP_OFFSET_RIM_H = 0.0    # ROI starts at the rim bottom
NET_ROI_BOTTOM_OFFSET_RIM_H = 1.5  # ROI extends 1.5 * rim_h below the rim

# Downscale the ROI to keep per-frame metric work tiny + deterministic.
ROI_TARGET_W = 64

# Median smoothing window (frames) applied to rim_x/y/w/h within a shot
# window to stabilise a jittery per-frame rim box.
RIM_SMOOTH_WINDOW = 9

_NETMOTION_VERSION = "p1b-netmotion-1"


def _work_prefix() -> str:
    base = os.environ.get(
        "S3_WORK_PREFIX", "s3://uball-cv-results/cv-results/dual-fusion-v2"
    ).rstrip("/")
    return base


def netmotion_dir(game_id: str) -> str:
    return f"{_work_prefix()}/netmotion/{game_id}"


def tracks_dir(game_id: str) -> str:
    return f"{_work_prefix()}/tracks/{game_id}"


# ---------------------------------------------------------------------------
# Tracks I/O — mirror extract_tracks._emit_rows: parquet primary, jsonl
# fallback. We only need a list[dict] of rows here.
# ---------------------------------------------------------------------------

def _read_tracks(local_dir: Path) -> List[dict]:
    pq = local_dir / "tracks.parquet"
    if pq.exists():
        import pyarrow.parquet as pqt  # noqa
        table = pqt.read_table(pq)
        return table.to_pylist()
    jl = local_dir / "tracks.jsonl"
    if jl.exists():
        rows = []
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    raise FileNotFoundError(
        f"no tracks.parquet or tracks.jsonl under {local_dir}"
    )


def _download_tracks(game_id: str, dst_dir: Path) -> List[dict]:
    """Fetch the immutable P1 tracks for this game (parquet or jsonl)."""
    src_dir = tracks_dir(game_id)
    meta_uri = f"{src_dir}/tracks_meta.json"
    if not s3_exists(meta_uri):
        raise RuntimeError(
            f"P1 tracks not found for {game_id} ({meta_uri}); run P1 first"
        )
    meta_local = dst_dir / "tracks_meta.json"
    s3_cp(meta_uri, str(meta_local))
    meta = json.loads(meta_local.read_text())
    fmt = meta.get("format", "parquet")
    name = "tracks.parquet" if fmt == "parquet" else "tracks.jsonl"
    s3_cp(f"{src_dir}/{name}", str(dst_dir / name))
    return _read_tracks(dst_dir)


# ---------------------------------------------------------------------------
# Rim-box stabilisation. Pure, deterministic; no numpy needed.
# ---------------------------------------------------------------------------

def _interp_hold(vals: List[Optional[float]]) -> List[Optional[float]]:
    """Linearly interpolate interior None gaps; hold the nearest known value
    at the edges. Returns all-None unchanged (no rim anywhere in window)."""
    n = len(vals)
    known = [i for i, v in enumerate(vals) if v is not None]
    if not known:
        return list(vals)
    out: List[Optional[float]] = [None] * n
    first, last = known[0], known[-1]
    for i in range(n):
        if vals[i] is not None:
            out[i] = float(vals[i])
        elif i < first:
            out[i] = float(vals[first])
        elif i > last:
            out[i] = float(vals[last])
        else:
            lo = max(k for k in known if k <= i)
            hi = min(k for k in known if k >= i)
            if lo == hi:
                out[i] = float(vals[lo])
            else:
                t = (i - lo) / (hi - lo)
                out[i] = float(vals[lo]) + t * (float(vals[hi]) - float(vals[lo]))
    return out


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    m = len(s)
    if m == 0:
        return 0.0
    mid = m // 2
    if m % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _median_smooth(vals: List[float], window: int) -> List[float]:
    """Centered odd-window median filter (edges shrink the window)."""
    if window < 1:
        return list(vals)
    half = window // 2
    n = len(vals)
    return [
        _median(vals[max(0, i - half):min(n, i + half + 1)])
        for i in range(n)
    ]


def stabilized_rim_boxes(
    rows: List[dict],
) -> Tuple[List[Optional[Tuple[float, float, float, float]]], List[bool]]:
    """Given window rows (ordered by frame), return a stabilised
    (rim_x, rim_y, rim_w, rim_h) per frame and a per-frame rim_ok flag.

    rim_ok is True when the parquet had a non-null rim for that frame
    (i.e. the box is observed, not interpolated/held). If no frame in the
    window has a rim, every box is None and every rim_ok is False.
    """
    raw_ok = [r.get("rim_x") is not None and r.get("rim_w") is not None
              for r in rows]

    def col(key: str) -> List[Optional[float]]:
        return [
            float(r[key]) if r.get(key) is not None else None
            for r in rows
        ]

    filled = {k: _interp_hold(col(k))
              for k in ("rim_x", "rim_y", "rim_w", "rim_h")}
    if all(v is None for v in filled["rim_x"]):
        return [None] * len(rows), [False] * len(rows)

    smoothed = {
        k: _median_smooth([v if v is not None else 0.0 for v in filled[k]],
                          RIM_SMOOTH_WINDOW)
        for k in filled
    }
    boxes: List[Optional[Tuple[float, float, float, float]]] = []
    for i in range(len(rows)):
        w = smoothed["rim_w"][i]
        h = smoothed["rim_h"][i]
        if w <= 0 or h <= 0:
            boxes.append(None)
            continue
        boxes.append((smoothed["rim_x"][i], smoothed["rim_y"][i], w, h))
    return boxes, raw_ok


def net_roi_from_rim(
    rim: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Map a (rim_x, rim_y, rim_w, rim_h) box to an integer
    (roi_x, roi_y, roi_w, roi_h) net ROI just below the iron, clipped to
    the frame. Returns None if the clipped ROI is degenerate."""
    rim_x, rim_y, rim_w, rim_h = rim
    rim_cx = rim_x + rim_w / 2.0
    rim_bottom = rim_y + rim_h

    x0 = rim_cx - NET_ROI_HALF_W_RIM_W * rim_w
    x1 = rim_cx + NET_ROI_HALF_W_RIM_W * rim_w
    y0 = rim_bottom + NET_ROI_TOP_OFFSET_RIM_H * rim_h
    y1 = rim_bottom + NET_ROI_BOTTOM_OFFSET_RIM_H * rim_h

    ix0 = max(0, int(math.floor(x0)))
    iy0 = max(0, int(math.floor(y0)))
    ix1 = min(int(frame_w), int(math.ceil(x1)))
    iy1 = min(int(frame_h), int(math.ceil(y1)))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    return ix0, iy0, ix1 - ix0, iy1 - iy0


# ---------------------------------------------------------------------------
# Net-motion metrics on a small ROI stack. numpy-only; OpenCV used for the
# Farneback flow (same cv2 the v1 bundle ships).
# ---------------------------------------------------------------------------

def _downscale_roi(gray_roi, target_w: int = ROI_TARGET_W):
    """Resize an HxW grayscale ROI so width==target_w (keep aspect),
    deterministic area interpolation. Returns float32 array."""
    import cv2
    import numpy as np

    h, w = gray_roi.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), dtype="float32")
    scale = target_w / float(w)
    tw = max(1, target_w)
    th = max(1, int(round(h * scale)))
    small = cv2.resize(gray_roi, (tw, th), interpolation=cv2.INTER_AREA)
    return small.astype("float32")


def net_motion_metrics(prev_small, cur_small) -> Tuple[float, float]:
    """(framediff_energy, flow_mag) between two downscaled grayscale ROIs of
    the SAME shape. framediff_energy = mean abs pixel difference. flow_mag =
    mean Farneback optical-flow magnitude. Deterministic given the inputs."""
    import cv2
    import numpy as np

    if prev_small is None or cur_small is None:
        return 0.0, 0.0
    if prev_small.shape != cur_small.shape:
        cur_small = cv2.resize(
            cur_small, (prev_small.shape[1], prev_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype("float32")

    framediff = float(np.mean(np.abs(cur_small - prev_small)))

    flow = cv2.calcOpticalFlowFarneback(
        prev_small.astype("uint8"), cur_small.astype("uint8"),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return framediff, float(np.mean(mag))


def roi_temporal_variance(roi_means: List[float]) -> float:
    """Population variance of per-frame ROI grayscale means across the
    window — a cheap global 'did this ROI move at all' scalar. Returned
    per-row (constant within a window) so downstream can group by window."""
    import numpy as np

    if not roi_means:
        return 0.0
    return float(np.var(np.asarray(roi_means, dtype="float64")))


# ---------------------------------------------------------------------------
# Window grouping. P1 emitted one row per (play_id, angle, frame). We
# reconstruct each shot window from those rows; the buffered window itself
# is implicit in the frames P1 already extracted.
# ---------------------------------------------------------------------------

def _group_windows(rows: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        key = (str(r["play_id"]), str(r["angle"]))
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key].sort(key=lambda x: int(x["frame_idx"]))
    return groups


def _emit_rows(records: List[dict], out_path: Path) -> str:
    """Write parquet if pyarrow importable, else jsonl. Returns format.
    Mirrors extract_tracks._emit_rows exactly."""
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa

        cols: Dict[str, list] = (
            {k: [] for k in records[0].keys()} if records else {}
        )
        for r in records:
            for k, v in r.items():
                cols[k].append(v)
        table = pa.table(cols) if records else pa.table({})
        pq.write_table(table, out_path)
        return "parquet"
    except Exception as e:
        eprint(f"[netmotion] parquet unavailable ({e}); writing jsonl")
        jl = out_path.with_suffix(".jsonl")
        with open(jl, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return "jsonl"


def _process_angle(
    video_path: Path,
    angle: str,
    game: Game,
    windows: Dict[Tuple[str, str], List[dict]],
) -> Tuple[List[dict], dict]:
    """Decode ONE angle video and compute net-motion metrics for every
    shot window of that angle. Returns (records, per_angle_meta)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    angle_windows = {k: v for k, v in windows.items() if k[1] == angle}
    records: List[dict] = []
    n_windows = 0

    try:
        for (play_id, _ang), wrows in sorted(angle_windows.items()):
            n_windows += 1
            classification = wrows[0].get("classification")
            boxes, rim_ok = stabilized_rim_boxes(wrows)

            prev_small = None
            roi_means: List[float] = []
            window_out: List[dict] = []
            for i, r in enumerate(wrows):
                fidx = int(r["frame_idx"])
                ts = float(r["timestamp"])
                box = boxes[i]
                roi = (
                    net_roi_from_rim(box, frame_w, frame_h)
                    if box is not None else None
                )

                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()

                framediff = 0.0
                flow_mag = 0.0
                roi_mean = 0.0
                if ok and roi is not None:
                    rx, ry, rw, rh = roi
                    gray = cv2.cvtColor(
                        frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY
                    )
                    cur_small = _downscale_roi(gray)
                    roi_mean = float(cur_small.mean())
                    framediff, flow_mag = net_motion_metrics(
                        prev_small, cur_small
                    )
                    prev_small = cur_small
                else:
                    prev_small = None  # break diff chain across gaps

                roi_means.append(roi_mean)
                window_out.append({
                    "game_id": game.game_id,
                    "play_id": play_id,
                    "classification": classification,
                    "angle": angle,
                    "frame_idx": fidx,
                    "timestamp": round(ts, 6),
                    "roi_x": roi[0] if roi else None,
                    "roi_y": roi[1] if roi else None,
                    "roi_w": roi[2] if roi else None,
                    "roi_h": roi[3] if roi else None,
                    "framediff_energy": round(framediff, 6),
                    "flow_mag": round(flow_mag, 6),
                    "rim_ok": bool(rim_ok[i]),
                })

            var = roi_temporal_variance(roi_means)
            for row in window_out:
                row["roi_var"] = round(var, 6)
            records.extend(window_out)
    finally:
        cap.release()

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "width": frame_w,
        "height": frame_h,
        "n_windows": n_windows,
        "emitted_frames": len(records),
    }
    return records, meta


def extract_game(game: Game, force: bool = False) -> dict:
    out_dir = netmotion_dir(game.game_id)
    meta_uri = f"{out_dir}/netmotion_meta.json"

    if not force and s3_exists(meta_uri):
        eprint(
            f"[netmotion] {game.game_id}: output exists, skipping "
            f"(use --force)"
        )
        return {"game_id": game.game_id, "status": "skipped",
                "s3_dir": out_dir}

    bucket = os.environ.get("UPLOAD_BUCKET", "uball-videos-production")
    records: List[dict] = []
    per_angle_meta: Dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="p1bnet_") as td:
        tdir = Path(td)
        rows = _download_tracks(game.game_id, tdir)
        windows = _group_windows(rows)
        if not windows:
            raise RuntimeError(
                f"P1 tracks for {game.game_id} contain no shot windows"
            )

        for angle in ANGLES:
            src = video_s3_uri(game, angle, bucket)
            local = tdir / f"{game.game23}_{angle}.mp4"
            if not s3_exists(src):
                raise RuntimeError(f"missing source video: {src}")
            eprint(
                f"[netmotion] {game.game_id} {angle}: downloading video"
            )
            s3_cp(src, str(local))
            arecs, ameta = _process_angle(local, angle, game, windows)
            records.extend(arecs)
            per_angle_meta[angle] = ameta
            # Free the cached video promptly (4 angles can be large).
            try:
                local.unlink()
            except OSError:
                pass

        out_local = tdir / "netmotion.parquet"
        fmt = _emit_rows(records, out_local)
        produced = (
            out_local if fmt == "parquet"
            else out_local.with_suffix(".jsonl")
        )
        out_name = (
            "netmotion.parquet" if fmt == "parquet" else "netmotion.jsonl"
        )
        out_uri = f"{out_dir}/{out_name}"

        meta = {
            "game_id": game.game_id,
            "split": game.split,
            "format": fmt,
            "netmotion_version": _NETMOTION_VERSION,
            "netmotion_s3_key": out_uri,
            "tracks_s3_dir": tracks_dir(game.game_id),
            "n_records": len(records),
            "n_windows": len(windows),
            "timestamp_buffer_seconds": TIMESTAMP_BUFFER_SECONDS,
            "roi_constants": {
                "net_roi_half_w_rim_w": NET_ROI_HALF_W_RIM_W,
                "net_roi_top_offset_rim_h": NET_ROI_TOP_OFFSET_RIM_H,
                "net_roi_bottom_offset_rim_h": NET_ROI_BOTTOM_OFFSET_RIM_H,
                "roi_target_w": ROI_TARGET_W,
                "rim_smooth_window": RIM_SMOOTH_WINDOW,
            },
            "git_sha": git_sha(),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "per_angle": per_angle_meta,
        }
        meta_local = tdir / "netmotion_meta.json"
        meta_local.write_text(json.dumps(meta, indent=2))

        s3_cp(str(produced), out_uri)
        s3_cp(str(meta_local), meta_uri)

    eprint(
        f"[netmotion] {game.game_id}: wrote {len(records)} rows -> {out_uri}"
    )
    return {"game_id": game.game_id, "status": "done",
            "s3_dir": out_dir, "netmotion_s3_key": out_uri,
            "n_records": len(records), "meta": meta}


def _find_game(game_id: str) -> Game:
    for g in load_manifest():
        if g.game_id == game_id:
            return g
    raise SystemExit(
        f"game_id {game_id!r} not in data/games_manifest.json"
    )


def main(argv: Optional[List[str]] = None) -> int:
    from common import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="P1b per-game net-motion extraction"
    )
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
