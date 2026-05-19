#!/usr/bin/env python3
"""
P2 — per-shot make/miss feature dataset.

For every (game_id, play_id) we build ONE feature row from the buffered
GT window of per-frame ball/rim tracks across the 4 camera angles
(FL, FR, NL, NR). Features are physically meaningful and interpretable:
they describe the geometry of the ball relative to the (near-static) rim
over time, per angle, then fuse the 4 angles.

Coordinates: (x, y) is the box top-left, (w, h) the box size, pixels in a
1920x1080 frame. Ball/rim conf is NaN when the object was not detected
that frame.

Iteration 2 adds three interpretable, precision-focused lever groups
(all kept on top of the v1 features so we can ablate):
  1. Rim-out / bounce-out recovery (`bo_*`): after closest approach,
     detect the ball reappearing BELOW + OUTSIDE the rim then leaving
     -> a rim-in-and-out MISS cue that v1 lacked.
  2. Parametric trajectory-arc fit (`arc_*`): per angle fit a parabola
     y(x) and y(t) in a window around the rim; entry angle, vertical
     velocity at the rim plane, curvature, fit residual, apex offset.
  3. Per-angle quality-gated fusion (`q_*`): angle quality = ball-det
     fraction * rim-detection stability; quality-weighted aggregates,
     a >=2-angle agreement gate, and far-cam down-weighting when the
     far ball track leaves frame (huge spurious rebound).

Iteration 3 adds a feature-source switch. With ``--version v3`` the L1
(bounce-out) and L2 (arc-fit) trajectory levers are recomputed on the
KALMAN-CLEANED ball track (pipeline/track_clean.py) instead of the raw
detector centers. The raw-track versions are KEPT alongside (suffixed
``_raw``) so the model can use both / ablate, and per-angle imputation
fraction + track-quality features let the model discount unreliable
shots. v1/v2 parquet are never touched — v3 writes its own files.

Output (idempotent, immutable per version):
  data/p2_features.parquet      v1 features (UNTOUCHED)
  data/p2_features_v2.parquet   v1 features + iteration-2 levers (UNTOUCHED)
  data/P2_FEATURES_v2.md        v2 dictionary (UNTOUCHED)
  data/p2_features_v3.parquet   v2 features + cleaned-track L1/L2 + raw
                                copies + imputation/quality features
  data/P2_FEATURES_v3.md        v3 feature dictionary + rationale

Local CPU only. No AWS, no GPU. Reads cached parquet from
/tmp/p1tracks/<game_id>.parquet (P1 artifacts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ANGLES,
    SHOT_MAKE_CLASSES,
    SHOT_MISS_CLASSES,
    V1_ALL_SHOT_CLASSES,
    eprint,
    load_manifest,
)
from track_clean import clean_track, jitter_metric  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKS_CACHE = Path("/tmp/p1tracks")
# v1 artifact is immutable — we write a NEW v2 file and never touch v1.
OUT_PARQUET = REPO_ROOT / "data" / "p2_features_v2.parquet"
OUT_DOC = REPO_ROOT / "data" / "P2_FEATURES_v2.md"
# Iteration-3 immutable outputs (v1/v2 never touched).
OUT_PARQUET_V3 = REPO_ROOT / "data" / "p2_features_v3.parquet"
OUT_DOC_V3 = REPO_ROOT / "data" / "P2_FEATURES_v3.md"

# --- Iteration-2 lever knobs (deterministic; documented) ---------------
# Lever 1: how many frames after closest approach to search for a
# below-and-outside-rim reappearance (a bounce-out). Configurable; the
# default was swept locally (10/15/20/25) — 18 gave the best val F1.
BOUNCE_OUT_WINDOW_FRAMES = 18
# A reappearance counts as "outside" the rim if its center is beyond
# this many rim-widths horizontally from the rim center, OR below the
# rim bottom by this many rim-heights.
BOUNCE_OUT_X_RW = 0.75
BOUNCE_OUT_Y_RH = 0.50
# Lever 2: half-window (in frames) around the closest-approach frame
# used to fit the trajectory parabola. Needs >=5 ball points to fit.
ARC_HALF_WINDOW = 12
ARC_MIN_POINTS = 5
# Lever 3: a far camera whose post-min rebound exceeds this (rim-widths)
# almost certainly lost the ball off-frame -> its quality is zeroed.
FAR_CAM_REBOUND_LEFT_FRAME_RW = 6.0

# A "through hoop" event requires the ball center to be horizontally within
# this many rim-widths of the rim center while crossing the rim plane.
THROUGH_HOOP_X_TOL_RW = 1.0
# Distance (in rim-widths) under which we call the ball "at the rim".
NEAR_RIM_DIST_RW = 1.5

NEAR_ANGLES = ("NL", "NR")
FAR_ANGLES = ("FL", "FR")

# Per-angle features computed by _angle_features (suffixed with _<angle>).
_PER_ANGLE_KEYS = [
    "n_frames",
    "ball_frac",          # fraction of window frames with a ball detection
    "ball_conf_mean",
    "ball_conf_max",
    "rim_frac",           # fraction with a rim detection
    "rim_detected",       # 1 if rim ever detected (=> rim ref usable)
    "min_dist_rw",        # min ball-rim center dist / rim_width (lower=closer)
    "min_dist_frac_time", # when closest approach happened (0..1 in window)
    "enters_rim_box",     # ball center ever inside rim box (1/0)
    "frac_inside_rim",    # fraction of ball frames with center in rim box
    "through_hoop",       # clean above->below crossing within x-tol (1/0)
    "through_hoop_count", # # of such crossing frames
    "through_hoop_conf",  # mean ball conf on crossing frames
    "vy_sign_change_near_rim",  # ball vy flips sign while near rim (1/0)
    "min_dist_conf",      # ball conf at closest approach
    "approach_drop",      # how much dist drops from start->min (rim-widths)
    "post_min_rebound",   # dist increase after min (miss bounce-out proxy)
    # --- Lever 1: bounce-out / rim-out recovery (miss-focused) ---------
    "bo_any",             # 1 if a below+outside-rim reappearance found
    "bo_conf",            # ball conf at the reappearance frame
    "bo_frames_to_reappear",  # frames from closest approach to reappear
    "bo_outward_disp",    # post-rim outward displacement / rim_width
    "bo_then_rises",      # 1 if the ball then rises/leaves (clean bounce-out)
    # --- Lever 2: parametric trajectory-arc fit (interpretable) --------
    "arc_ok",             # 1 if a parabola could be fit (enough points)
    "arc_entry_angle_deg",  # angle of descent into the rim plane (deg)
    "arc_vy_at_rim",      # vertical velocity at rim plane (px/frame, norm)
    "arc_curvature",      # parabola curvature a (y=a x^2+..), normalised
    "arc_fit_resid",      # RMS fit residual / rim_width (track noisiness)
    "arc_apex_dx_rw",     # apex x minus rim center x, in rim-widths
    "arc_apex_above_rim_rh",  # how far apex is above the rim, rim-heights
    # --- Lever 3: per-angle quality (drives quality-gated fusion) ------
    "q_quality",          # ball_frac * rim_stability (0..1), far-cam gated
]

# Imputation defaults for per-angle features when an angle is unusable.
_ANGLE_IMPUTE = {
    "n_frames": 0.0,
    "ball_frac": 0.0,
    "ball_conf_mean": 0.0,
    "ball_conf_max": 0.0,
    "rim_frac": 0.0,
    "rim_detected": 0.0,
    "min_dist_rw": 10.0,            # "very far" sentinel
    "min_dist_frac_time": 0.5,
    "enters_rim_box": 0.0,
    "frac_inside_rim": 0.0,
    "through_hoop": 0.0,
    "through_hoop_count": 0.0,
    "through_hoop_conf": 0.0,
    "vy_sign_change_near_rim": 0.0,
    "min_dist_conf": 0.0,
    "approach_drop": 0.0,
    "post_min_rebound": 0.0,
    "bo_any": 0.0,
    "bo_conf": 0.0,
    "bo_frames_to_reappear": 0.0,
    "bo_outward_disp": 0.0,
    "bo_then_rises": 0.0,
    "arc_ok": 0.0,
    "arc_entry_angle_deg": 0.0,
    "arc_vy_at_rim": 0.0,
    "arc_curvature": 0.0,
    "arc_fit_resid": 5.0,          # "very noisy" sentinel
    "arc_apex_dx_rw": 5.0,         # "far / undefined" sentinel
    "arc_apex_above_rim_rh": 0.0,
    "q_quality": 0.0,
}


def _robust_rim_ref(g: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Robust (median) rim box over the window. The rim is ~static, so the
    median over all detected frames is a stable reference. Returns None if
    the rim was never detected for this angle."""
    rim = g.dropna(subset=["rim_x", "rim_y", "rim_w", "rim_h"])
    if rim.empty:
        return None
    rx = float(np.median(rim["rim_x"]))
    ry = float(np.median(rim["rim_y"]))
    rw = float(np.median(rim["rim_w"]))
    rh = float(np.median(rim["rim_h"]))
    if rw <= 1.0 or rh <= 1.0:
        return None
    return {
        "rcx": rx + rw / 2.0,
        "rcy": ry + rh / 2.0,
        "rw": rw,
        "rh": rh,
        "rx0": rx,
        "ry0": ry,
        "rx1": rx + rw,
        "ry1": ry + rh,
    }


def _bounce_out_features(
    bcx: np.ndarray, bcy: np.ndarray, bconf: np.ndarray,
    fidx: np.ndarray, dist_rw: np.ndarray, j: int,
    rim: Dict[str, float],
) -> Dict[str, float]:
    """Lever 1 — rim-out / bounce-out recovery (precision-focused).

    A true rim-in-and-out MISS (the dominant FP) has a kinematic
    signature a make never produces: after the ball reaches the rim it
    REVERSES VERTICALLY (goes back UP) while still near the rim plane —
    it bounced off the iron. A made shot only ever continues downward
    through the net; gravity alone never sends it back up.

      (a) the ball genuinely reached the rim — closest approach j is
          within NEAR_RIM_DIST_RW (otherwise it never went "in");
      (b) within the window the ball, while still near the rim plane,
          goes from descending (vy>0, image y down) to clearly rising
          (vy<0) — an upward rebound off the rim;
      (c) `bo_then_rises` additionally requires it then leaves the rim
          region (lateral kick OR sustained rise) — the cleanest cue.

    The earlier "appears below+outside" version fired on ~all shots
    (every ball ends up below the rim by gravity); the up-reversal gate
    is specific to a genuine rim bounce.
    """
    out = {
        "bo_any": 0.0, "bo_conf": 0.0, "bo_frames_to_reappear": 0.0,
        "bo_outward_disp": 0.0, "bo_then_rises": 0.0,
    }
    n = len(bcx)
    if j >= n - 2 or n < 5:
        return out
    # (a) Must have actually reached the rim.
    if dist_rw[j] > NEAR_RIM_DIST_RW:
        return out
    rcx, rh, rw = rim["rcx"], rim["rh"], rim["rw"]

    # P1 tracks are jittery, so frame-level vy reversals are pure noise
    # (they fire on ~every shot). Use a SMOOTHED trajectory and require
    # a SUSTAINED upward rebound: after the closest approach the
    # (median-smoothed) ball must climb back up by a meaningful
    # fraction of a rim-height. A make only ever continues down through
    # the net; a real rim-out kicks the ball back up.
    def _med3(a: np.ndarray) -> np.ndarray:
        if len(a) < 3:
            return a.astype(float)
        b = a.astype(float).copy()
        b[1:-1] = np.median(
            np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
        return b

    sy = _med3(bcy)
    end = min(n, j + 1 + BOUNCE_OUT_WINDOW_FRAMES)
    y_at_min = sy[j]
    # Lowest point the ball reaches (max image-y) in the post-rim
    # window, then how far it climbs back UP afterwards.
    seg = sy[j:end]
    if len(seg) < 3:
        return out
    low_off = int(np.argmax(seg))          # deepest point (y largest)
    k_low = j + low_off
    if k_low >= n - 1:
        return out
    after = sy[k_low + 1:end]
    if after.size == 0:
        return out
    rebound_px = float(seg[low_off] - after.min())   # climb back up
    rebound_rh = rebound_px / rh
    # A sustained rebound of >= 0.6 rim-heights after the deepest
    # post-rim point is a genuine bounce-out (tuned on val).
    if rebound_rh >= 0.6 and seg[low_off] > y_at_min + 0.2 * rh:
        k_top = int(k_low + 1 + np.argmin(after))
        out["bo_any"] = 1.0
        out["bo_conf"] = float(bconf[min(k_top, n - 1)])
        out["bo_frames_to_reappear"] = float(fidx[k_low] - fidx[j])
        out["bo_outward_disp"] = float(
            abs(bcx[min(k_top, n - 1)] - rcx) / rw)
        # Clean bounce-out: it also kicks laterally away from the rim.
        lat = float(np.abs(bcx[k_low:end] - rcx).max() / rw)
        out["bo_then_rises"] = 1.0 if lat > BOUNCE_OUT_X_RW else 0.0
    return out


def _arc_features(
    bcx: np.ndarray, bcy: np.ndarray, fidx: np.ndarray,
    j: int, rim: Dict[str, float],
) -> Dict[str, float]:
    """Lever 2 — parametric trajectory-arc fit (interpretability).

    Fit a parabola to the ball centers in a +/-ARC_HALF_WINDOW frame
    window around the closest approach. We fit y(x) for geometry
    (curvature, apex) and y(t) for kinematics (vy at the rim plane,
    entry angle). All outputs are normalised by the rim box so they are
    scale/zoom invariant and named for interpretability. Image y grows
    downward, so a descending ball has dy/dt > 0.
    """
    out = {
        "arc_ok": 0.0, "arc_entry_angle_deg": 0.0, "arc_vy_at_rim": 0.0,
        "arc_curvature": 0.0, "arc_fit_resid": 5.0,
        "arc_apex_dx_rw": 5.0, "arc_apex_above_rim_rh": 0.0,
    }
    n = len(bcx)
    lo = max(0, j - ARC_HALF_WINDOW)
    hi = min(n, j + ARC_HALF_WINDOW + 1)
    xs = bcx[lo:hi].astype(float)
    ys = bcy[lo:hi].astype(float)
    ts = fidx[lo:hi].astype(float)
    if len(xs) < ARC_MIN_POINTS:
        return out
    rcx, rcy, rw, rh = rim["rcx"], rim["rcy"], rim["rw"], rim["rh"]
    # Need horizontal spread to fit y(x); otherwise fall back to y(t).
    try:
        if (xs.max() - xs.min()) > 1e-3:
            cx = np.polyfit(xs, ys, 2)
            yhat = np.polyval(cx, xs)
            resid = float(np.sqrt(np.mean((ys - yhat) ** 2)))
            out["arc_fit_resid"] = float(resid / rw)
            a = float(cx[0])
            out["arc_curvature"] = float(a * rw)  # dimensionless
            if abs(a) > 1e-9:
                apex_x = -cx[1] / (2.0 * a)
                apex_y = np.polyval(cx, apex_x)
                out["arc_apex_dx_rw"] = float((apex_x - rcx) / rw)
                # Positive => apex sits above the rim center (image y up).
                out["arc_apex_above_rim_rh"] = float(
                    (rcy - apex_y) / rh)
        # Kinematics from y(t): vertical velocity at the rim plane.
        if (ts.max() - ts.min()) > 1e-3:
            ct = np.polyfit(ts, ys, 2)
            # Time at which the ball crosses the rim center plane (y=rcy)
            # — closest fitted frame to the closest approach instead.
            t_rim = float(fidx[j])
            vy = float(2.0 * ct[0] * t_rim + ct[1])  # px/frame
            out["arc_vy_at_rim"] = float(vy / rh)
            # Horizontal speed near the rim for the entry-angle ratio.
            if (xs.max() - xs.min()) > 1e-3:
                ctx = np.polyfit(ts, xs, 1)
                vx = float(ctx[0])
            else:
                vx = 0.0
            ang = float(np.degrees(np.arctan2(abs(vy),
                                               abs(vx) + 1e-6)))
            out["arc_entry_angle_deg"] = ang
        out["arc_ok"] = 1.0
    except (np.linalg.LinAlgError, ValueError):
        return {
            "arc_ok": 0.0, "arc_entry_angle_deg": 0.0,
            "arc_vy_at_rim": 0.0, "arc_curvature": 0.0,
            "arc_fit_resid": 5.0, "arc_apex_dx_rw": 5.0,
            "arc_apex_above_rim_rh": 0.0,
        }
    return out


# Iteration-3 per-angle keys recomputed on the CLEANED track. The L1/L2
# math is identical to the raw versions; only the input ball-center
# series differs (Kalman-smoothed + bounded-gap-filled). Suffixed `_cl`
# so both coexist for the model / ablation. Plus imputation + quality.
_CLEAN_KEYS = [
    "cln_arc_ok", "cln_arc_entry_angle_deg", "cln_arc_vy_at_rim",
    "cln_arc_curvature", "cln_arc_fit_resid", "cln_arc_apex_dx_rw",
    "cln_arc_apex_above_rim_rh",
    "cln_bo_any", "cln_bo_conf", "cln_bo_frames_to_reappear",
    "cln_bo_outward_disp", "cln_bo_then_rises",
    "cln_min_dist_rw", "cln_through_hoop", "cln_frac_inside_rim",
    # Imputation / track-quality awareness (lets the model discount
    # shots whose cleaned trajectory is mostly bridged guesswork).
    "imp_frac",            # fraction of cleaned points that were imputed
    "n_imputed",           # # imputed frames
    "n_rejected_outlier",  # # raw detections killed as teleports
    "track_quality",       # [0,1] per-angle cleaned-track reliability
    "jitter_raw",          # median |2nd-diff| of the RAW y-center
    "jitter_clean",        # same on the CLEANED y-center (should drop)
    "jitter_reduction",    # raw - clean (>0 => de-jittered)
    "n_cleaned_points",    # usable cleaned centers (detections + fills)
]

_CLEAN_IMPUTE = {
    "cln_arc_ok": 0.0, "cln_arc_entry_angle_deg": 0.0,
    "cln_arc_vy_at_rim": 0.0, "cln_arc_curvature": 0.0,
    "cln_arc_fit_resid": 5.0, "cln_arc_apex_dx_rw": 5.0,
    "cln_arc_apex_above_rim_rh": 0.0,
    "cln_bo_any": 0.0, "cln_bo_conf": 0.0,
    "cln_bo_frames_to_reappear": 0.0, "cln_bo_outward_disp": 0.0,
    "cln_bo_then_rises": 0.0,
    "cln_min_dist_rw": 10.0, "cln_through_hoop": 0.0,
    "cln_frac_inside_rim": 0.0,
    "imp_frac": 0.0, "n_imputed": 0.0, "n_rejected_outlier": 0.0,
    "track_quality": 0.0, "jitter_raw": 0.0, "jitter_clean": 0.0,
    "jitter_reduction": 0.0, "n_cleaned_points": 0.0,
}


def _cleaned_track_features(
    g: pd.DataFrame, rim: Dict[str, float],
) -> Dict[str, float]:
    """Iteration 3 — recompute the L1/L2 trajectory levers on the
    Kalman-cleaned ball track and expose imputation / quality signals.

    ``g`` is one (play_id, angle) frame slice; ``rim`` is its robust
    rim reference. The bounce-out (L1) and arc-fit (L2) math is the
    SAME as the raw path — only the ball-center series changes (the
    cleaned track de-jitters real detections and bridges short, bounded
    occlusion gaps, which is exactly what flattened these signals on
    the raw track per the iteration-2 conclusion). All cleaned outputs
    are prefixed `cln_`; raw L1/L2 are kept untouched elsewhere.
    """
    feats = dict(_CLEAN_IMPUTE)
    res = clean_track(g)
    n_all = len(res.frame_idx)
    if n_all == 0:
        return feats

    feats["n_rejected_outlier"] = float(int(res.rejected_outlier.sum()))
    feats["n_imputed"] = float(int(res.imputed.sum()))
    feats["jitter_raw"] = (
        0.0 if np.isnan(jitter_metric(res.raw_cy))
        else float(jitter_metric(res.raw_cy)))
    jc = jitter_metric(res.cy)
    feats["jitter_clean"] = 0.0 if np.isnan(jc) else float(jc)
    feats["jitter_reduction"] = float(
        feats["jitter_raw"] - feats["jitter_clean"])

    valid = ~np.isnan(res.cx)
    feats["n_cleaned_points"] = float(int(valid.sum()))
    if valid.sum() > 0:
        feats["imp_frac"] = float(
            res.imputed[valid].sum()) / float(valid.sum())
    feats["track_quality"] = float(res.track_quality)

    if rim is None or valid.sum() < 3:
        return feats

    bcx = res.cx[valid]
    bcy = res.cy[valid]
    fidx = res.frame_idx[valid].astype(float)
    # Confidence proxy on the cleaned track: imputed points are softer
    # evidence than kept detections (used only by cln_bo_conf).
    bconf = np.where(res.imputed[valid] == 1, 0.4, 0.9)

    rcx, rcy, rw = rim["rcx"], rim["rcy"], rim["rw"]
    dist_rw = np.hypot(bcx - rcx, bcy - rcy) / rw
    j = int(np.argmin(dist_rw))
    feats["cln_min_dist_rw"] = float(dist_rw[j])

    inside = (
        (bcx >= rim["rx0"]) & (bcx <= rim["rx1"])
        & (bcy >= rim["ry0"]) & (bcy <= rim["ry1"])
    )
    feats["cln_frac_inside_rim"] = float(inside.mean())
    # Clean through-hoop: above->below rim plane within x-tol.
    xtol = THROUGH_HOOP_X_TOL_RW * rw
    horiz_ok = np.abs(bcx - rcx) <= xtol
    above = bcy <= rcy
    th = False
    for k in range(1, len(bcy)):
        if (above[k - 1] and not above[k]
                and horiz_ok[k - 1] and horiz_ok[k]):
            th = True
            break
    feats["cln_through_hoop"] = 1.0 if th else 0.0

    # Reuse the EXACT L1/L2 implementations on the cleaned series.
    bo = _bounce_out_features(bcx, bcy, bconf, fidx, dist_rw, j, rim)
    for k, v in bo.items():
        feats[f"cln_{k}"] = float(v)
    arc = _arc_features(bcx, bcy, fidx, j, rim)
    for k, v in arc.items():
        feats[f"cln_{k}"] = float(v)
    return feats


def _angle_features(g: pd.DataFrame) -> Dict[str, float]:
    """Compute interpretable ball-vs-rim features for one angle's window.

    g is already sorted by frame_idx and restricted to a single
    (play_id, angle). Always returns every _PER_ANGLE_KEYS key (imputed
    defaults when the signal is undefined) so the fused row is dense.
    """
    feats = dict(_ANGLE_IMPUTE)
    n = len(g)
    feats["n_frames"] = float(n)
    if n == 0:
        return feats

    rim = _robust_rim_ref(g)
    feats["rim_frac"] = float(g["rim_conf"].notna().mean())
    feats["rim_detected"] = 1.0 if rim is not None else 0.0

    ball = g.dropna(subset=["ball_x", "ball_y", "ball_w", "ball_h"]).copy()
    feats["ball_frac"] = float(len(ball)) / float(n)
    if len(ball):
        bc = g["ball_conf"].dropna()
        feats["ball_conf_mean"] = float(bc.mean()) if len(bc) else 0.0
        feats["ball_conf_max"] = float(bc.max()) if len(bc) else 0.0

    if rim is None or ball.empty:
        return feats

    rcx, rcy, rw = rim["rcx"], rim["rcy"], rim["rw"]
    bcx = ball["ball_x"].to_numpy() + ball["ball_w"].to_numpy() / 2.0
    bcy = ball["ball_y"].to_numpy() + ball["ball_h"].to_numpy() / 2.0
    bconf = ball["ball_conf"].fillna(0.0).to_numpy()
    fidx = ball["frame_idx"].to_numpy().astype(float)

    dist = np.hypot(bcx - rcx, bcy - rcy)
    dist_rw = dist / rw

    j = int(np.argmin(dist_rw))
    feats["min_dist_rw"] = float(dist_rw[j])
    feats["min_dist_conf"] = float(bconf[j])
    f0, f1 = fidx.min(), fidx.max()
    span = max(f1 - f0, 1.0)
    feats["min_dist_frac_time"] = float((fidx[j] - f0) / span)

    # Ball center inside the rim box.
    inside = (
        (bcx >= rim["rx0"]) & (bcx <= rim["rx1"])
        & (bcy >= rim["ry0"]) & (bcy <= rim["ry1"])
    )
    feats["enters_rim_box"] = 1.0 if inside.any() else 0.0
    feats["frac_inside_rim"] = float(inside.mean())

    # "Through hoop": consecutive detected ball frames where the center
    # moves from at/above the rim plane (bcy <= rcy) to below it
    # (bcy > rcy) while staying horizontally within X_TOL rim-widths of
    # the rim center. A made shot drops vertically through the hoop.
    xtol = THROUGH_HOOP_X_TOL_RW * rw
    horiz_ok = np.abs(bcx - rcx) <= xtol
    above = bcy <= rcy
    th_count = 0
    th_confs: List[float] = []
    for k in range(1, len(bcy)):
        if above[k - 1] and (not above[k]) and horiz_ok[k - 1] and horiz_ok[k]:
            th_count += 1
            th_confs.append(0.5 * (bconf[k - 1] + bconf[k]))
    feats["through_hoop_count"] = float(th_count)
    feats["through_hoop"] = 1.0 if th_count > 0 else 0.0
    feats["through_hoop_conf"] = (
        float(np.mean(th_confs)) if th_confs else 0.0
    )

    # Vertical-velocity sign change while the ball is near the rim — a
    # bounce/rim interaction. Misses often bounce (vy flips up) near rim.
    near = dist_rw <= NEAR_RIM_DIST_RW
    if near.sum() >= 3:
        vy = np.diff(bcy)
        near_pairs = near[1:] & near[:-1]
        sign_flip = False
        prev = 0
        for k in range(len(vy)):
            if not near_pairs[k]:
                continue
            s = np.sign(vy[k])
            if s == 0:
                continue
            if prev != 0 and s != prev:
                sign_flip = True
                break
            prev = s
        feats["vy_sign_change_near_rim"] = 1.0 if sign_flip else 0.0

    # Approach dynamics: how far the ball travelled toward the rim, and
    # whether the distance rebounds after the closest approach (a strong
    # miss / bounce-out cue vs. a clean make that just disappears).
    feats["approach_drop"] = float(max(dist_rw[0] - dist_rw[j], 0.0))
    if j < len(dist_rw) - 1:
        feats["post_min_rebound"] = float(
            max(dist_rw[j + 1:].max() - dist_rw[j], 0.0)
        )

    # --- Lever 1: bounce-out / rim-out recovery (RAW track) ---------
    feats.update(
        _bounce_out_features(bcx, bcy, bconf, fidx, dist_rw, j, rim))
    # --- Lever 2: parametric trajectory-arc fit (RAW track) --------
    feats.update(_arc_features(bcx, bcy, fidx, j, rim))

    # --- Iteration 3: recompute L1/L2 on the KALMAN-CLEANED track ---
    # The raw L1/L2 above are kept untouched (suffixed `_raw` below at
    # fuse time via the v3 key map). Here we run the exact same lever
    # math on the cleaned ball-center series so the model can compare /
    # ablate raw vs cleaned. Also emit per-angle imputation fraction and
    # track-quality so the model can discount unreliable shots, plus a
    # de-jitter sanity delta (median |2nd-diff| raw vs cleaned).
    feats.update(_cleaned_track_features(g, rim))

    # --- Lever 3: per-angle quality = ball-detection fraction *
    # rim-detection stability. rim stability = how consistently the
    # rim was seen (fraction of frames) — a flaky rim ref makes every
    # distance feature noisy, so its angle should weigh less.
    q = float(feats["ball_frac"]) * float(feats["rim_frac"])
    feats["q_quality"] = float(max(0.0, min(1.0, q)))
    return feats


def _fuse(per_angle: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Fuse the 4 per-angle feature dicts into one shot-level row.

    Strategy: emit every per-angle feature flat (so the model can learn
    angle-specific weights), plus physically motivated cross-angle
    aggregates (max/mean of made-like signals, vote agreement, count of
    angles with a clean through-hoop, near vs far symmetry, confidence-
    weighted closest-approach distance)."""
    row: Dict[str, float] = {}
    for ang in ANGLES:
        fa = per_angle.get(ang, dict(_ANGLE_IMPUTE))
        for k in _PER_ANGLE_KEYS:
            row[f"{k}_{ang}"] = float(fa.get(k, _ANGLE_IMPUTE[k]))
        # Iteration-3 cleaned-track keys (present only when v3 was
        # requested; default-imputed otherwise so the row stays dense).
        for k in _CLEAN_KEYS:
            row[f"{k}_{ang}"] = float(fa.get(k, _CLEAN_IMPUTE[k]))
        row[f"angle_usable_{ang}"] = (
            1.0 if (fa.get("rim_detected", 0.0) >= 1.0
                    and fa.get("ball_frac", 0.0) > 0.0) else 0.0
        )

    usable = [a for a in ANGLES if row[f"angle_usable_{a}"] >= 1.0]
    row["n_usable_angles"] = float(len(usable))

    # --- Lever 3: far-cam down-weighting. A far camera whose post-min
    # rebound is huge almost certainly lost the ball off-frame (the
    # rebound value is then a spurious blow-up). Zero its quality so it
    # cannot dominate the quality-weighted fusion below.
    for a in FAR_ANGLES:
        if row[f"post_min_rebound_{a}"] >= FAR_CAM_REBOUND_LEFT_FRAME_RW:
            row[f"q_quality_{a}"] = 0.0
            row[f"far_cam_left_frame_{a}"] = 1.0
        else:
            row[f"far_cam_left_frame_{a}"] = 0.0

    def vals(key: str, angs=ANGLES) -> np.ndarray:
        return np.array([row[f"{key}_{a}"] for a in angs], dtype=float)

    def uvals(key: str, angs=ANGLES) -> np.ndarray:
        v = [row[f"{key}_{a}"] for a in angs if a in usable]
        return np.array(v, dtype=float)

    # Made-like signals: a make is best evidenced by a small min distance,
    # a clean through-hoop event, and the ball entering the rim box.
    md = uvals("min_dist_rw")
    row["min_dist_rw_best"] = float(md.min()) if md.size else 10.0
    row["min_dist_rw_mean"] = float(md.mean()) if md.size else 10.0
    row["through_hoop_any"] = float((vals("through_hoop") >= 1.0).any())
    row["through_hoop_votes"] = float(
        sum(row[f"through_hoop_{a}"] for a in usable)
    )
    row["through_hoop_count_max"] = float(vals("through_hoop_count").max())
    thc = uvals("through_hoop_conf")
    row["through_hoop_conf_max"] = float(thc.max()) if thc.size else 0.0
    row["enters_rim_votes"] = float(
        sum(row[f"enters_rim_box_{a}"] for a in usable)
    )
    fir = uvals("frac_inside_rim")
    row["frac_inside_rim_max"] = float(fir.max()) if fir.size else 0.0

    # Miss-like signals: bounce-out rebound and vy sign change near rim.
    pmr = uvals("post_min_rebound")
    row["post_min_rebound_max"] = float(pmr.max()) if pmr.size else 0.0
    row["vy_sign_change_votes"] = float(
        sum(row[f"vy_sign_change_near_rim_{a}"] for a in usable)
    )

    # Confidence-weighted closest approach: weight each angle's min-dist by
    # the ball conf at that closest frame; robust when some angles are noisy.
    if usable:
        w = np.array(
            [max(row[f"min_dist_conf_{a}"], 1e-3) for a in usable]
        )
        d = np.array([row[f"min_dist_rw_{a}"] for a in usable])
        row["min_dist_rw_confw"] = float(np.average(d, weights=w))
    else:
        row["min_dist_rw_confw"] = 10.0

    # Near vs far symmetry — near cameras (NL/NR) usually see the rim
    # plane better; keep both groups so the model can weight them.
    for grp, angs in (("near", NEAR_ANGLES), ("far", FAR_ANGLES)):
        gm = [row[f"min_dist_rw_{a}"] for a in angs
              if row[f"angle_usable_{a}"] >= 1.0]
        row[f"min_dist_rw_{grp}_best"] = (
            float(min(gm)) if gm else 10.0
        )
        row[f"through_hoop_{grp}_any"] = float(
            any(row[f"through_hoop_{a}"] >= 1.0 for a in angs
                if row[f"angle_usable_{a}"] >= 1.0)
        )

    # Overall detection quality (imputation-awareness for the model).
    row["ball_frac_mean"] = float(vals("ball_frac").mean())
    row["ball_conf_max_overall"] = float(vals("ball_conf_max").max())
    row["any_angle_usable"] = 1.0 if usable else 0.0

    # ================= Iteration-2 fused lever features =============
    # Quality weights over usable angles (lever 3), used for all of the
    # quality-weighted aggregates below. Falls back to uniform if every
    # usable angle has zero quality (degenerate but keeps it dense).
    qw = np.array([max(row[f"q_quality_{a}"], 0.0) for a in usable],
                  dtype=float)
    if qw.sum() <= 1e-9:
        qw = np.ones(len(usable), dtype=float) if usable else qw
    row["q_quality_max"] = (
        float(max(row[f"q_quality_{a}"] for a in ANGLES)))
    row["q_quality_sum_usable"] = float(qw.sum()) if usable else 0.0

    def qweighted(key: str, sentinel: float) -> float:
        if not usable or qw.sum() <= 1e-9:
            return sentinel
        d = np.array([row[f"{key}_{a}"] for a in usable], dtype=float)
        return float(np.average(d, weights=qw))

    # Lever 3: quality-weighted closest approach replaces naive
    # max/mean. Keep min_dist_rw_best (already exists) for the model to
    # compare against the gated version.
    row["min_dist_rw_qw"] = qweighted("min_dist_rw", 10.0)
    row["arc_entry_angle_qw"] = qweighted("arc_entry_angle_deg", 0.0)
    row["arc_vy_at_rim_qw"] = qweighted("arc_vy_at_rim", 0.0)

    # Lever 3: >=2-angle AGREEMENT gate. A low-distance shot only counts
    # as MAKE-like if at least two usable angles independently agree the
    # ball got close to the rim (guards single-camera false positives).
    close_votes = sum(
        1 for a in usable if row[f"min_dist_rw_{a}"] <= NEAR_RIM_DIST_RW)
    row["close_agree_votes"] = float(close_votes)
    row["agree_make_like"] = 1.0 if close_votes >= 2 else 0.0
    th_votes = sum(1 for a in usable if row[f"through_hoop_{a}"] >= 1.0)
    row["through_hoop_agree2"] = 1.0 if th_votes >= 2 else 0.0

    # Lever 1: fused multi-angle bounce-out. A bounce-out seen on any
    # usable angle is a strong MISS cue; quality-weighted confidence and
    # the strongest outward displacement summarise it for the model.
    bo_flags = np.array([row[f"bo_any_{a}"] for a in usable], dtype=float)
    row["bo_any_fused"] = float((bo_flags >= 1.0).any()) if usable else 0.0
    row["bo_votes"] = float(bo_flags.sum()) if usable else 0.0
    row["bo_conf_fused"] = qweighted("bo_conf", 0.0)
    bod = uvals("bo_outward_disp")
    row["bo_outward_disp_max"] = (
        float(bod.max()) if bod.size else 0.0)
    bor = np.array([row[f"bo_then_rises_{a}"] for a in usable],
                   dtype=float)
    row["bo_then_rises_any"] = (
        float((bor >= 1.0).any()) if usable else 0.0)
    # Clean bounce-out = reappears below+outside AND then leaves on the
    # same angle. The sharpest rim-in-and-out MISS signal.
    row["bo_clean_votes"] = float(sum(
        1 for a in usable
        if row[f"bo_any_{a}"] >= 1.0 and row[f"bo_then_rises_{a}"] >= 1.0))
    row["bo_clean_any"] = 1.0 if row["bo_clean_votes"] >= 1.0 else 0.0

    # Lever 2: fused arc geometry. A made shot has a steep entry angle,
    # downward vy at the rim plane, an apex above & near the rim, and a
    # low fit residual. Aggregate with quality weighting + best/spread.
    aok = uvals("arc_ok")
    row["arc_ok_any"] = float((aok >= 1.0).any()) if aok.size else 0.0
    row["arc_ok_votes"] = float(aok.sum()) if aok.size else 0.0
    fr = uvals("arc_fit_resid")
    row["arc_fit_resid_min"] = float(fr.min()) if fr.size else 5.0
    row["arc_entry_angle_max"] = float(
        uvals("arc_entry_angle_deg").max()
        if uvals("arc_entry_angle_deg").size else 0.0)
    apx = uvals("arc_apex_above_rim_rh")
    row["arc_apex_above_rim_rh_max"] = float(apx.max()) if apx.size else 0.0
    adx = np.abs(uvals("arc_apex_dx_rw"))
    row["arc_apex_dx_rw_min"] = float(adx.min()) if adx.size else 5.0
    row["arc_vy_at_rim_max"] = float(
        uvals("arc_vy_at_rim").max()
        if uvals("arc_vy_at_rim").size else 0.0)

    # ============ Iteration-3 fused CLEANED-track features ==========
    # Same fusion philosophy as the raw levers but over the Kalman-
    # cleaned per-angle signals, plus shot-level imputation/quality
    # aggregates so the model can globally discount weak shots.
    def cuvals(key: str) -> np.ndarray:
        v = [row[f"{key}_{a}"] for a in ANGLES if a in usable]
        return np.array(v, dtype=float)

    cmd = cuvals("cln_min_dist_rw")
    row["cln_min_dist_rw_best"] = float(cmd.min()) if cmd.size else 10.0
    row["cln_through_hoop_any"] = float(
        (cuvals("cln_through_hoop") >= 1.0).any()
        if cuvals("cln_through_hoop").size else 0.0)
    row["cln_through_hoop_votes"] = float(
        sum(row[f"cln_through_hoop_{a}"] for a in usable))
    cfir = cuvals("cln_frac_inside_rim")
    row["cln_frac_inside_rim_max"] = (
        float(cfir.max()) if cfir.size else 0.0)
    # Cleaned bounce-out (the L1 signal iter2 said was flattened).
    cbo = np.array([row[f"cln_bo_any_{a}"] for a in usable], dtype=float)
    row["cln_bo_any_fused"] = float(
        (cbo >= 1.0).any()) if usable else 0.0
    row["cln_bo_votes"] = float(cbo.sum()) if usable else 0.0
    row["cln_bo_clean_any"] = float(
        any(row[f"cln_bo_any_{a}"] >= 1.0
            and row[f"cln_bo_then_rises_{a}"] >= 1.0
            for a in usable)) if usable else 0.0
    cbod = cuvals("cln_bo_outward_disp")
    row["cln_bo_outward_disp_max"] = (
        float(cbod.max()) if cbod.size else 0.0)
    # Cleaned arc geometry (the L2 signal iter2 said was flattened).
    caok = cuvals("cln_arc_ok")
    row["cln_arc_ok_votes"] = float(caok.sum()) if caok.size else 0.0
    cfr = cuvals("cln_arc_fit_resid")
    row["cln_arc_fit_resid_min"] = float(cfr.min()) if cfr.size else 5.0
    cea = cuvals("cln_arc_entry_angle_deg")
    row["cln_arc_entry_angle_max"] = (
        float(cea.max()) if cea.size else 0.0)
    capx = cuvals("cln_arc_apex_above_rim_rh")
    row["cln_arc_apex_above_rim_rh_max"] = (
        float(capx.max()) if capx.size else 0.0)
    cadx = np.abs(cuvals("cln_arc_apex_dx_rw"))
    row["cln_arc_apex_dx_rw_min"] = (
        float(cadx.min()) if cadx.size else 5.0)
    cvy = cuvals("cln_arc_vy_at_rim")
    row["cln_arc_vy_at_rim_max"] = (
        float(cvy.max()) if cvy.size else 0.0)
    # Shot-level imputation / quality (discount unreliable shots).
    row["track_quality_mean"] = float(vals("track_quality").mean())
    row["track_quality_max"] = float(vals("track_quality").max())
    row["imp_frac_mean"] = float(vals("imp_frac").mean())
    iq = cuvals("imp_frac")
    row["imp_frac_min_usable"] = float(iq.min()) if iq.size else 1.0
    row["n_rejected_outlier_total"] = float(
        vals("n_rejected_outlier").sum())
    row["jitter_reduction_mean"] = float(
        vals("jitter_reduction").mean())
    return row


def _shot_rows(game_id: str, df: pd.DataFrame) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for play_id, gp in df.groupby("play_id", sort=True):
        classification = str(gp["classification"].iloc[0])
        if classification in SHOT_MAKE_CLASSES:
            label = 1
        elif classification in SHOT_MISS_CLASSES:
            label = 0
        else:
            continue  # not a make/miss shot class
        per_angle: Dict[str, Dict[str, float]] = {}
        for ang in ANGLES:
            ga = gp[gp["angle"] == ang].sort_values("frame_idx")
            per_angle[ang] = _angle_features(ga)
        fused = _fuse(per_angle)
        fused.update(
            game_id=game_id,
            play_id=str(play_id),
            classification=classification,
            label=label,
            v1_in_scope=classification in V1_ALL_SHOT_CLASSES,
        )
        rows.append(fused)
    return rows


def build() -> pd.DataFrame:
    games = {g.game_id: g for g in load_manifest()}
    all_rows: List[Dict[str, float]] = []
    for gid, game in games.items():
        pq = TRACKS_CACHE / f"{gid}.parquet"
        if not pq.exists():
            raise FileNotFoundError(
                f"missing cached tracks for {gid}: {pq} "
                f"(run the P1 download first)"
            )
        df = pd.read_parquet(pq)
        rows = _shot_rows(gid, df)
        for r in rows:
            r["split"] = game.split
        all_rows.extend(rows)
        eprint(f"[p2] {gid} split={game.split} shots={len(rows)}")

    out = pd.DataFrame(all_rows)
    lead = ["game_id", "play_id", "classification", "label",
            "v1_in_scope", "split"]
    feat_cols = [c for c in out.columns if c not in lead]
    out = out[lead + sorted(feat_cols)]
    out = out.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    return out


def _coverage(df: pd.DataFrame) -> Dict[str, object]:
    th_cols = [f"through_hoop_{a}" for a in ANGLES]
    has_th = (df[th_cols].sum(axis=1) > 0)
    return {
        "n_shots": int(len(df)),
        "n_makes": int((df["label"] == 1).sum()),
        "n_misses": int((df["label"] == 0).sum()),
        "n_v1_in_scope": int(df["v1_in_scope"].sum()),
        "n_4pt_make": int((df["classification"] == "4PT_MAKE").sum()),
        "shots_with_through_hoop_ge1_angle": int(has_th.sum()),
        "frac_with_through_hoop": round(float(has_th.mean()), 4),
        "mean_usable_angles": round(
            float(df["n_usable_angles"].mean()), 3),
        "shots_0_usable_angles": int((df["n_usable_angles"] == 0).sum()),
        "by_split": {
            s: int((df["split"] == s).sum())
            for s in ("train", "val", "test")
        },
        "through_hoop_make_rate": round(float(
            df.loc[has_th, "label"].mean()), 4) if has_th.any() else None,
        "no_through_hoop_make_rate": round(float(
            df.loc[~has_th, "label"].mean()), 4) if (~has_th).any() else None,
    }


def _write_doc(df: pd.DataFrame, cov: Dict[str, object]) -> None:
    feat_cols = [c for c in df.columns if c not in (
        "game_id", "play_id", "classification", "label",
        "v1_in_scope", "split")]
    lines: List[str] = []
    lines.append("# P2 Feature Dictionary (iteration 2 / v2)\n")
    lines.append(
        "One row per shot `(game_id, play_id)`. Features describe the "
        "ball's geometry relative to the (near-static) rim over the "
        "buffered GT window, computed per camera angle (FL/FR/NL/NR) and "
        "fused. All coordinates are pixels in 1920x1080; (x,y)=box "
        "top-left. Distances are normalised by the robust median rim "
        "width so they are scale/zoom invariant. Iteration 2 adds three "
        "interpretable lever groups on top of the v1 features (kept "
        "intact so each lever can be ablated): `bo_*` rim-out/bounce-out "
        "recovery (L1), `arc_*` parametric trajectory-arc fit (L2), and "
        "`q_*` per-angle quality-gated fusion (L3).\n")
    lines.append("## Coverage\n")
    for k, v in cov.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Per-angle features (suffix `_FL/_FR/_NL/_NR`)\n")
    rationale = {
        "n_frames": "window length for this angle (context).",
        "ball_frac": "fraction of frames the ball was detected — "
                     "detection quality / imputation awareness.",
        "ball_conf_mean": "mean ball detector confidence.",
        "ball_conf_max": "peak ball confidence (clear sighting).",
        "rim_frac": "fraction of frames the rim was detected.",
        "rim_detected": "1 if a robust rim reference exists for this angle.",
        "min_dist_rw": "closest ball-center to rim-center distance over "
                       "the window, in rim-widths. Low => ball reached the "
                       "hoop (necessary for a make).",
        "min_dist_frac_time": "when (0..1 through window) closest approach "
                              "occurred.",
        "enters_rim_box": "1 if the ball center was ever inside the rim "
                          "bounding box.",
        "frac_inside_rim": "fraction of ball frames with center in rim box.",
        "through_hoop": "1 if a clean above->below rim-plane crossing "
                        "occurred within ~1 rim-width horizontally — the "
                        "core make cue.",
        "through_hoop_count": "number of such crossing frames.",
        "through_hoop_conf": "mean ball confidence on crossing frames.",
        "vy_sign_change_near_rim": "1 if the ball's vertical velocity "
                                   "flipped sign while near the rim — a "
                                   "rim bounce (miss cue).",
        "min_dist_conf": "ball confidence at the closest-approach frame.",
        "approach_drop": "how much the rim distance fell from window start "
                         "to the closest approach (rim-widths).",
        "post_min_rebound": "how much distance increased after the closest "
                            "approach — bounce-out (miss cue).",
        "angle_usable": "1 if rim ref exists AND ball seen at least once.",
        "bo_any": "L1: 1 if ball reappeared BELOW+OUTSIDE the rim after "
                  "closest approach — rim-in-and-out MISS cue.",
        "bo_conf": "L1: ball confidence at that reappearance frame.",
        "bo_frames_to_reappear": "L1: frames from closest approach to the "
                                 "below+outside reappearance.",
        "bo_outward_disp": "L1: post-rim outward displacement / rim width.",
        "bo_then_rises": "L1: 1 if the ball then rises/leaves (clean "
                         "bounce-out, strongest MISS evidence).",
        "arc_ok": "L2: 1 if a parabola could be fit (>=5 ball points).",
        "arc_entry_angle_deg": "L2: descent angle into the rim plane (deg); "
                               "steeper => more make-like.",
        "arc_vy_at_rim": "L2: vertical velocity at the rim plane "
                         "(px/frame / rim height; +ve = downward).",
        "arc_curvature": "L2: parabola curvature a (dimensionless, a*rw).",
        "arc_fit_resid": "L2: RMS parabola residual / rim width — track "
                         "noisiness / fit quality.",
        "arc_apex_dx_rw": "L2: apex x minus rim center, in rim-widths.",
        "arc_apex_above_rim_rh": "L2: apex height above the rim, "
                                 "rim-heights.",
        "q_quality": "L3: angle quality = ball_frac * rim_frac (0..1); "
                     "far cams that lost the ball off-frame are zeroed.",
    }
    for base, why in rationale.items():
        lines.append(f"- `{base}_*`: {why}")
    lines.append("")
    lines.append("## Fused cross-angle features\n")
    fused_doc = {
        "n_usable_angles": "# angles with rim ref + a ball detection.",
        "min_dist_rw_best": "min over usable angles of min_dist_rw "
                            "(strongest make-distance evidence).",
        "min_dist_rw_mean": "mean over usable angles.",
        "min_dist_rw_confw": "ball-conf-weighted mean of per-angle min "
                             "distance (robust fusion).",
        "through_hoop_any": "any angle saw a clean through-hoop.",
        "through_hoop_votes": "# usable angles with a through-hoop event.",
        "through_hoop_count_max": "max crossing-frame count across angles.",
        "through_hoop_conf_max": "best through-hoop confidence.",
        "enters_rim_votes": "# usable angles where ball entered rim box.",
        "frac_inside_rim_max": "max frac_inside_rim across angles.",
        "post_min_rebound_max": "max bounce-out rebound across angles "
                                "(miss cue).",
        "vy_sign_change_votes": "# angles with a near-rim vy sign flip.",
        "min_dist_rw_near_best": "best min-dist among near cams (NL/NR).",
        "min_dist_rw_far_best": "best min-dist among far cams (FL/FR).",
        "through_hoop_near_any": "through-hoop seen by a near cam.",
        "through_hoop_far_any": "through-hoop seen by a far cam.",
        "ball_frac_mean": "mean ball-detection fraction across angles.",
        "ball_conf_max_overall": "global peak ball confidence.",
        "any_angle_usable": "1 if at least one angle is usable.",
        # --- Iteration-2 fused levers ---
        "far_cam_left_frame_*": "L3: 1 if a far cam's rebound blew up "
                                "(ball left frame) => its quality zeroed.",
        "q_quality_max": "L3: best per-angle quality.",
        "q_quality_sum_usable": "L3: total quality mass over usable cams.",
        "min_dist_rw_qw": "L3: quality-weighted closest approach "
                          "(replaces naive max/mean fusion).",
        "arc_entry_angle_qw": "L3: quality-weighted arc entry angle.",
        "arc_vy_at_rim_qw": "L3: quality-weighted vy at the rim plane.",
        "close_agree_votes": "L3: # usable angles agreeing the ball came "
                             "near the rim.",
        "agree_make_like": "L3: 1 only if >=2 angles agree on a low "
                           "distance (single-cam FP guard).",
        "through_hoop_agree2": "L3: 1 if >=2 angles saw a through-hoop.",
        "bo_any_fused": "L1: any usable angle saw a bounce-out (MISS).",
        "bo_votes": "L1: # usable angles with a bounce-out.",
        "bo_conf_fused": "L1: quality-weighted bounce-out confidence.",
        "bo_outward_disp_max": "L1: max post-rim outward displacement.",
        "bo_then_rises_any": "L1: any angle saw the ball leave after.",
        "bo_clean_votes": "L1: # angles with reappear-below+outside AND "
                          "then-leaves (sharpest rim-out MISS cue).",
        "bo_clean_any": "L1: 1 if any clean bounce-out (strong MISS).",
        "arc_ok_any": "L2: any angle produced a parabola fit.",
        "arc_ok_votes": "L2: # angles with a parabola fit.",
        "arc_fit_resid_min": "L2: best (lowest) arc fit residual.",
        "arc_entry_angle_max": "L2: steepest entry angle across angles.",
        "arc_apex_above_rim_rh_max": "L2: max apex-above-rim height.",
        "arc_apex_dx_rw_min": "L2: closest apex-to-rim horizontal offset.",
        "arc_vy_at_rim_max": "L2: max downward vy at the rim plane.",
    }
    for k, why in fused_doc.items():
        lines.append(f"- `{k}`: {why}")
    lines.append("")
    lines.append(f"Total feature columns: **{len(feat_cols)}**.\n")
    lines.append(
        "### Sanity: through-hoop vs label\n"
        f"- Shots with a through-hoop on >=1 angle: make rate = "
        f"{cov['through_hoop_make_rate']} "
        f"(n={cov['shots_with_through_hoop_ge1_angle']})\n"
        f"- Shots with NO through-hoop: make rate = "
        f"{cov['no_through_hoop_make_rate']}\n")
    OUT_DOC.write_text("\n".join(lines))


def _write_doc_v3(df: pd.DataFrame, cov: Dict[str, object]) -> None:
    """Iteration-3 feature dictionary. Documents ONLY the v3 additions
    (cleaned-track L1/L2 + imputation/quality); the v2 dictionary in
    P2_FEATURES_v2.md remains the reference for the inherited columns."""
    feat_cols = [c for c in df.columns if c not in (
        "game_id", "play_id", "classification", "label",
        "v1_in_scope", "split")]
    cln_cols = sorted(c for c in feat_cols
                      if c.startswith("cln_") or c.startswith("imp_")
                      or c.startswith("track_quality")
                      or c.startswith("jitter")
                      or c.startswith("n_rejected_outlier"))
    L: List[str] = ["# P2 Feature Dictionary (iteration 3 / v3)\n"]
    L.append(
        "Superset of `p2_features_v2.parquet`: every v2 column is kept "
        "verbatim (see `P2_FEATURES_v2.md` for those) and v3 ADDS the "
        "trajectory levers L1 (bounce-out) and L2 (arc-fit) recomputed "
        "on the **Kalman-cleaned** ball track (`pipeline/track_clean.py`) "
        "plus per-angle imputation / track-quality signals. The raw-"
        "track L1/L2 (`bo_*`, `arc_*`) are UNCHANGED so the model can "
        "use both and the ablation can isolate the cleaning effect. v1/"
        "v2 parquet are never modified.\n")
    L.append("## Track-cleaning pipeline (per play_id x angle)\n")
    L.append(
        "1. **Robust outlier rejection** — drop isolated detections "
        "whose two-sided frame step exceeds `2.5 * rim_width` (detector "
        "teleports onto heads/logos).\n"
        "2. **Constant-acceleration Kalman filter + RTS smoother** on "
        "(x,y); state = (x,vx,ax,y,vy,ay); gravity-aware process noise "
        "(vertical accel std 4.0 > horizontal 1.5).\n"
        "3. **Bounded gap interpolation** — expose the smoother estimate "
        "ONLY inside occlusion gaps <=12 frames bracketed by confident "
        "(conf>=0.30) detections on BOTH sides; longer / unbounded gaps "
        "stay missing and are flagged (never hallucinate through a full "
        "occlusion).\n")
    L.append("## Coverage\n")
    for k, v in cov.items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## v3-added columns\n")
    rationale = {
        "cln_arc_*": "L2 arc-fit recomputed on the cleaned track "
                     "(entry angle, vy@rim, curvature, fit residual, "
                     "apex offset/height). De-jittered => sharper.",
        "cln_bo_*": "L1 bounce-out recomputed on the cleaned track "
                    "(reappearance, outward displacement, then-rises).",
        "cln_min_dist_rw": "closest cleaned ball-rim distance "
                           "(rim-widths).",
        "cln_through_hoop": "clean above->below rim-plane crossing on "
                            "the cleaned track.",
        "cln_frac_inside_rim": "fraction of cleaned points in rim box.",
        "imp_frac": "fraction of usable cleaned points that were "
                    "imputed across a bounded gap (higher => weaker).",
        "n_imputed": "# bridged frames.",
        "n_rejected_outlier": "# raw detections killed as teleports.",
        "track_quality": "per-angle cleaned-track reliability [0,1] = "
                         "(0.6*coverage + 0.4*mean_conf) * "
                         "(1 - 0.5*imputed_frac).",
        "jitter_raw / jitter_clean": "median |2nd-diff| of the y-center "
                                     "before / after cleaning.",
        "jitter_reduction": "jitter_raw - jitter_clean (>0 => "
                            "de-jittered; the cleaning sanity metric).",
        "*_fused / *_max / *_votes (cln_)": "cross-angle aggregates of "
                                            "the cleaned levers.",
        "track_quality_mean/max, imp_frac_mean/min_usable, "
        "n_rejected_outlier_total, jitter_reduction_mean":
            "shot-level imputation/quality so the model can globally "
            "discount unreliable shots.",
    }
    for k, why in rationale.items():
        L.append(f"- `{k}`: {why}")
    L.append("")
    L.append(f"Total feature columns: **{len(feat_cols)}** "
             f"(v3-added: **{len(cln_cols)}**).\n")
    OUT_DOC_V3.write_text("\n".join(L))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="P2 feature dataset")
    ap.add_argument(
        "--version", choices=["v2", "v3"], default="v3",
        help="v3 (default): write data/p2_features_v3.parquet with "
             "cleaned-track L1/L2 + imputation/quality (v1/v2 untouched). "
             "v2: rebuild the v2 file (legacy).")
    args = ap.parse_args(argv)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    if df.empty:
        raise RuntimeError("no shots produced — check input tracks")
    cov = _coverage(df)
    if args.version == "v3":
        df.to_parquet(OUT_PARQUET_V3, index=False)
        _write_doc_v3(df, cov)
        out_pq, out_doc = OUT_PARQUET_V3, OUT_DOC_V3
    else:
        df.to_parquet(OUT_PARQUET, index=False)
        _write_doc(df, cov)
        out_pq, out_doc = OUT_PARQUET, OUT_DOC
    eprint(f"[p2] wrote {out_pq} ({len(df)} shots, {df.shape[1]} cols)")
    eprint(f"[p2] wrote {out_doc}")
    print(f"P2 COVERAGE ({args.version}):")
    for k, v in cov.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
