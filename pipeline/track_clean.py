#!/usr/bin/env python3
"""
P1.5 — ball-track cleaning (iteration 3).

The iteration-2 conclusion was unambiguous: the binding constraint is P1
ball-track jitter plus ball occlusion behind the net / rim (3-8 frame
gaps), which flattens the bounce-out (L1) and arc (L2) trajectory
signals. This module cleans the raw per-frame ball-center series BEFORE
features are recomputed, so the downstream trajectory math sees a
physically-plausible track instead of detector noise.

Pipeline, applied per ``(play_id, angle)`` independently (the ball is one
projectile per shot per camera), all deterministic:

  (a) ROBUST OUTLIER REJECTION. The detector occasionally snaps the
      "ball" onto a head/logo/another ball, producing a single-frame
      teleport. We reject any detection whose frame-to-frame speed
      exceeds ``OUTLIER_K * rim_width`` px/frame (rim width is a stable,
      scale-invariant ruler) when that speed is also inconsistent with
      its neighbours (an isolated spike, not sustained motion). Rejected
      points become missing and are treated like an occlusion gap.

  (b) CONSTANT-ACCELERATION KALMAN FILTER + RTS SMOOTHER. State is
      (x, vx, ax, y, vy, ay). Process noise is gravity-aware: the
      vertical channel is allowed more acceleration freedom than the
      horizontal one (a basketball accelerates ~g downward, drifts
      horizontally). The forward causal filter handles measurement
      noise; a backward Rauch-Tung-Striebel pass smooths the whole
      track (offline is fine — features are computed on a buffered
      window after the fact).

  (c) BOUNDED GAP INTERPOLATION. The smoother naturally produces a
      state estimate for missing frames, but we only TRUST (mark as a
      usable cleaned point) an imputed frame inside a gap of length
      <= ``MAX_IMPUTE_GAP`` that is bounded on BOTH sides by confident
      detections (conf >= ``BOUND_CONF``). Longer gaps, or gaps not
      bracketed by confident detections, are left missing and FLAGGED
      (we never hallucinate a long trajectory through a full occlusion).

Outputs, per input (play_id, angle) frame row, a cleaned track with:
  - ``cx, cy``           cleaned ball-center (NaN where still missing)
  - ``imputed``          1 if this frame's value came from the smoother
                         across a (short, bounded) gap, else 0
  - ``rejected_outlier`` 1 if the raw detection was rejected in (a)
  - ``raw_cx, raw_cy``   the raw center (kept for raw-track features)
  - a per-(play,angle) ``track_quality`` score in [0, 1]

Local CPU only. No AWS, no GPU, no network. Deterministic (no RNG used;
seed pinned for parity with the rest of the pipeline). Pure functions —
importing this module has no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 42

# --- Outlier rejection knob -------------------------------------------
# Max plausible ball travel between consecutive frames, in rim-widths.
# A regulation rim is 18in; a hard NBA shot tops out ~30 mph. At broadcast
# zoom the ball never crosses more than ~a couple rim-widths per frame.
# 2.5 is deliberately permissive — we only want to kill teleports, not
# clip fast-but-real motion. An outlier must ALSO be an isolated spike.
OUTLIER_K = 2.5
# A flagged jump is only rejected if the point's two-sided jump is large
# (a teleport out AND back) rather than a sustained fast pass-through.
OUTLIER_RETURN_FRAC = 0.5

# --- Gap interpolation knobs ------------------------------------------
# Max occlusion gap (frames) we will interpolate across. ~12 frames at
# 30fps ≈ 0.4 s — long enough for the ball to vanish behind the net/rim
# and reappear, short enough that a const-accel model stays valid.
MAX_IMPUTE_GAP = 12
# Both gap edges must be at least this confident for us to trust the
# interpolation (don't bridge between two shaky detections).
BOUND_CONF = 0.30

# --- Kalman / process-noise knobs (gravity-aware) ---------------------
# Measurement noise std (px). The detector box center is good to a few px.
MEAS_STD = 6.0
# Process accel noise std (px/frame^2). Vertical gets more freedom than
# horizontal because real vertical accel (gravity + rim impulse) is
# larger and more variable than horizontal drift.
ACC_STD_X = 1.5
ACC_STD_Y = 4.0
# Initial state covariance (large => trust first measurements quickly).
INIT_VAR = 1e3


@dataclass(frozen=True)
class CleanResult:
    """Per-(play, angle) cleaned track + provenance, immutable."""

    frame_idx: np.ndarray            # int frame indices (sorted)
    cx: np.ndarray                   # cleaned x center (NaN if missing)
    cy: np.ndarray                   # cleaned y center (NaN if missing)
    raw_cx: np.ndarray               # raw x center (NaN if no detection)
    raw_cy: np.ndarray
    imputed: np.ndarray              # 1 = value bridged across a gap
    rejected_outlier: np.ndarray     # 1 = raw detection rejected in (a)
    track_quality: float             # [0,1] reliability of this track


def _rim_width_ruler(g: pd.DataFrame) -> Optional[float]:
    """Robust median rim width (px). The rim is ~static, so its median
    box width is a stable scale-invariant ruler. None if never seen."""
    rw = g["rim_w"].dropna()
    rw = rw[rw > 1.0]
    if rw.empty:
        return None
    return float(np.median(rw))


def _reject_outliers(
    cx: np.ndarray, cy: np.ndarray, ruler: float
) -> np.ndarray:
    """(a) Robust outlier rejection.

    Flag a detection as an outlier iff it is an ISOLATED teleport: the
    step into it AND the step out of it both exceed ``OUTLIER_K`` rim-
    widths, i.e. the point jumps far away from BOTH neighbours (a real
    fast pass-through only has one large step, not a there-and-back).
    Endpoints are compared against their single available neighbour.
    Returns a boolean mask (True = reject), same length as the input.
    """
    n = len(cx)
    rej = np.zeros(n, dtype=bool)
    det = ~np.isnan(cx)
    idx = np.where(det)[0]
    if len(idx) < 3:
        return rej
    thr = OUTLIER_K * ruler
    for a in range(len(idx)):
        i = idx[a]
        prev_i = idx[a - 1] if a > 0 else None
        next_i = idx[a + 1] if a < len(idx) - 1 else None
        d_prev = (
            np.hypot(cx[i] - cx[prev_i], cy[i] - cy[prev_i])
            if prev_i is not None else None
        )
        d_next = (
            np.hypot(cx[i] - cx[next_i], cy[i] - cy[next_i])
            if next_i is not None else None
        )
        if d_prev is not None and d_next is not None:
            # Interior point: teleport out AND back -> reject. If the
            # neighbours are close to each other but far from this point
            # it is a spike; if neighbours straddle a real fast motion
            # the prev->next distance is itself large (skip).
            neigh = np.hypot(cx[next_i] - cx[prev_i],
                             cy[next_i] - cy[prev_i])
            if (d_prev > thr and d_next > thr
                    and neigh < OUTLIER_RETURN_FRAC * (d_prev + d_next)):
                rej[i] = True
        elif d_prev is not None and d_prev > thr:
            rej[i] = True
        elif d_next is not None and d_next > thr:
            rej[i] = True
    return rej


def _kalman_rts(
    z_x: np.ndarray, z_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(b) Constant-acceleration Kalman filter + RTS smoother.

    State s = [x, vx, ax, y, vy, ay]^T, unit time step (per frame). The
    x and y sub-systems are independent (block-diagonal F/Q) but solved
    jointly for code simplicity. ``z_x``/``z_y`` are measurements with
    NaN where there is no detection (those steps are predict-only).
    Returns smoothed (x, y) for every frame (defined everywhere, even
    over gaps — the caller decides which imputed frames to trust).
    """
    n = len(z_x)
    # Per-axis constant-acceleration transition (dt = 1 frame).
    F1 = np.array([[1.0, 1.0, 0.5],
                   [0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0]])
    F = np.zeros((6, 6))
    F[:3, :3] = F1
    F[3:, 3:] = F1
    H = np.zeros((2, 6))
    H[0, 0] = 1.0  # measure x
    H[1, 3] = 1.0  # measure y

    def _Q(acc_std: float) -> np.ndarray:
        # Discrete white-noise-acceleration covariance for a 3-state
        # const-accel block (dt = 1).
        q = acc_std ** 2
        return q * np.array([[0.25, 0.5, 0.5],
                             [0.50, 1.0, 1.0],
                             [0.50, 1.0, 1.0]])

    Q = np.zeros((6, 6))
    Q[:3, :3] = _Q(ACC_STD_X)
    Q[3:, 3:] = _Q(ACC_STD_Y)
    R = np.eye(2) * (MEAS_STD ** 2)

    # Initialise at the first available measurement.
    det = ~np.isnan(z_x)
    first = int(np.argmax(det)) if det.any() else 0
    s = np.zeros(6)
    s[0] = z_x[first] if det.any() else 0.0
    s[3] = z_y[first] if det.any() else 0.0
    P = np.eye(6) * INIT_VAR

    xs_pred = np.zeros((n, 6))
    Ps_pred = np.zeros((n, 6, 6))
    xs_filt = np.zeros((n, 6))
    Ps_filt = np.zeros((n, 6, 6))

    for k in range(n):
        # Predict.
        s = F @ s
        P = F @ P @ F.T + Q
        xs_pred[k] = s
        Ps_pred[k] = P
        # Update only if this frame has a (non-rejected) measurement.
        if det[k]:
            z = np.array([z_x[k], z_y[k]])
            y = z - H @ s
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            s = s + K @ y
            P = (np.eye(6) - K @ H) @ P
        xs_filt[k] = s
        Ps_filt[k] = P

    # RTS backward smoothing pass.
    xs_smooth = xs_filt.copy()
    Ps_smooth = Ps_filt.copy()
    for k in range(n - 2, -1, -1):
        Pp = Ps_pred[k + 1]
        try:
            C = Ps_filt[k] @ F.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            C = Ps_filt[k] @ F.T @ np.linalg.pinv(Pp)
        xs_smooth[k] = (
            xs_filt[k] + C @ (xs_smooth[k + 1] - xs_pred[k + 1]))
        Ps_smooth[k] = (
            Ps_filt[k] + C @ (Ps_smooth[k + 1] - Pp) @ C.T)
    return xs_smooth[:, 0], xs_smooth[:, 3]


def _trusted_gap_mask(
    det_after_outlier: np.ndarray, conf: np.ndarray
) -> np.ndarray:
    """(c) Which missing frames may be filled from the smoother.

    A missing frame is trustworthy iff it lies in a contiguous gap of
    length <= MAX_IMPUTE_GAP that is bounded on BOTH sides by a
    detection with conf >= BOUND_CONF. Leading / trailing gaps (no
    detection on one side) are never trusted — we don't extrapolate a
    trajectory off the ends. Returns a boolean mask of frames whose
    smoothed value we will expose as a cleaned (imputed) point.
    """
    n = len(det_after_outlier)
    trust = np.zeros(n, dtype=bool)
    k = 0
    while k < n:
        if det_after_outlier[k]:
            k += 1
            continue
        j = k
        while j < n and not det_after_outlier[j]:
            j += 1
        # Gap is [k, j-1]; left edge k-1, right edge j.
        gap_len = j - k
        left_ok = (
            k - 1 >= 0 and det_after_outlier[k - 1]
            and not np.isnan(conf[k - 1]) and conf[k - 1] >= BOUND_CONF)
        right_ok = (
            j < n and det_after_outlier[j]
            and not np.isnan(conf[j]) and conf[j] >= BOUND_CONF)
        if gap_len <= MAX_IMPUTE_GAP and left_ok and right_ok:
            trust[k:j] = True
        k = j
    return trust


def _quality(
    n_frames: int, det_raw: np.ndarray, rejected: np.ndarray,
    trusted: np.ndarray, conf: np.ndarray, ruler: Optional[float],
) -> float:
    """Per-track reliability in [0, 1] for the model to discount
    unreliable shots. Combines four interpretable factors:
      - detection coverage (raw detections kept after outlier removal),
      - mean confidence of kept detections,
      - imputation penalty (more bridged frames => less trustworthy),
      - a hard zero if no rim ruler exists (every distance is undefined).
    """
    if ruler is None or n_frames == 0:
        return 0.0
    kept = det_raw & ~rejected
    cov = float(kept.mean())
    cgood = conf[kept]
    cgood = cgood[~np.isnan(cgood)]
    mconf = float(np.clip(cgood.mean(), 0.0, 1.0)) if cgood.size else 0.0
    imp_frac = float(trusted.mean())
    # Coverage and confidence are the backbone; imputation linearly
    # discounts (a track that is mostly bridged is weak evidence).
    q = (0.6 * cov + 0.4 * mconf) * (1.0 - 0.5 * imp_frac)
    return float(np.clip(q, 0.0, 1.0))


def clean_track(g: pd.DataFrame) -> CleanResult:
    """Clean ONE (play_id, angle) frame-sorted slice.

    ``g`` must contain frame_idx, ball_x/y/w/h, ball_conf, rim_w. It is
    not mutated. Always returns a dense CleanResult aligned to the
    sorted unique frame grid of ``g``.
    """
    g = g.sort_values("frame_idx")
    fidx = g["frame_idx"].to_numpy().astype(int)
    bx = g["ball_x"].to_numpy(dtype=float)
    by = g["ball_y"].to_numpy(dtype=float)
    bw = g["ball_w"].to_numpy(dtype=float)
    bh = g["ball_h"].to_numpy(dtype=float)
    conf = g["ball_conf"].to_numpy(dtype=float)
    n = len(fidx)

    raw_cx = bx + bw / 2.0
    raw_cy = by + bh / 2.0

    ruler = _rim_width_ruler(g)
    if n == 0:
        empty = np.array([], dtype=float)
        return CleanResult(
            frame_idx=np.array([], dtype=int), cx=empty, cy=empty,
            raw_cx=empty, raw_cy=empty,
            imputed=np.array([], dtype=int),
            rejected_outlier=np.array([], dtype=int),
            track_quality=0.0)

    det_raw = ~np.isnan(raw_cx)

    # (a) Robust outlier rejection (needs a ruler; skip if no rim).
    if ruler is not None:
        rejected = _reject_outliers(
            np.where(det_raw, raw_cx, np.nan),
            np.where(det_raw, raw_cy, np.nan), ruler)
    else:
        rejected = np.zeros(n, dtype=bool)

    # Measurements fed to the filter: detections that survived (a).
    keep = det_raw & ~rejected
    z_x = np.where(keep, raw_cx, np.nan)
    z_y = np.where(keep, raw_cy, np.nan)

    if keep.sum() == 0:
        # Nothing usable — return raw passthrough, zero quality.
        nanv = np.full(n, np.nan)
        return CleanResult(
            frame_idx=fidx, cx=nanv.copy(), cy=nanv.copy(),
            raw_cx=raw_cx, raw_cy=raw_cy,
            imputed=np.zeros(n, dtype=int),
            rejected_outlier=rejected.astype(int),
            track_quality=0.0)

    # (b) Kalman filter + RTS smoother over the whole window.
    sx, sy = _kalman_rts(z_x, z_y)

    # (c) Decide which frames are exposed as cleaned points.
    trusted_gap = _trusted_gap_mask(keep, conf)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    imputed = np.zeros(n, dtype=int)
    # Real (surviving) detections -> use the smoothed value (denoised
    # but anchored to data). This is the de-jitter benefit.
    cx[keep] = sx[keep]
    cy[keep] = sy[keep]
    # Trusted short bounded gaps -> expose the smoother estimate, flagged.
    fill = trusted_gap & ~keep
    cx[fill] = sx[fill]
    cy[fill] = sy[fill]
    imputed[fill] = 1
    # Everything else (long / unbounded gaps, rejected outliers) stays
    # NaN: we do NOT hallucinate the ball through a real occlusion.

    quality = _quality(n, det_raw, rejected, fill, conf, ruler)

    return CleanResult(
        frame_idx=fidx, cx=cx, cy=cy,
        raw_cx=raw_cx, raw_cy=raw_cy,
        imputed=imputed,
        rejected_outlier=rejected.astype(int),
        track_quality=quality)


def jitter_metric(centers: np.ndarray) -> float:
    """Median absolute 2nd difference of a 1-D series (NaNs dropped).

    The 2nd difference is ~zero for a smooth constant-acceleration
    trajectory and large for frame-to-frame detector jitter, so its
    median magnitude is a clean de-noising sanity metric. Returns NaN if
    fewer than 3 valid points.
    """
    v = centers[~np.isnan(centers)]
    if v.size < 3:
        return float("nan")
    d2 = np.diff(v, n=2)
    return float(np.median(np.abs(d2)))


def clean_game(df: pd.DataFrame) -> Dict[Tuple[str, str], CleanResult]:
    """Clean every (play_id, angle) track in a game's raw parquet.

    Returns a dict keyed by (play_id, angle). Deterministic; no RNG.
    """
    np.random.seed(SEED)  # parity only — no stochastic step is used
    out: Dict[Tuple[str, str], CleanResult] = {}
    for (pid, ang), g in df.groupby(["play_id", "angle"], sort=True):
        out[(str(pid), str(ang))] = clean_track(g)
    return out


def _self_check() -> None:
    """Synthetic sanity check: a noisy, partially-occluded parabola
    should come out smoother (lower median |2nd-diff|) and the bounded
    gap should be filled while a long gap is left missing."""
    rng = np.random.default_rng(SEED)
    n = 60
    fidx = np.arange(n)
    t = fidx.astype(float)
    true_x = 50.0 + 4.0 * t
    true_y = 300.0 - 10.0 * t + 0.4 * t ** 2  # gravity-like parabola
    obs_x = true_x + rng.normal(0, 6, n)
    obs_y = true_y + rng.normal(0, 6, n)
    conf = np.full(n, 0.8)
    # Short bounded occlusion (len 6) and a long one (len 20).
    obs_x[20:26] = np.nan
    obs_y[20:26] = np.nan
    obs_x[35:55] = np.nan
    obs_y[35:55] = np.nan
    # A teleport outlier.
    obs_x[10] = 1800.0
    obs_y[10] = 50.0
    g = pd.DataFrame({
        "frame_idx": fidx,
        "ball_x": obs_x - 10.0, "ball_y": obs_y - 10.0,
        "ball_w": np.where(np.isnan(obs_x), np.nan, 20.0),
        "ball_h": np.where(np.isnan(obs_y), np.nan, 20.0),
        "ball_conf": np.where(np.isnan(obs_x), np.nan, conf),
        "rim_w": 100.0,
    })
    res = clean_track(g)
    raw_j = jitter_metric(res.raw_cy)
    cln_j = jitter_metric(res.cy)
    print(f"[self-check] outlier@10 rejected="
          f"{bool(res.rejected_outlier[10])}")
    print(f"[self-check] short gap[20:26] imputed="
          f"{res.imputed[20:26].tolist()}")
    print(f"[self-check] long gap[35:55] imputed sum="
          f"{int(res.imputed[35:55].sum())} (expect 0)")
    print(f"[self-check] jitter raw={raw_j:.3f} cleaned={cln_j:.3f} "
          f"(cleaned should be lower)")
    print(f"[self-check] track_quality={res.track_quality:.3f}")
    assert bool(res.rejected_outlier[10]), "outlier not rejected"
    assert res.imputed[20:26].sum() == 6, "short gap not filled"
    assert res.imputed[35:55].sum() == 0, "long gap hallucinated"
    assert cln_j < raw_j, "cleaning did not reduce jitter"
    print("[self-check] OK")


if __name__ == "__main__":
    _self_check()
