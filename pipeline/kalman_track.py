#!/usr/bin/env python3
"""2D constant-velocity Kalman filter for per-camera ball tracking.

Designed to fill detection gaps so that the triangulation pipeline has more
frames where both FR and NR ball positions are available (raw or predicted).

Usage:
    detections = [(frame_idx, px, py, conf), ...]   # may have gaps
    smoothed = kalman_smooth(detections, all_frames)
    # smoothed = [(frame_idx, px, py, sigma, was_observed), ...]

The filter's process noise is tuned for a basketball in flight: typical
horizontal accelerations near 0 plus gravity (image-space gravity depends on
camera tilt, but the y-acceleration in pixels is approximately the projection
of 9.8 m/s² down — for ~1000 px/m at FR's distance ≈ 10 px/s²).

We use forward filter + backward RTS smoother for best gap interpolation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Track:
    frame: int
    t: float
    px: float
    py: float
    sigma: float       # √trace(P[:2,:2]) — position uncertainty in pixels
    observed: bool     # True if a YOLO detection landed here, False if predicted


def _F(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix."""
    return np.array([[1.0, 0.0,  dt, 0.0],
                     [0.0, 1.0, 0.0,  dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def _Q(dt: float, q_acc: float) -> np.ndarray:
    """Process noise covariance for a constant-velocity model with white-noise
    acceleration q_acc² per unit time, in image-pixel space."""
    dt2 = dt*dt
    dt3 = dt2*dt
    dt4 = dt2*dt2
    return q_acc * q_acc * np.array([
        [dt4/4, 0.0,   dt3/2, 0.0  ],
        [0.0,   dt4/4, 0.0,   dt3/2],
        [dt3/2, 0.0,   dt2,   0.0  ],
        [0.0,   dt3/2, 0.0,   dt2  ],
    ])


_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def kalman_smooth(detections: list[tuple],
                  all_frames: list[tuple[int, float]],
                  meas_sigma_px: float = 6.0,
                  q_acc_px_s2: float = 1500.0,
                  max_predict_gap_s: float = 0.25,
                  outlier_gate_sigma: float = 4.0) -> list[Track]:
    """Forward + RTS-backward smoother on a sparse pixel-track.

    detections : list of (frame_idx, time_s, px, py, conf), sorted by frame
    all_frames : list of (frame_idx, time_s) — every frame to produce output for
    meas_sigma_px : per-axis pixel noise from YOLO
    q_acc_px_s2 : process-noise "acceleration" (cm/s² in image-pixel units)
    max_predict_gap_s : longest gap we'll fill with prediction; beyond that,
                       output sigma=inf so the caller can drop the frame
    outlier_gate_sigma : Mahalanobis-distance gate; reject detections that look
                       like detector errors (e.g. grabbed a different ball)
    """
    if not all_frames or not detections:
        return [Track(f, t, np.nan, np.nan, np.inf, False) for f, t in all_frames]

    # Initialize from first detection
    f0, t0, px0, py0, _ = detections[0]
    x = np.array([px0, py0, 0.0, 0.0])
    P = np.diag([meas_sigma_px**2, meas_sigma_px**2, 1e6, 1e6])

    R = np.eye(2) * meas_sigma_px * meas_sigma_px

    # Bookkeep for RTS smoother
    f_states_pred: list[tuple[float, np.ndarray, np.ndarray]] = []  # (t, x_pred, P_pred)
    f_states_post: list[tuple[float, np.ndarray, np.ndarray, bool]] = []  # post-update

    det_iter = iter(detections)
    next_det = next(det_iter, None)

    # Move detections cursor up to (or just before) first all_frames entry
    while next_det is not None and next_det[0] < all_frames[0][0]:
        next_det = next(det_iter, None)

    t_prev = t0
    for f_idx, t_now in all_frames:
        dt = max(t_now - t_prev, 1e-3)
        # Predict
        F = _F(dt); Q = _Q(dt, q_acc_px_s2)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        f_states_pred.append((t_now, x_pred.copy(), P_pred.copy()))

        # Find detection at this exact frame (if any)
        det_here = None
        while next_det is not None and next_det[0] == f_idx:
            det_here = next_det
            next_det = next(det_iter, None)
        # Skip any detections that belong to past frames (shouldn't happen with sorted input)
        while next_det is not None and next_det[0] < f_idx:
            next_det = next(det_iter, None)

        if det_here is not None:
            _, _, dpx, dpy, _ = det_here
            z = np.array([dpx, dpy])
            y_innov = z - _H @ x_pred
            S = _H @ P_pred @ _H.T + R
            # Outlier gate (Mahalanobis distance)
            try:
                md2 = float(y_innov @ np.linalg.inv(S) @ y_innov)
            except np.linalg.LinAlgError:
                md2 = 0.0
            if md2 < outlier_gate_sigma * outlier_gate_sigma:
                K = P_pred @ _H.T @ np.linalg.inv(S)
                x = x_pred + K @ y_innov
                P = (np.eye(4) - K @ _H) @ P_pred
                observed = True
            else:
                # Treat as missed measurement — keep prediction
                x = x_pred; P = P_pred; observed = False
        else:
            x = x_pred; P = P_pred; observed = False

        f_states_post.append((t_now, x.copy(), P.copy(), observed))
        t_prev = t_now

    # ---- RTS backward pass for smoother estimates in the gaps ----
    n = len(f_states_post)
    smoothed = [None] * n
    x_s = f_states_post[-1][1].copy()
    P_s = f_states_post[-1][2].copy()
    smoothed[-1] = (f_states_post[-1][0], x_s.copy(), P_s.copy(), f_states_post[-1][3])
    for k in range(n - 2, -1, -1):
        t_k, x_k, P_k, obs_k = f_states_post[k]
        t_kp1, x_pred_kp1, P_pred_kp1 = f_states_pred[k + 1]
        dt = max(t_kp1 - t_k, 1e-3)
        F = _F(dt)
        try:
            C = P_k @ F.T @ np.linalg.inv(P_pred_kp1)
        except np.linalg.LinAlgError:
            smoothed[k] = (t_k, x_k.copy(), P_k.copy(), obs_k); continue
        x_s = x_k + C @ (x_s - x_pred_kp1)
        P_s = P_k + C @ (P_s - P_pred_kp1) @ C.T
        smoothed[k] = (t_k, x_s.copy(), P_s.copy(), obs_k)

    # ---- Decide which frames to expose ----
    # Detection times set:
    det_times = {d[0] for d in detections}
    # Build per-frame nearest detection-distance in seconds for the
    # max_predict_gap_s cap
    det_t_array = np.array([d[1] for d in detections])

    out: list[Track] = []
    for (f_idx, t_now), (_, xs, Ps, obs) in zip(all_frames, smoothed):
        sigma = float(np.sqrt(max(Ps[0, 0] + Ps[1, 1], 0.0)) / np.sqrt(2.0))
        # Distance (in time) to nearest detection
        if len(det_t_array):
            nearest = float(np.min(np.abs(det_t_array - t_now)))
        else:
            nearest = np.inf
        # Drop frames where we'd be predicting past the gap budget
        if nearest > max_predict_gap_s and not obs:
            sigma = np.inf
        out.append(Track(f_idx, t_now, float(xs[0]), float(xs[1]),
                         sigma, observed=obs))
    return out


if __name__ == "__main__":
    # Quick smoke test
    dets = [(0, 0.00, 100.0, 200.0, 0.9),
            (1, 0.03, 110.0, 210.0, 0.9),
            (2, 0.06, 121.0, 219.0, 0.9),
            # gap 3..5
            (6, 0.20, 180.0, 240.0, 0.9),
            (7, 0.23, 190.0, 244.0, 0.9)]
    frames = [(i, i * 0.0333) for i in range(8)]
    smoothed = kalman_smooth(dets, frames)
    for s in smoothed:
        print(f"f={s.frame:>2} t={s.t:.3f}  ({s.px:7.1f},{s.py:7.1f}) "
              f"sigma={s.sigma:6.2f}  observed={s.observed}")
