#!/usr/bin/env python3
"""Automatic FR<->NR sync via audio cross-correlation.

Both cameras record the same court audio. The same event appears at NR file
time = FR file time + k/fps when NR_frame = FR_frame + k (Rohit's manual
frame-marking convention). Cross-correlating the two audio tracks therefore
gives k automatically, to ~1 frame (acoustic path differences between the
two mic positions contribute <1.5 frames; verified against manual marks).

Usage:
  python pipeline/audio_sync_detect.py --video-a FR.mp4 --video-b NR.mp4 \
      --windows 500,1300,2200 [--win-len 40]
Prints per-window offsets and the median.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FPS = 29.97
SR = 16000
MAX_LAG_S = 2.0


def extract_wav(video: str, t0: float, dur: float, out: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{t0:.2f}", "-i", video, "-t", f"{dur:.2f}",
           "-vn", "-ac", "1", "-ar", str(SR), str(out)]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 1000


def load(p: Path) -> np.ndarray:
    _, x = wavfile.read(p)
    x = x.astype(np.float64)
    x -= x.mean()
    # emphasize transients (ball bounces, rim, whistle): first difference
    x = np.diff(x)
    s = x.std()
    return x / s if s > 0 else x


def xcorr_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Lag (seconds) of b relative to a, by FFT cross-correlation.
    Positive = the same event appears LATER in b's file."""
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    corr = np.fft.irfft(B * np.conj(A), nfft)
    corr = np.concatenate([corr[-len(a) + 1:], corr[:len(b)]])
    lags = np.arange(-len(a) + 1, len(b))
    keep = np.abs(lags) <= int(MAX_LAG_S * SR)
    corr, lags = corr[keep], lags[keep]
    i = int(np.argmax(corr))
    peak = corr[i] / (np.median(np.abs(corr)) + 1e-9)   # peak prominence
    return lags[i] / SR, float(peak)


def measure(video_a: str, video_b: str, windows: list[float],
            win_len: float = 40.0) -> list[dict]:
    out = []
    with tempfile.TemporaryDirectory() as td:
        for t0 in windows:
            wa, wb = Path(td) / "a.wav", Path(td) / "b.wav"
            if not (extract_wav(video_a, t0, win_len, wa)
                    and extract_wav(video_b, t0, win_len, wb)):
                out.append(dict(t0=t0, ok=False))
                continue
            lag_s, peak = xcorr_offset(load(wa), load(wb))
            out.append(dict(t0=t0, ok=True, lag_s=lag_s,
                            frames=lag_s * FPS, peak=peak))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-a", required=True, help="reference (FR)")
    ap.add_argument("--video-b", required=True, help="other (NR)")
    ap.add_argument("--windows", required=True,
                    help="comma-separated start times (s)")
    ap.add_argument("--win-len", type=float, default=40.0)
    args = ap.parse_args()
    windows = [float(w) for w in args.windows.split(",")]
    rows = measure(args.video_a, args.video_b, windows, args.win_len)
    frames = []
    for r in rows:
        if not r["ok"]:
            print(f"  t0={r['t0']:8.1f}  EXTRACT FAIL")
            continue
        print(f"  t0={r['t0']:8.1f}  lag={r['lag_s']*1000:+8.1f}ms "
              f"= {r['frames']:+6.2f} frames  (peak {r['peak']:.0f}x)")
        frames.append(r["frames"])
    if frames:
        est, support = cluster_estimate(frames)
        print(f"median offset: {np.median(frames):+.2f} frames (n={len(frames)})")
        print(f"CLUSTER offset: {est:+.2f} frames "
              f"({support}/{len(frames)} windows within ±1.25)")
    return 0


def cluster_estimate(frames: list, tol: float = 1.25) -> tuple[float, int]:
    """Densest-cluster estimator: music creates scattered false peaks, but
    true-offset windows agree to ~±1 frame. For each window, count
    neighbours within tol; the largest neighbourhood's mean is the offset.
    (Validated 2026-06-10 vs 5 manual marks: ±1 frame.)"""
    arr = np.asarray(frames, dtype=float)
    best_n, best_mean = -1, 0.0
    for c in arr:
        sel = arr[np.abs(arr - c) <= tol]
        if len(sel) > best_n:
            best_n, best_mean = len(sel), float(sel.mean())
    return best_mean, best_n


if __name__ == "__main__":
    sys.exit(main())
