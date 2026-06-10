#!/usr/bin/env python3
"""Per-game FR<->NR sync from the per-shot clip pairs' AUDIO — zero extra
S3 downloads (deep -ss audio seeks over presigned URLs are unreliable).

Clips are extracted with an assumed sync offset baked into the NR cut, so
cross-correlating each pair's audio measures the RESIDUAL error:
    true_offset = baked_offset + cluster(residuals)
~100 pairs per game feed the density-cluster estimator -> robust to music.

Usage:
  python pipeline/clip_audio_sync.py --clips-dir <dir> --baked 13 [--limit 60]
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
MAX_LAG_S = 1.5


def wav_of(clip: Path, out: Path) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(clip), "-vn", "-ac", "1", "-ar", str(SR),
                        str(out)], capture_output=True, timeout=60)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 1000


def load(p: Path) -> np.ndarray:
    _, x = wavfile.read(p)
    x = np.diff(x.astype(np.float64))
    s = x.std()
    return x / s if s > 0 else x


def xcorr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    c = np.fft.irfft(B * np.conj(A), nfft)
    c = np.concatenate([c[-len(a) + 1:], c[:len(b)]])
    lags = np.arange(-len(a) + 1, len(b))
    keep = np.abs(lags) <= int(MAX_LAG_S * SR)
    c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    return lags[i] / SR, float(c[i] / (np.median(np.abs(c)) + 1e-9))


def cluster_estimate(vals: list, tol: float = 1.0) -> tuple[float, int]:
    arr = np.asarray(vals, dtype=float)
    best_n, best_mean = -1, 0.0
    for c in arr:
        sel = arr[np.abs(arr - c) <= tol]
        if len(sel) > best_n:
            best_n, best_mean = len(sel), float(sel.mean())
    return best_mean, best_n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--baked", type=float, required=True,
                    help="sync frames baked into the NR clips at extraction")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-peak", type=float, default=8.0)
    args = ap.parse_args()
    clips = Path(args.clips_dir)
    pairs = []
    for fr in sorted(clips.glob("*_FR.mp4")):
        nr = clips / fr.name.replace("_FR.mp4", "_NR.mp4")
        if nr.exists():
            pairs.append((fr, nr))
    pairs = pairs[:args.limit]
    print(f"{len(pairs)} clip pairs (baked offset {args.baked:+.0f} frames)")

    residuals = []
    with tempfile.TemporaryDirectory() as td:
        for fr, nr in pairs:
            wa, wb = Path(td) / "a.wav", Path(td) / "b.wav"
            try:
                if not (wav_of(fr, wa) and wav_of(nr, wb)):
                    continue
                lag_s, peak = xcorr(load(wa), load(wb))
            except Exception:
                continue
            if peak >= args.min_peak:
                residuals.append(lag_s * FPS)
    if not residuals:
        print("no usable pairs")
        return 1
    est, support = cluster_estimate(residuals)
    print(f"usable pairs: {len(residuals)}  "
          f"residual cluster: {est:+.2f} frames "
          f"({support}/{len(residuals)} within ±1)")
    print(f"TRUE OFFSET = {args.baked:+.0f} + {est:+.2f} = "
          f"{args.baked + est:+.2f} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
