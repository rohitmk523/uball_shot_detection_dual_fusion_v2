#!/usr/bin/env python3
"""Audio rim-contact signal probe on existing fresh-game footage.

Tests whether the audio (clean swish vs metallic rim clang vs silence/airball)
is independent of the visual depth illusion and provides discriminative make/
miss signal that the visual fusion can't see. NEAR-side camera audio is used
(closest mic to the play's hoop -> best SNR).

For each fresh shot we:
  - find the rim-cross time in the NEAR camera tracks (so audio + video share
    the same clock - no sync offset needed),
  - slice a 2.0 s window around it (0.5 s pre + 1.5 s post),
  - compute simple band-energy + onset + centroid features,
and then quantify make-vs-miss separation (single-feature AUC) plus compare
the FALSE-POSITIVE shots (depth illusion) against TRUE positives.
"""
from __future__ import annotations
import json, subprocess, sys, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from sync_aware_fusion import event_frame                    # noqa: E402

AUDIO_DIR = Path("/tmp/audio_fresh"); AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRACKS = Path("/tmp/p1tracks_fresh")
PLAYS_META = Path("/tmp/plays_meta.json")
GT_FRESH = Path("/tmp/gt_fresh.json")

SR = 16000
WIN_BEFORE = 0.5
WIN_AFTER = 1.5
FPS = 30000 / 1001

FEATS = ["rms", "peak", "onset_peak",
         "e_50_500", "e_500_2k", "e_2k_5k", "e_5k_8k",
         "centroid", "peak_time_norm"]


def fresh_games() -> list:
    m = json.loads((ROOT / "data" / "games_manifest.json").read_text())
    return [g for g in m["games"] if g.get("split") == "fresh"]


def video_uri(game, cam):
    return (f"s3://uball-videos-production/{game['s3_prefix']}"
            f"{game['date']}_{game['game_id'][:23]}_{cam}.mp4")


def presign(u):
    return subprocess.run(["aws", "s3", "presign", u, "--expires-in", "10800"],
                          capture_output=True, text=True, check=True).stdout.strip()


def fetch_audio(args):
    """aws s3 cp the mp4 locally (multi-part, fast), then ffmpeg-extract the
    audio (fast local I/O). ffmpeg-over-HTTP pulled the whole file at modest
    rate -> timed out at 15 min; multi-part S3 cp is ~5-10x faster, and the
    local extract is seconds."""
    game, cam = args
    dst = AUDIO_DIR / f"{game['game_id']}_{cam}.wav"
    if dst.exists() and dst.stat().st_size > 5_000_000:
        return f"cached {game['game_id'][:8]} {cam}"
    uri = video_uri(game, cam)
    tmp_mp4 = AUDIO_DIR / f"_dl_{game['game_id']}_{cam}.mp4"
    subprocess.run(["aws", "s3", "cp", uri, str(tmp_mp4), "--quiet"],
                   check=True, timeout=1800)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(tmp_mp4), "-vn", "-ac", "1", "-ar", str(SR),
                    "-acodec", "pcm_s16le", str(dst)],
                   check=True, timeout=300)
    tmp_mp4.unlink()
    return f"fetched {game['game_id'][:8]} {cam} ({dst.stat().st_size/1e6:.0f} MB wav)"


def features(x: np.ndarray, sr: int) -> dict:
    if x is None or len(x) < sr // 10:
        return {k: np.nan for k in FEATS}
    n = len(x)
    out = {"rms": float(np.sqrt(np.mean(x ** 2))),
           "peak": float(np.max(np.abs(x)))}
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bands = {"e_50_500": (50, 500), "e_500_2k": (500, 2000),
             "e_2k_5k": (2000, 5000), "e_5k_8k": (5000, 8000)}
    raw = {k: float(np.sum(spec[(freqs >= lo) & (freqs < hi)]))
           for k, (lo, hi) in bands.items()}
    total = sum(raw.values()) + 1e-12
    out.update({k: v / total for k, v in raw.items()})
    out["centroid"] = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    hop = sr // 50
    rms_t = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2))
                      for i in range(0, len(x) - hop, hop)])
    out["onset_peak"] = float(np.max(rms_t)) if len(rms_t) else np.nan
    out["peak_time_norm"] = float(np.argmax(rms_t) / max(1, len(rms_t))) if len(rms_t) else np.nan
    return out


def near_rim_time(gid: str, pid: str, side: str) -> float | None:
    df = pd.read_parquet(TRACKS / f"{gid}.parquet")
    cam = "NL" if side == "LEFT" else "NR"
    g = df[(df.play_id == pid) & (df.angle == cam)].sort_values("frame_idx")
    fr = event_frame(g)
    return float(fr / FPS) if fr is not None else None


def main():
    games = fresh_games()
    plays = json.loads(PLAYS_META.read_text()) if PLAYS_META.exists() else {}
    gt = json.loads(GT_FRESH.read_text())["games"]
    side_of = {pid: p["angle"] for pid, p in plays.items()}
    for gid, shots in gt.items():
        for s in shots:
            side_of.setdefault(s["play_id"], s.get("angle", "LEFT"))

    # ---- 1. sequential audio download (5 games x 2 near cams = 10) ----
    # Sequential because each `aws s3 cp` already does multi-part parallel
    # within one file; running multiple in parallel just contends for bandwidth
    # and makes ffmpeg time out (the previous attempt hit the 15min cap).
    print(f"[audio] fetching {len(games)*2} audio tracks (NL+NR per game) ...")
    todo = [(g, c) for g in games for c in ("NL", "NR")]
    for job in todo:
        print(f"  {fetch_audio(job)}")

    # ---- 2. load audio, compute per-shot features ----
    cache = {}
    for g in games:
        for c in ("NL", "NR"):
            sr, x = wavfile.read(str(AUDIO_DIR / f"{g['game_id']}_{c}.wav"))
            cache[(g["game_id"], c)] = (sr, x.astype(np.float32) / 32768.0)
    pred = pd.read_parquet(ROOT / "data" / "p3_fresh_predictions.parquet")
    pred["play_id"] = pred.play_id.astype(str)

    rows = []
    n_skip = 0
    for _, r in pred.iterrows():
        side = side_of.get(r.play_id, "LEFT")
        cam = "NL" if side == "LEFT" else "NR"
        sr, x = cache[(r.game_id, cam)]
        t = near_rim_time(r.game_id, r.play_id, side)
        rec = {"game_id": r.game_id, "play_id": r.play_id, "cam": cam}
        if t is None:
            rec.update({k: np.nan for k in FEATS}); rec["why"] = "no_rim_event"
            n_skip += 1; rows.append(rec); continue
        s_idx = int((t - WIN_BEFORE) * sr); e_idx = int((t + WIN_AFTER) * sr)
        if s_idx < 0 or e_idx >= len(x):
            rec.update({k: np.nan for k in FEATS}); rec["why"] = "out_of_range"
            n_skip += 1; rows.append(rec); continue
        rec.update(features(x[s_idx:e_idx], sr)); rec["why"] = "ok"
        rows.append(rec)
    af = pd.DataFrame(rows)
    af.to_csv(ROOT / "data" / "audio_features_fresh.csv", index=False)
    print(f"[audio] features computed for {len(af)-n_skip}/{len(af)} shots "
          f"(skipped {n_skip})")

    # ---- 3. separation analysis ----
    from sklearn.metrics import roc_auc_score
    j = pred.merge(af, on=["game_id", "play_id"], how="left")
    j = j[j.why == "ok"]
    print(f"\n=== AUDIO FEATURE SEPARATION  (MAKE vs MISS,  n={len(j)}) ===")
    print(f"{'feature':<18} {'MAKE mean':>11} {'MISS mean':>11} {'AUC':>6}")
    for k in FEATS:
        s = j[j[k].notna()]
        if s.label.nunique() < 2: continue
        mu_m = s[s.label == 1][k].mean(); mu_s = s[s.label == 0][k].mean()
        auc = roc_auc_score(s.label, s[k])
        # flip if AUC <0.5 (feature predicts miss better than make)
        auc = max(auc, 1 - auc)
        print(f"  {k:<18} {mu_m:>11.4f} {mu_s:>11.4f} {auc:>6.3f}")

    print(f"\n=== FP (depth-illusion misses called MAKE) vs TP (true makes) ===")
    fps = j[(j.label == 0) & (j.pred == 1)]
    tps = j[(j.label == 1) & (j.pred == 1)]
    print(f"  FP n={len(fps)}   TP n={len(tps)}")
    print(f"{'feature':<18} {'FP mean':>11} {'TP mean':>11} {'diff':>9}")
    for k in FEATS:
        a, b = fps[k].dropna(), tps[k].dropna()
        if len(a) == 0 or len(b) == 0: continue
        print(f"  {k:<18} {a.mean():>11.4f} {b.mean():>11.4f} "
              f"{a.mean() - b.mean():>+9.4f}")

    # ---- 4. quick "veto" rule: can audio flip high-conf FPs without hurting TPs ----
    print(f"\n=== VETO POTENTIAL: would a quiet-audio veto flip FPs? ===")
    j["loudness"] = j["onset_peak"]
    for pct in (10, 20, 30):
        thr = np.percentile(j["loudness"].dropna(), pct)
        flips_in_fps = (fps["onset_peak"] < thr).sum()
        flips_in_tps = (tps["onset_peak"] < thr).sum()
        print(f"  onset_peak < p{pct} ({thr:.3f}): would flip {flips_in_fps}/{len(fps)} FPs  "
              f"and accidentally kill {flips_in_tps}/{len(tps)} TPs")


if __name__ == "__main__":
    main()
