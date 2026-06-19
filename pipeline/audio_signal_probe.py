#!/usr/bin/env python3
"""Single-game audio probe (b3c1f62c only) — feasibility test for whether
swish/clang/silence audio carries make-miss signal independent of the visual
depth illusion. Uses the NR camera audio for ALL shots (LEFT plays included)
because only NR audio was successfully downloaded for this game. Imperfect
SNR for LEFT plays but a same-gym mic still hears the action.

If a signal shows up here, justify pulling the other 9 audio tracks (slow on
the current ~2.5 MB/s link). If not, audio likely won't break the ceiling on
existing footage.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from sync_aware_fusion import event_frame                    # noqa: E402
from audio_signal import features, FEATS                     # noqa: E402

GID = "b3c1f62c-1a02-47c9-8d2a-4e05a27dc14d"
AUDIO = Path(f"/tmp/audio_fresh/{GID}_NR.wav")
TRACKS = Path("/tmp/p1tracks_fresh")
WIN_BEFORE = 0.5
WIN_AFTER = 1.5
FPS = 30000 / 1001


def near_event_time(pid: str) -> float | None:
    """rim-cross time in NEAR-RIGHT camera (whose clock matches NR audio)."""
    df = pd.read_parquet(TRACKS / f"{GID}.parquet")
    g = df[(df.play_id == pid) & (df.angle == "NR")].sort_values("frame_idx")
    fr = event_frame(g)
    return float(fr / FPS) if fr is not None else None


def main():
    sr, x = wavfile.read(str(AUDIO))
    x = x.astype(np.float32) / 32768.0
    print(f"[audio] {GID[:8]} NR  sr={sr}Hz dur={len(x)/sr:.1f}s")

    pred = pd.read_parquet(ROOT / "data" / "p3_fresh_predictions.parquet")
    pred = pred[pred.game_id == GID].copy()
    pred["play_id"] = pred.play_id.astype(str)
    print(f"[audio] {len(pred)} shots in this game "
          f"(makes={int(pred.label.sum())} misses={int((1-pred.label).sum())} "
          f"FPs={int(((pred.label==0)&(pred.pred==1)).sum())} "
          f"TPs={int(((pred.label==1)&(pred.pred==1)).sum())})")

    rows = []
    n_skip = 0
    for _, r in pred.iterrows():
        rec = {"play_id": r.play_id, "label": int(r.label), "pred": int(r.pred),
               "prob": float(r.prob)}
        t = near_event_time(r.play_id)
        if t is None:
            n_skip += 1
            rec.update({k: np.nan for k in FEATS}); rows.append(rec); continue
        s_idx, e_idx = int((t - WIN_BEFORE) * sr), int((t + WIN_AFTER) * sr)
        if s_idx < 0 or e_idx >= len(x):
            n_skip += 1
            rec.update({k: np.nan for k in FEATS}); rows.append(rec); continue
        rec.update(features(x[s_idx:e_idx], sr)); rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "audio_features_b3c1f62c.csv", index=False)
    print(f"[audio] features for {len(df)-n_skip}/{len(df)} shots (skipped {n_skip})\n")

    from sklearn.metrics import roc_auc_score
    sub = df.dropna(subset=FEATS)
    print(f"=== MAKE vs MISS separation ({len(sub)} shots, single-feat AUC) ===")
    print(f"{'feature':<18}{'MAKE mean':>11}{'MISS mean':>11}{'AUC':>7}")
    for k in FEATS:
        try:
            auc = roc_auc_score(sub.label, sub[k])
        except Exception:
            auc = float("nan")
        auc = max(auc, 1 - auc)  # report orientation-invariant AUC
        print(f"  {k:<18}{sub[sub.label==1][k].mean():>11.4f}"
              f"{sub[sub.label==0][k].mean():>11.4f}{auc:>7.3f}")

    fps = sub[(sub.label==0) & (sub.pred==1)]
    tps = sub[(sub.label==1) & (sub.pred==1)]
    print(f"\n=== FP (depth-illusion calls MAKE) vs TP (true MAKE) ===")
    print(f"  FP n={len(fps)}   TP n={len(tps)}")
    print(f"{'feature':<18}{'FP mean':>11}{'TP mean':>11}{'gap':>9}")
    for k in FEATS:
        a, b = fps[k].dropna(), tps[k].dropna()
        if len(a)==0 or len(b)==0: continue
        print(f"  {k:<18}{a.mean():>11.4f}{b.mean():>11.4f}{a.mean()-b.mean():>+9.4f}")

    if len(fps) >= 3:
        print(f"\n=== quiet-audio VETO would-be impact on the {len(fps)} FPs ===")
        for pct in (10, 20, 30):
            thr = np.percentile(sub["onset_peak"].dropna(), pct)
            flips_fp = int((fps["onset_peak"] < thr).sum())
            kills_tp = int((tps["onset_peak"] < thr).sum())
            print(f"  onset_peak < p{pct}={thr:.3f}: flips {flips_fp}/{len(fps)} FPs, "
                  f"accidentally kills {kills_tp}/{len(tps)} TPs")


if __name__ == "__main__":
    main()
