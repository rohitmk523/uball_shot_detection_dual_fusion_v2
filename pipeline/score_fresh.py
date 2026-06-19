#!/usr/bin/env python3
"""Out-of-sample validation of the PRODUCTION angle-aware model on fresh games.

These 5 games (manifest split=="fresh") were recorded 2026-05-16/05-19 and were
NEVER part of training/val/test. We replicate the EXACT production feature
pipeline for them, then score with the committed p3_model_angleaware.joblib —
NO retraining, NO threshold re-tuning on fresh data. This measures the true
2D-software ceiling on unseen footage.

Feature lineage (identical code paths to production):
  box    -> p2_dataset._shot_rows          (v3 box-trajectory, far_v16 tracks)
  geo    -> geometry_features.build_geometry (g_* features)
  nm     -> p2p3_v4.nm_feats_for_game        (net-motion aggregation)
  fuse   -> angle-aware transform (FARSIDE/NEARSIDE per plays.angle)
  score  -> p3_model_angleaware.joblib + production threshold

Threshold is recovered deterministically from the production train/val split
(seed 42) and cross-checked against the committed test predictions, so fresh
shots are judged by the SAME decision boundary the client model ships with.

Run AFTER both GPU passes (tracks + netmotion) have populated
s3://uball-cv-results/cv-results/dual-fusion-v2-fresh/. Local CPU only.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from p2_dataset import _shot_rows                 # noqa: E402
from geometry_features import build_geometry      # noqa: E402
from p2p3_v4 import nm_feats_for_game             # noqa: E402
import p3_angleaware as P3                        # noqa: E402

FRESH_WORK = "s3://uball-cv-results/cv-results/dual-fusion-v2-fresh"
GT_FRESH_S3 = "s3://uball-cv-results/cv-results/dual-fusion-v2/gt_fresh/gt_windows.json"
REGION = "us-east-1"

TRACKS_LOCAL = Path("/tmp/p1tracks_fresh")
NM_LOCAL = Path("/tmp/p1netmotion_fresh")
GT_LOCAL = Path("/tmp/gt_fresh.json")

V8FAR_TRAIN = ROOT / "data" / "p2_features_v8far.parquet"
AA_TRAIN = ROOT / "data" / "p2_features_v8far_angleaware.parquet"
MODEL = ROOT / "data" / "p3_model_angleaware.joblib"
TEST_PRED = ROOT / "data" / "p3_test_predictions_angleaware.parquet"
OUT_PRED = ROOT / "data" / "p3_fresh_predictions.parquet"

SUFFIX = re.compile(r"_(FL|FR|NL|NR)$")


def _sh(cmd: list) -> None:
    subprocess.run(cmd, check=True)


def fresh_game_ids() -> list:
    m = json.loads((ROOT / "data" / "games_manifest.json").read_text())
    return [g["game_id"] for g in m["games"] if g.get("split") == "fresh"]


def pull_artifacts(gids: list) -> None:
    """Download fresh tracks + netmotion from S3 (idempotent)."""
    TRACKS_LOCAL.mkdir(parents=True, exist_ok=True)
    NM_LOCAL.mkdir(parents=True, exist_ok=True)
    if not GT_LOCAL.exists():
        _sh(["aws", "s3", "cp", GT_FRESH_S3, str(GT_LOCAL),
             "--region", REGION, "--quiet"])
    for gid in gids:
        tdst = TRACKS_LOCAL / f"{gid}.parquet"
        if not tdst.exists():
            _sh(["aws", "s3", "cp",
                 f"{FRESH_WORK}/tracks/{gid}/tracks.parquet", str(tdst),
                 "--region", REGION, "--quiet"])
        ndst = NM_LOCAL / gid / "netmotion.parquet"
        ndst.parent.mkdir(parents=True, exist_ok=True)
        if not ndst.exists():
            _sh(["aws", "s3", "cp",
                 f"{FRESH_WORK}/netmotion/{gid}/netmotion.parquet", str(ndst),
                 "--region", REGION, "--quiet"])


def angle_map() -> dict:
    gt = json.loads(GT_LOCAL.read_text())["games"]
    return {s["play_id"]: s.get("angle", "LEFT")
            for sh in gt.values() for s in sh}


def build_fresh_v8far(gids: list) -> pd.DataFrame:
    """Replicate build_v8far for the fresh games (no v8 carry-over: nm + labels
    come from the fresh extraction/GT, not the original 18-game parquet)."""
    box_rows = []
    tracks = {}
    for gid in gids:
        df = pd.read_parquet(TRACKS_LOCAL / f"{gid}.parquet")
        tracks[gid] = df
        box_rows.extend(_shot_rows(gid, df))
    box = pd.DataFrame(box_rows)
    box["play_id"] = box["play_id"].astype(str)

    geo = build_geometry(tracks)
    geo["play_id"] = geo["play_id"].astype(str)

    nm_frames = []
    for gid in gids:
        f = nm_feats_for_game(NM_LOCAL / gid / "netmotion.parquet")
        f["game_id"] = gid
        nm_frames.append(f)
    nm = pd.concat(nm_frames, ignore_index=True)
    nm["play_id"] = nm["play_id"].astype(str)

    key = ["game_id", "play_id"]
    merged = (box
              .merge(nm, on=key, how="left")
              .merge(geo, on=key, how="left"))
    nm_cols = [c for c in nm.columns if c not in key]
    for c in nm_cols:
        merged[c] = merged[c].fillna(0.0)   # production fills missing nm with 0
    merged["split"] = "fresh"

    # Align feature columns to the TRAINING v8far schema so the downstream
    # angle-aware transform sees identical feature families. Missing columns
    # (none expected) become NaN and are imputed by the model's median imputer.
    train_cols = list(pd.read_parquet(V8FAR_TRAIN).columns)
    for c in train_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    extra = [c for c in merged.columns if c not in train_cols]
    return merged[train_cols + [c for c in ("split",) if c in extra]]


def apply_angle_aware(v: pd.DataFrame, amap: dict) -> pd.DataFrame:
    """Identical mapping to p3_angleaware.build_angle_aware."""
    side = v.play_id.map(amap).fillna("LEFT")
    is_left = (side == "LEFT").to_numpy()
    families: dict = {}
    for c in v.columns:
        m = SUFFIX.search(c)
        if m:
            families.setdefault(c[:m.start()], {})[m.group(1)] = c
    new_cols, drop_raw = {}, []
    for base, d in families.items():
        if all(k in d for k in ("FL", "FR", "NL", "NR")):
            fl, fr = v[d["FL"]].to_numpy(float), v[d["FR"]].to_numpy(float)
            nl, nr = v[d["NL"]].to_numpy(float), v[d["NR"]].to_numpy(float)
            new_cols[f"{base}_FARSIDE"] = np.where(is_left, fl, fr)
            new_cols[f"{base}_NEARSIDE"] = np.where(is_left, nl, nr)
            drop_raw += list(d.values())
    aa = pd.concat([v.drop(columns=drop_raw).reset_index(drop=True),
                    pd.DataFrame(new_cols)], axis=1)
    aa["side_is_left"] = is_left.astype(int)
    return aa


def production_threshold(cols: list) -> float:
    """Recover the exact production threshold by re-running the deterministic
    val tuning (seed 42), and assert it reproduces the committed test preds."""
    aa = pd.read_parquet(AA_TRAIN)
    tr, va, te = (aa[aa.split == s] for s in ("train", "val", "test"))
    pipe, thr = P3._fit_with_balanced_threshold(tr, va, cols)
    saved = pd.read_parquet(TEST_PRED)
    prob = pipe.predict_proba(te[cols].astype(float).values)[:, 1]
    # te order == saved order (both from the same test slice, same seed build)
    diff = float(np.abs(prob - saved["prob"].to_numpy()).max())
    print(f"[thr] recovered production threshold = {thr:.3f} "
          f"(refit vs committed test preds max|Δprob|={diff:.2e})")
    if diff > 1e-6:
        print("[thr] WARNING: refit probs drift from committed preds; "
              "falling back to threshold recovered from the predictions file.")
        lo = saved.loc[saved.pred == 0, "prob"].max()
        hi = saved.loc[saved.pred == 1, "prob"].min()
        thr = round(float((lo + hi) / 2), 4)
        print(f"[thr] fallback threshold = {thr:.4f}")
    return thr


def report(out: pd.DataFrame) -> None:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, confusion_matrix)
    y, p = out.label.to_numpy(), out.pred.to_numpy()
    print("\n================ FRESH OUT-OF-SAMPLE RESULTS ================")
    print(f"{'game':<14}{'shots':>6}{'makes':>6}{'acc':>8}{'prec':>8}"
          f"{'rec':>8}{'FP':>5}{'FN':>5}")
    for gid, g in out.groupby("game_id"):
        yg, pg = g.label.to_numpy(), g.pred.to_numpy()
        cm = confusion_matrix(yg, pg, labels=[0, 1])
        fp, fn = int(cm[0, 1]), int(cm[1, 0])
        print(f"{gid[:12]:<14}{len(g):>6}{int(yg.sum()):>6}"
              f"{accuracy_score(yg, pg):>8.3f}"
              f"{precision_score(yg, pg, zero_division=0):>8.3f}"
              f"{recall_score(yg, pg, zero_division=0):>8.3f}{fp:>5}{fn:>5}")
    cm = confusion_matrix(y, p, labels=[0, 1])
    print("-" * 60)
    print(f"{'OVERALL':<14}{len(out):>6}{int(y.sum()):>6}"
          f"{accuracy_score(y, p):>8.3f}"
          f"{precision_score(y, p, zero_division=0):>8.3f}"
          f"{recall_score(y, p, zero_division=0):>8.3f}"
          f"{int(cm[0,1]):>5}{int(cm[1,0]):>5}")
    print(f"confusion [[TN FP][FN TP]] = {cm.tolist()}")
    print("training test ceiling was: acc=0.955 prec=0.909 rec=0.990")


def main() -> None:
    gids = fresh_game_ids()
    print(f"[fresh] {len(gids)} games: {', '.join(g[:8] for g in gids)}")
    pull_artifacts(gids)

    v = build_fresh_v8far(gids)
    print(f"[fresh] v8far: {len(v)} shots x {v.shape[1]} cols")

    aa = apply_angle_aware(v, angle_map())
    cols = P3._feature_cols(pd.read_parquet(AA_TRAIN))
    for c in cols:
        if c not in aa.columns:
            aa[c] = np.nan
    print(f"[fresh] angle-aware: {len(aa)} shots, {len(cols)} model features")

    thr = production_threshold(cols)
    model = joblib.load(MODEL)
    prob = model.predict_proba(aa[cols].astype(float).values)[:, 1]
    pred = (prob >= thr).astype(int)

    out = aa[["game_id", "play_id", "classification", "label"]].copy()
    out["prob"] = prob
    out["pred"] = pred
    out["correct"] = out.pred == out.label
    out.to_parquet(OUT_PRED, index=False)
    report(out)
    print(f"\nsaved {OUT_PRED}")


if __name__ == "__main__":
    main()
