"""
P2v4 + P3v4 in one shot.

Adds per-shot net-motion features (per-angle + fused) to the v3 feature set,
re-fits HistGradientBoosting + LogReg with whole-game splits and a precision-
aware threshold on val, then evaluates ONCE on test. Reports the ablation
v3-only vs v3+net-motion, the previously-ambiguous-band recovery (iter3 prob
in 0.40-0.70), per-game test breakdown, and top features.

Local CPU only. Deterministic seed 42. Immutable outputs (writes _v4 files).
"""
from __future__ import annotations
import glob, json, os, hashlib
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
RNG = 42
NM_DIR = Path("/tmp/p1netmotion")
V3_PATH = ROOT / "data" / "p2_features_v3.parquet"
OUT_V4 = ROOT / "data" / "p2_features_v4.parquet"
RESULTS_MD = ROOT / "data" / "P3_RESULTS_v4.md"
PRED_OUT = ROOT / "data" / "p3_test_predictions_v4.parquet"
ITER3_PRED = ROOT / "data" / "p3_test_predictions_v3.parquet"

ANGLES = ["FL", "FR", "NL", "NR"]


def nm_feats_for_game(parquet: Path) -> pd.DataFrame:
    """Per-shot net-motion features for one game. Per-angle + fused."""
    df = pq.read_table(str(parquet)).to_pandas()
    rows = []
    for (pid, ang), g in df.groupby(["play_id", "angle"]):
        g = g.sort_values("frame_idx")
        n = len(g)
        if n == 0:
            continue
        fd = g["framediff_energy"].astype(float).values
        fl = g["flow_mag"].astype(float).values
        rok = g["rim_ok"].astype(bool).values if "rim_ok" in g else np.ones(n, bool)
        # tail = last 50% of window (post likely rim-approach)
        cut = max(1, int(n * 0.5))
        tail_fd = fd[cut:] if len(fd) > cut else fd
        tail_fl = fl[cut:] if len(fl) > cut else fl
        # peak position (0=start, 1=end)
        peak_pos = float(np.argmax(fd) / max(1, n - 1)) if n > 1 else 0.0
        # sustained: fraction of frames above per-window median
        med = float(np.median(fd)) if n else 0.0
        sustained = float((fd > med).mean()) if n else 0.0
        rim_ok_frac = float(rok.mean()) if n else 0.0
        rows.append({
            "play_id": pid, "angle": ang,
            "nm_fd_peak": float(fd.max()) if n else 0.0,
            "nm_flow_peak": float(fl.max()) if n else 0.0,
            "nm_fd_mean": float(fd.mean()) if n else 0.0,
            "nm_fd_tail_max": float(tail_fd.max()) if len(tail_fd) else 0.0,
            "nm_fd_tail_mean": float(tail_fd.mean()) if len(tail_fd) else 0.0,
            "nm_flow_tail_mean": float(tail_fl.mean()) if len(tail_fl) else 0.0,
            "nm_peak_pos": peak_pos,
            "nm_sustained_frac": sustained,
            "nm_rim_ok_frac": rim_ok_frac,
            "nm_n_frames": n,
        })
    pa = pd.DataFrame(rows)
    # Pivot per-angle features
    feat_cols = [c for c in pa.columns if c.startswith("nm_")]
    wide = pa.pivot_table(index="play_id", columns="angle", values=feat_cols)
    wide.columns = [f"{c}_{a}" for c, a in wide.columns]
    wide = wide.reset_index()
    # Fused features across the 4 angles
    fuse_inp = pa.pivot_table(index="play_id", columns="angle",
                              values=["nm_fd_peak", "nm_flow_peak",
                                      "nm_fd_tail_max", "nm_rim_ok_frac"])
    fused = {}
    for metric in ["nm_fd_peak", "nm_flow_peak", "nm_fd_tail_max"]:
        sub = fuse_inp[metric]
        fused[f"{metric}_max"] = sub.max(axis=1)
        fused[f"{metric}_mean"] = sub.mean(axis=1)
        # quality-weighted (weight = rim_ok_frac per angle)
        q = fuse_inp["nm_rim_ok_frac"].fillna(0.0)
        qw_num = (sub.fillna(0.0) * q).sum(axis=1)
        qw_den = q.sum(axis=1).replace(0, np.nan)
        fused[f"{metric}_qw"] = (qw_num / qw_den).fillna(0.0)
    # agreement: # angles with fd_peak above each shot's own median across angles
    fdp = fuse_inp["nm_fd_peak"]
    med = fdp.median(axis=1)
    fused["nm_agree_high"] = (fdp.sub(med, axis=0) > 0).sum(axis=1)
    fused_df = pd.DataFrame(fused).reset_index()
    out = wide.merge(fused_df, on="play_id", how="outer")
    return out


def build_v4() -> pd.DataFrame:
    print("[v4] reading v3 features ...")
    v3 = pq.read_table(str(V3_PATH)).to_pandas()
    print(f"  v3: {len(v3)} shots x {len(v3.columns)} cols")
    nm_frames = []
    for p in sorted(glob.glob(str(NM_DIR / "*/netmotion.parquet"))):
        gid = Path(p).parent.name
        f = nm_feats_for_game(Path(p))
        f["game_id"] = gid
        nm_frames.append(f)
    nm = pd.concat(nm_frames, ignore_index=True)
    print(f"  netmotion: {len(nm)} shots, {len(nm.columns)} cols")
    v4 = v3.merge(nm, on=["game_id", "play_id"], how="left")
    nm_cols = [c for c in nm.columns if c not in ("game_id", "play_id")]
    # Fill missing net-motion (shouldn't happen — all 20 games extracted)
    for c in nm_cols:
        v4[c] = v4[c].fillna(0.0)
    print(f"  v4: {len(v4)} shots x {len(v4.columns)} cols  (+{len(nm_cols)} nm)")
    return v4, nm_cols


def feature_matrix(df: pd.DataFrame, cols: List[str]):
    return df[cols].astype(float).values


def fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
             precision_floor: float = 0.85) -> dict:
    """HGB with precision-aware threshold tuned on val."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05,
            min_samples_leaf=20, random_state=RNG)),
    ])
    pipe.fit(X_tr, y_tr)
    pv = pipe.predict_proba(X_va)[:, 1]
    # Precision-aware threshold: highest threshold that keeps recall reasonable
    # AND precision >= floor on val.
    best = {"thr": 0.5, "score": -1, "prec": 0.0, "rec": 0.0}
    for thr in np.arange(0.30, 0.85, 0.01):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0: continue
        p = precision_score(y_va, pred, zero_division=0)
        r = recall_score(y_va, pred, zero_division=0)
        if p < precision_floor:
            score = p  # below floor: maximize precision
        else:
            score = f1_score(y_va, pred, zero_division=0)
        if score > best["score"]:
            best = {"thr": float(thr), "score": float(score),
                    "prec": float(p), "rec": float(r)}
    thr = best["thr"]
    pt = pipe.predict_proba(X_te)[:, 1]
    yp = (pt >= thr).astype(int)
    return {
        "pipe": pipe, "thr": thr,
        "val_thr_pick": best,
        "test": {
            "acc": float(accuracy_score(y_te, yp)),
            "prec": float(precision_score(y_te, yp, zero_division=0)),
            "rec": float(recall_score(y_te, yp, zero_division=0)),
            "f1": float(f1_score(y_te, yp, zero_division=0)),
            "auc": float(roc_auc_score(y_te, pt)),
            "cm": confusion_matrix(y_te, yp).tolist(),
            "probs": pt, "preds": yp,
        }
    }


def main():
    np.random.seed(RNG)
    v4, nm_cols = build_v4()
    v4.to_parquet(OUT_V4, index=False)
    md5 = hashlib.md5(OUT_V4.read_bytes()).hexdigest()
    print(f"[v4] wrote {OUT_V4} md5={md5[:12]}  ({OUT_V4.stat().st_size/1e6:.1f}MB)")

    # Splits
    tr = v4[v4["split"] == "train"].reset_index(drop=True)
    va = v4[v4["split"] == "val"].reset_index(drop=True)
    te = v4[v4["split"] == "test"].reset_index(drop=True)
    print(f"\nsplits  train={len(tr)} val={len(va)} test={len(te)}")
    print(f"test games: {sorted(te.game_id.unique())}")

    # Feature columns: numeric only, drop ids/labels
    drop = {"game_id", "play_id", "classification", "label", "v1_in_scope", "split"}
    all_num = [c for c in v4.columns if c not in drop
               and pd.api.types.is_numeric_dtype(v4[c])]
    v3_cols = [c for c in all_num if not c.startswith("nm_")]
    v4_cols = all_num
    print(f"v3 features: {len(v3_cols)}   v4 (v3+nm): {len(v4_cols)}   nm cols: {len(nm_cols)}")

    y = lambda d: d["label"].astype(int).values

    # ---- Ablation: v3-only re-fit, then v3+nm ----
    rows = []
    for tag, cols in [("v3-only", v3_cols), ("v3+nm (v4)", v4_cols)]:
        Xtr = feature_matrix(tr, cols); Xva = feature_matrix(va, cols); Xte = feature_matrix(te, cols)
        r = fit_eval(Xtr, y(tr), Xva, y(va), Xte, y(te))
        t = r["test"]
        rows.append({"tag": tag, **{k: t[k] for k in ("acc","prec","rec","f1","auc")},
                     "TN_FP_FN_TP": t["cm"], "thr": r["thr"]})
        if tag == "v3+nm (v4)":
            final = r
    print("\n=== Ablation (held-out TEST, opened once per model) ===")
    for r in rows:
        print(f"  {r['tag']:14s}  acc={r['acc']:.3f} prec={r['prec']:.3f} rec={r['rec']:.3f}  thr={r['thr']:.2f}  cm={r['TN_FP_FN_TP']}")

    # ---- Slices on v4 ----
    pt = final["test"]["probs"]; yp = final["test"]["preds"]
    yt = y(te)
    print("\n=== v4 held-out TEST slices ===")
    # Overall already in ablation
    # v1_in_scope subset
    sc = te["v1_in_scope"].astype(bool).values
    yt_v1 = yt[sc]; yp_v1 = yp[sc]; pt_v1 = pt[sc]
    print(f"  v1_in_scope  n={sc.sum()}  acc={accuracy_score(yt_v1,yp_v1):.3f} "
          f"prec={precision_score(yt_v1,yp_v1,zero_division=0):.3f} "
          f"rec={recall_score(yt_v1,yp_v1,zero_division=0):.3f}")
    # Per-game (test)
    for gid, g in te.groupby("game_id"):
        idx = g.index.values
        ytg = yt[idx]; ypg = yp[idx]
        print(f"  game {gid[:8]} n={len(g)} acc={accuracy_score(ytg,ypg):.3f} "
              f"prec={precision_score(ytg,ypg,zero_division=0):.3f} "
              f"rec={recall_score(ytg,ypg,zero_division=0):.3f}")

    # ---- Previously-ambiguous-band recovery ----
    if ITER3_PRED.exists():
        prev = pq.read_table(str(ITER3_PRED)).to_pandas()
        keycol = "prob" if "prob" in prev.columns else ("p" if "p" in prev.columns else None)
        if keycol:
            amb = prev[(prev[keycol] >= 0.40) & (prev[keycol] <= 0.70)]
            iter3_correct = prev["correct"].astype(bool) if "correct" in prev else None
            te_index = te.set_index("play_id")
            v4_pred_map = dict(zip(te["play_id"], yp))
            v4_correct = {pid: (v4_pred_map.get(pid) == lab)
                          for pid, lab in zip(te["play_id"], yt)}
            amb_play_ids = amb["play_id"].tolist()
            in_test = [p for p in amb_play_ids if p in v4_pred_map]
            n_amb = len(in_test)
            now_correct = sum(1 for p in in_test if v4_correct.get(p, False))
            print(f"\n=== Previously-ambiguous band (iter3 prob 0.40-0.70) ===")
            print(f"  ambiguous shots in test: {n_amb}")
            print(f"  now correct (v4):        {now_correct}/{n_amb}  ({(now_correct/max(1,n_amb))*100:.1f}%)")

    # ---- GroupKFold stability (train+val) ----
    g_tv = pd.concat([tr, va], ignore_index=True)
    X_tv = feature_matrix(g_tv, v4_cols); y_tv = y(g_tv); grp = g_tv["game_id"].values
    gkf = GroupKFold(n_splits=5)
    accs, aucs = [], []
    for tr_i, va_i in gkf.split(X_tv, y_tv, grp):
        p = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", HistGradientBoostingClassifier(
                        max_depth=3, max_iter=300, learning_rate=0.05,
                        min_samples_leaf=20, random_state=RNG))])
        p.fit(X_tv[tr_i], y_tv[tr_i])
        prob = p.predict_proba(X_tv[va_i])[:, 1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(y_tv[va_i], pred))
        aucs.append(roc_auc_score(y_tv[va_i], prob))
    print(f"\n=== GroupKFold (train+val, by game, 5 folds) ===")
    print(f"  acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   auc {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # ---- Top features (HGB permutation-like via gain) ----
    clf = final["pipe"].named_steps["clf"]
    imps = getattr(clf, "feature_importances_", None)
    if imps is not None and len(imps) == len(v4_cols):
        top = pd.DataFrame({"feature": v4_cols, "importance": imps}).sort_values("importance", ascending=False).head(15)
        print("\n=== Top features (HGB feature_importances_) ===")
        for _, r in top.iterrows():
            mark = " *NM*" if r.feature.startswith("nm_") else ""
            print(f"  {r.importance:.4f}  {r.feature}{mark}")

    # Save predictions parquet for downstream audit
    out = te[["game_id", "play_id", "classification", "label", "v1_in_scope"]].copy()
    out["prob"] = pt
    out["pred"] = yp
    out["correct"] = (yp == yt)
    out.to_parquet(PRED_OUT, index=False)
    print(f"\n[v4] wrote {PRED_OUT} ({PRED_OUT.stat().st_size/1e3:.1f} KB)")

    # Final summary table
    print("\n" + "="*68)
    print("FINAL TABLE (held-out TEST, anchor c2a354fe always in test)")
    print("="*68)
    print(f"  v1 baseline       0.857 / 0.795 / 0.892")
    print(f"  iter1             0.847 / 0.774 / 0.896")
    print(f"  iter2             0.874 / 0.837 / 0.866")
    print(f"  iter3 (boxes)     0.878 / 0.852 / 0.856")
    rv4 = rows[-1]
    print(f"  iter4 (v3+nm)     {rv4['acc']:.3f} / {rv4['prec']:.3f} / {rv4['rec']:.3f}")
    print(f"  target            >=0.92 / >=0.90 / >=0.90")


if __name__ == "__main__":
    main()
