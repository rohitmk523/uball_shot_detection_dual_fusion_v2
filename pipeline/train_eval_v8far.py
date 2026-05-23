#!/usr/bin/env python3
"""Retrain the iter8 model on v8far and evaluate head-to-head vs iter8 (v8).

Reproduces the iter8 production pipeline EXACTLY (verified to reproduce the
committed v8 test predictions: acc=0.9052 prec=0.8645 rec=0.9158 FP=29 FN=17):
  * features  = all numeric columns minus
                {game_id,play_id,classification,label,v1_in_scope,split}
  * model     = Pipeline(median SimpleImputer ->
                HistGradientBoosting(max_depth=4, min_samples_leaf=40,
                learning_rate=0.08, max_iter=500, random_state=42))
  * threshold = tuned on val to MINIMIZE
                max(0.90-precision,0)+max(0.90-recall,0)
                over thr in [0.30, 0.85) step 0.01.
  * test opened ONCE.

Deterministic seed 42. Local CPU only. Never overwrites v8 artifacts.
Writes data/p3_model_v8far.joblib + data/p3_test_predictions_v8far.parquet.
Also runs GroupKFold(5, by game) on train+val and full 18-game LOGO.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
RNG = 42
DROP = {"game_id", "play_id", "classification", "label",
        "v1_in_scope", "split"}

V8FAR = REPO_ROOT / "data" / "p2_features_v8far.parquet"
V8 = REPO_ROOT / "data" / "p2_features_v8.parquet"
V8_PRED = REPO_ROOT / "data" / "p3_test_predictions_v8.parquet"
MODEL_OUT = REPO_ROOT / "data" / "p3_model_v8far.joblib"
PRED_OUT = REPO_ROOT / "data" / "p3_test_predictions_v8far.parquet"

HGB_PARAMS = dict(max_depth=4, min_samples_leaf=40, learning_rate=0.08,
                  max_iter=500, random_state=RNG)


def make_pipe() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(**HGB_PARAMS)),
    ])


def feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]


def balanced_threshold(pv: np.ndarray, yv: np.ndarray) -> float:
    """Minimize max(0.90-precision,0)+max(0.90-recall,0) on val."""
    best_thr, best_loss = 0.5, float("inf")
    for thr in np.arange(0.30, 0.85, 0.01):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(yv, pred, zero_division=0)
        r = recall_score(yv, pred, zero_division=0)
        loss = max(0.90 - p, 0.0) + max(0.90 - r, 0.0)
        if loss < best_loss - 1e-12:
            best_loss, best_thr = loss, float(thr)
    return best_thr


def metrics(y: np.ndarray, yp: np.ndarray, pr: np.ndarray) -> dict:
    cm = confusion_matrix(y, yp, labels=[0, 1])
    return {
        "acc": accuracy_score(y, yp),
        "prec": precision_score(y, yp, zero_division=0),
        "rec": recall_score(y, yp, zero_division=0),
        "f1": f1_score(y, yp, zero_division=0),
        "auc": roc_auc_score(y, pr),
        "cm": cm.tolist(),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
    }


def train_eval(df: pd.DataFrame) -> Tuple[Pipeline, float, np.ndarray,
                                          np.ndarray, dict]:
    cols = feat_cols(df)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    Xtr, ytr = tr[cols].astype(float).values, tr.label.values
    Xva, yva = va[cols].astype(float).values, va.label.values
    Xte, yte = te[cols].astype(float).values, te.label.values
    pipe = make_pipe()
    pipe.fit(Xtr, ytr)
    pv = pipe.predict_proba(Xva)[:, 1]
    thr = balanced_threshold(pv, yva)
    pt = pipe.predict_proba(Xte)[:, 1]
    yp = (pt >= thr).astype(int)
    return pipe, thr, pt, yp, metrics(yte, yp, pt)


def groupkfold(df: pd.DataFrame) -> dict:
    cols = feat_cols(df)
    tv = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    X = tv[cols].astype(float).values
    y = tv.label.values
    grp = tv.game_id.values
    gkf = GroupKFold(n_splits=5)
    accs, aucs = [], []
    for tri, vai in gkf.split(X, y, grp):
        p = make_pipe()
        p.fit(X[tri], y[tri])
        prob = p.predict_proba(X[vai])[:, 1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(y[vai], pred))
        aucs.append(roc_auc_score(y[vai], prob))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def logo(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-game-out: hold each game out as test, tune balanced thr on
    an inner val (one held-out train game), train on the rest."""
    cols = feat_cols(df)
    games = sorted(df.game_id.unique())
    rng = np.random.RandomState(RNG)
    rows = []
    for held in games:
        test = df[df.game_id == held]
        rest = df[df.game_id != held]
        rest_games = sorted(rest.game_id.unique())
        # Deterministic inner-val pick: the rest-game whose index hashes
        # consistently (use RNG seeded by held game for reproducibility).
        val_game = rest_games[rng.randint(len(rest_games))]
        va = rest[rest.game_id == val_game]
        trn = rest[rest.game_id != val_game]
        Xtr, ytr = trn[cols].astype(float).values, trn.label.values
        Xva, yva = va[cols].astype(float).values, va.label.values
        Xte, yte = test[cols].astype(float).values, test.label.values
        p = make_pipe()
        p.fit(Xtr, ytr)
        pv = p.predict_proba(Xva)[:, 1]
        thr = balanced_threshold(pv, yva)
        pt = p.predict_proba(Xte)[:, 1]
        yp = (pt >= thr).astype(int)
        m = metrics(yte, yp, pt)
        rows.append({"game": held, "n": len(test), "thr": thr,
                     "acc": m["acc"], "prec": m["prec"], "rec": m["rec"]})
    return pd.DataFrame(rows)


def per_game(df_test: pd.DataFrame, yp: np.ndarray) -> None:
    te = df_test.reset_index(drop=True)
    for gid, g in te.groupby("game_id"):
        idx = g.index.values
        cm = confusion_matrix(g.label.values, yp[idx], labels=[0, 1])
        print(f"  {gid[:8]} n={len(g):3d} acc={accuracy_score(g.label,yp[idx]):.3f} "
              f"prec={precision_score(g.label,yp[idx],zero_division=0):.3f} "
              f"rec={recall_score(g.label,yp[idx],zero_division=0):.3f} "
              f"FP={cm[0,1]} FN={cm[1,0]}")


def main() -> None:
    vf = pd.read_parquet(V8FAR)
    v8 = pd.read_parquet(V8)

    pipe, thr, pt, yp, m = train_eval(vf)
    joblib.dump(pipe, MODEL_OUT)

    te = vf[vf.split == "test"].reset_index(drop=True)
    out = te[["game_id", "play_id", "classification", "label",
              "v1_in_scope"]].copy()
    out["prob"] = pt
    out["pred"] = yp
    out["correct"] = (yp == te.label.values)
    out.to_parquet(PRED_OUT, index=False)

    # ---- iter8 baseline (committed predictions) ----
    base = pd.read_parquet(V8_PRED)
    bm = metrics(base.label.values, base.pred.values, base.prob.values)

    print("=" * 72)
    print("HEAD-TO-HEAD: iter8 (v8)  vs  v8far  — held-out TEST (opened once)")
    print("=" * 72)
    print(f"  iter8  acc={bm['acc']:.4f} prec={bm['prec']:.4f} "
          f"rec={bm['rec']:.4f} f1={bm['f1']:.4f} auc={bm['auc']:.4f} "
          f"FP={bm['FP']} FN={bm['FN']} cm={bm['cm']}")
    print(f"  v8far  acc={m['acc']:.4f} prec={m['prec']:.4f} "
          f"rec={m['rec']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f} "
          f"FP={m['FP']} FN={m['FN']} cm={m['cm']}  (thr={thr:.2f})")

    print("\n--- per-test-game: iter8 ---")
    per_game(base, base.pred.values)
    print("--- per-test-game: v8far ---")
    per_game(te, yp)

    # ---- error-class diagnostics on TEST ----
    def depth_fp(pred_df, feat_df):
        f = feat_df.set_index(["game_id", "play_id"])
        n = 0
        for _, r in pred_df.iterrows():
            if r.label == 0 and r.pred == 1:  # false positive
                row = f.loc[(r.game_id, r.play_id)]
                if (row.g_far_dmin > 2.0 and row.g_far_bdet > 0.6
                        and row.g_near_inside > 0.05):
                    n += 1
        return n

    def swish_fn(pred_df, feat_df):
        f = feat_df.set_index(["game_id", "play_id"])
        n = 0
        for _, r in pred_df.iterrows():
            if r.label == 1 and r.pred == 0:  # false negative
                row = f.loc[(r.game_id, r.play_id)]
                if (row.frac_inside_rim_max < 0.15
                        and row.through_hoop_conf_max > 0.5):
                    n += 1
        return n

    base_pred = base.copy()
    vf_pred = out.copy()
    print("\n--- DEPTH-ILLUSION FP class "
          "(FP & g_far_dmin>2 & g_far_bdet>0.6 & g_near_inside>0.05) ---")
    print(f"  iter8(v8)={depth_fp(base_pred, v8)}   "
          f"v8far={depth_fp(vf_pred, vf)}")
    print("--- SWISH FN class "
          "(FN & frac_inside_rim_max<0.15 & through_hoop_conf_max>0.5) ---")
    print(f"  iter8(v8)={swish_fn(base_pred, v8)}   "
          f"v8far={swish_fn(vf_pred, vf)}")

    # ---- GroupKFold(5) stability ----
    print("\n--- GroupKFold(5, by game) on train+val ---")
    gk8 = groupkfold(v8)
    gkf = groupkfold(vf)
    print(f"  iter8(v8) acc={gk8['acc_mean']:.3f}+/-{gk8['acc_std']:.3f} "
          f"auc={gk8['auc_mean']:.3f}+/-{gk8['auc_std']:.3f}")
    print(f"  v8far     acc={gkf['acc_mean']:.3f}+/-{gkf['acc_std']:.3f} "
          f"auc={gkf['auc_mean']:.3f}+/-{gkf['auc_std']:.3f}")

    # ---- Full 18-game LOGO (same procedure on v8 and v8far) ----
    def logo_summary(df):
        lg = logo(df)
        w = np.array(lg.n) / np.array(lg.n).sum()
        wacc = float((lg.acc * w).sum())
        wprec = float((lg.prec * w).sum())
        wrec = float((lg.rec * w).sum())
        n_all = int(((lg.acc >= 0.92) & (lg.prec >= 0.90)
                     & (lg.rec >= 0.90)).sum())
        return lg, wacc, wprec, wrec, n_all

    print("\n--- Full 18-game LOGO (balanced thr / inner val) ---")
    lg8, a8, p8, r8, t8 = logo_summary(v8)
    lgf, af, pf, rf, tf = logo_summary(vf)
    print("  per-game (acc / prec / rec): iter8(v8)  ||  v8far")
    m8 = lg8.set_index("game")
    mf = lgf.set_index("game")
    for g in sorted(mf.index):
        r8r = m8.loc[g]
        rfr = mf.loc[g]
        hit8 = (r8r.acc >= 0.92 and r8r.prec >= 0.90 and r8r.rec >= 0.90)
        hitf = (rfr.acc >= 0.92 and rfr.prec >= 0.90 and rfr.rec >= 0.90)
        print(f"  {g[:8]} n={int(rfr.n):3d}  "
              f"{r8r.acc:.3f}/{r8r.prec:.3f}/{r8r.rec:.3f}{'*' if hit8 else ' '}  ||  "
              f"{rfr.acc:.3f}/{rfr.prec:.3f}/{rfr.rec:.3f}{'*' if hitf else ' '}")
    print(f"\n  LOGO weighted  iter8(v8): acc={a8:.3f} prec={p8:.3f} "
          f"rec={r8:.3f}  games-at-target={t8}/18")
    print(f"  LOGO weighted  v8far    : acc={af:.3f} prec={pf:.3f} "
          f"rec={rf:.3f}  games-at-target={tf}/18")
    print(f"  (task reference for iter8 LOGO: 0.883 weighted, 2 at target)")

    print(f"\n[v8far] wrote {MODEL_OUT.name} and {PRED_OUT.name}")


if __name__ == "__main__":
    main()
