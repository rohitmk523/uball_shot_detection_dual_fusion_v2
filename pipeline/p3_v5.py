"""
P3 iter5: v4 features + explicit net-motion x box-trajectory interactions +
HGB hyperparameter search (val-tuned, test opened once).

Local CPU only. Deterministic seed 42. Whole-game splits, anchor held out.
Immutable: never overwrite v4 artifacts.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
RNG = 42


def add_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Net-motion x box-trajectory interactions, lossless additions."""
    new = pd.DataFrame(index=df.index)
    # Helpful safe getters
    def col(name, default=0.0):
        return df[name].astype(float) if name in df.columns else pd.Series(default, index=df.index)

    nm_peak = col("nm_fd_peak_max")            # max net-energy across angles
    nm_peak_qw = col("nm_fd_peak_qw")          # quality-weighted
    nm_flow_peak = col("nm_flow_peak_max")
    nm_tail = col("nm_fd_tail_max_max")
    nm_agree = col("nm_agree_high")

    # box-trajectory ceiling-relevant signals (best features from iter1-3)
    frac_in = col("frac_inside_rim_max")
    thru_conf = col("through_hoop_conf_max")
    arc_entry = col("arc_entry_angle_qw")
    min_dist = col("min_dist_rw_near_best")

    # 1. geometry x motion: ball-in-rim AND net moved -> make-like
    new["ix_frac_x_nmpeak"]    = frac_in * nm_peak
    new["ix_thru_x_nmpeak"]    = thru_conf * nm_peak
    new["ix_arc_x_nmpeak_qw"]  = arc_entry * nm_peak_qw
    # 2. rim-out signature: HIGH motion energy but LOW rim-pass = bounce-out
    #    (motion came from chaotic rim-out trajectory, not a clean pass)
    new["ix_nmpeak_div_frac"]  = nm_peak / (frac_in + 0.01)
    new["ix_nmpeak_div_thru"]  = nm_peak / (thru_conf + 0.01)
    # 3. close approach + net moved -> make-like; close + no net move = airball/clean miss
    new["ix_close_x_nmpeak"]   = (1.0 / (min_dist + 0.5)) * nm_peak
    # 4. multi-angle agreement weighted by total motion (consistent make/miss)
    new["ix_agree_x_nmpeak"]   = nm_agree * nm_peak
    new["ix_agree_x_thru"]     = nm_agree * thru_conf
    # 5. tail-heavy vs peak-heavy: late motion = ball rattling around rim
    new["ix_tail_over_peak"]   = nm_tail / (nm_peak + 0.01)
    new["ix_flow_over_fd"]     = nm_flow_peak / (nm_peak + 0.01)
    out = pd.concat([df, new], axis=1)
    return out, list(new.columns)


def fit_test(X_tr, y_tr, X_va, y_va, X_te, y_te, params, prec_floor=0.85):
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(
                      random_state=RNG, **params))])
    p.fit(X_tr, y_tr)
    pv = p.predict_proba(X_va)[:, 1]
    best = {"thr": 0.5, "score": -1, "prec": 0, "rec": 0}
    for thr in np.arange(0.30, 0.85, 0.01):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0: continue
        prec = precision_score(y_va, pred, zero_division=0)
        rec  = recall_score(y_va, pred, zero_division=0)
        score = prec if prec < prec_floor else f1_score(y_va, pred, zero_division=0)
        if score > best["score"]:
            best = {"thr": float(thr), "score": float(score),
                    "prec": float(prec), "rec": float(rec)}
    thr = best["thr"]
    pt = p.predict_proba(X_te)[:, 1]
    yp = (pt >= thr).astype(int)
    return p, thr, best, pt, yp, {
        "acc": accuracy_score(y_te, yp),
        "prec": precision_score(y_te, yp, zero_division=0),
        "rec":  recall_score(y_te, yp, zero_division=0),
        "f1":   f1_score(y_te, yp, zero_division=0),
        "auc":  roc_auc_score(y_te, pt),
        "cm":   confusion_matrix(y_te, yp).tolist(),
    }


def main():
    v4 = pq.read_table(ROOT / "data" / "p2_features_v4.parquet").to_pandas()
    v5, ix_cols = add_interactions(v4)
    print(f"v4: {len(v4.columns)} cols -> v5: {len(v5.columns)} cols (+{len(ix_cols)} interactions)")
    v5.to_parquet(ROOT / "data" / "p2_features_v5.parquet", index=False)

    tr = v5[v5.split=="train"].reset_index(drop=True)
    va = v5[v5.split=="val"].reset_index(drop=True)
    te = v5[v5.split=="test"].reset_index(drop=True)

    drop = {"game_id","play_id","classification","label","v1_in_scope","split"}
    cols = [c for c in v5.columns if c not in drop and pd.api.types.is_numeric_dtype(v5[c])]

    Xtr = tr[cols].astype(float).values; ytr = tr.label.values
    Xva = va[cols].astype(float).values; yva = va.label.values
    Xte = te[cols].astype(float).values; yte = te.label.values

    # Hyperparameter grid (small, val-tuned)
    grid = list(itertools.product(
        [3, 4, 5],          # max_depth
        [10, 20, 40],       # min_samples_leaf
        [0.03, 0.05, 0.08], # learning_rate
        [300, 500],         # max_iter
    ))
    # Score by val F1 with precision floor; SELECT model using VAL only.
    best_val_score = -1; best_params = None
    for d, m, lr, n in grid:
        params = dict(max_depth=d, min_samples_leaf=m,
                      learning_rate=lr, max_iter=n)
        p = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", HistGradientBoostingClassifier(
                          random_state=RNG, **params))])
        p.fit(Xtr, ytr)
        pv = p.predict_proba(Xva)[:, 1]
        # pick best val score with precision floor
        s = -1
        for thr in np.arange(0.30, 0.85, 0.01):
            pred = (pv >= thr).astype(int)
            if pred.sum() == 0: continue
            prec = precision_score(yva, pred, zero_division=0)
            sc = prec if prec < 0.85 else f1_score(yva, pred, zero_division=0)
            if sc > s: s = sc
        if s > best_val_score:
            best_val_score = s; best_params = params
    print(f"best HGB params (val-selected): {best_params}  val_score={best_val_score:.4f}")

    # Fit final ONCE with best params, evaluate on TEST ONCE
    pipe, thr, best, pt, yp, m = fit_test(Xtr, ytr, Xva, yva, Xte, yte, best_params)
    print(f"\n=== iter5 held-out TEST (v5: v4 + {len(ix_cols)} interactions + tuned HGB) ===")
    print(f"  acc={m['acc']:.3f} prec={m['prec']:.3f} rec={m['rec']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} thr={thr:.2f} cm={m['cm']}")

    # Slices
    sc = te.v1_in_scope.astype(bool).values
    print(f"  v1_in_scope n={sc.sum()} acc={accuracy_score(yte[sc],yp[sc]):.3f} "
          f"prec={precision_score(yte[sc],yp[sc],zero_division=0):.3f} "
          f"rec={recall_score(yte[sc],yp[sc],zero_division=0):.3f}")
    for gid, g in te.groupby("game_id"):
        idx = g.index.values
        print(f"  game {gid[:8]} n={len(g)} acc={accuracy_score(yte[idx],yp[idx]):.3f} "
              f"prec={precision_score(yte[idx],yp[idx],zero_division=0):.3f} "
              f"rec={recall_score(yte[idx],yp[idx],zero_division=0):.3f}")

    # Ablation
    print("\n=== Ablation (each refit + val-tuned threshold, test opened once) ===")
    for tag, cset in [("v4 only (no ix)", [c for c in cols if c not in ix_cols]),
                      ("v5 (v4 + ix)",   cols)]:
        Xtr2 = tr[cset].astype(float).values; Xva2 = va[cset].astype(float).values; Xte2 = te[cset].astype(float).values
        _,_,_,_,_, ma = fit_test(Xtr2, ytr, Xva2, yva, Xte2, yte, best_params)
        print(f"  {tag:22s} acc={ma['acc']:.3f} prec={ma['prec']:.3f} rec={ma['rec']:.3f} cm={ma['cm']}")

    # GroupKFold stability on train+val
    tv = pd.concat([tr, va], ignore_index=True)
    Xtv = tv[cols].astype(float).values; ytv = tv.label.values; grp = tv.game_id.values
    gkf = GroupKFold(n_splits=5); accs, aucs = [], []
    for tri, vai in gkf.split(Xtv, ytv, grp):
        p = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", HistGradientBoostingClassifier(random_state=RNG, **best_params))])
        p.fit(Xtv[tri], ytv[tri])
        prob = p.predict_proba(Xtv[vai])[:,1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(ytv[vai], pred))
        aucs.append(roc_auc_score(ytv[vai], prob))
    print(f"\nGroupKFold (train+val by game, 5 folds): acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}  auc {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # Persist v5 predictions
    out = te[["game_id","play_id","classification","label","v1_in_scope"]].copy()
    out["prob"]=pt; out["pred"]=yp; out["correct"]=(yp==yte)
    out.to_parquet(ROOT / "data" / "p3_test_predictions_v5.parquet", index=False)

    # Final table
    print("\n" + "="*68)
    print("FINAL TABLE (held-out TEST, anchor c2a354fe held out)")
    print("="*68)
    print(f"  v1 baseline    0.857 / 0.795 / 0.892")
    print(f"  iter1          0.847 / 0.774 / 0.896")
    print(f"  iter2          0.874 / 0.837 / 0.866")
    print(f"  iter3          0.878 / 0.852 / 0.856")
    print(f"  iter4 (v4)     0.893 / 0.854 / 0.896")
    print(f"  iter5 (v5+tune){m['acc']:7.3f} / {m['prec']:.3f} / {m['rec']:.3f}")
    print(f"  target         >=0.92 / >=0.90 / >=0.90")


if __name__ == "__main__":
    main()
