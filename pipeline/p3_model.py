#!/usr/bin/env python3
"""
P3 — interpretable make/miss model with whole-game, leakage-free eval.

- Train ONLY on `train` split games.
- Tune model choice + decision threshold on `val` split games.
- Report FINAL metrics on the held-out `test` split games (incl. the
  v1 anchor game c2a354fe). The test set is opened once.
- Whole-game splits: a game never spans two splits, so there is no
  cross-game leakage. A GroupKFold CV by game over train+val gives a
  stability estimate.

Models (all interpretable):
  - Logistic Regression (standardised) — read coefficients directly.
  - HistGradientBoosting (shallow) — permutation importances.

Deterministic (fixed seed). Immutable artifacts. Local CPU only.

Outputs:
  data/p3_model.joblib
  data/p3_test_predictions.parquet
  data/P3_RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import eprint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES = REPO_ROOT / "data" / "p2_features.parquet"
OUT_MODEL = REPO_ROOT / "data" / "p3_model.joblib"
OUT_PREDS = REPO_ROOT / "data" / "p3_test_predictions.parquet"
OUT_DOC = REPO_ROOT / "data" / "P3_RESULTS.md"

SEED = 42
ANCHOR = "c2a354fe-eb34-4980-af00-8f5ff6b00143"
META_COLS = ["game_id", "play_id", "classification", "label",
             "v1_in_scope", "split"]


def _load() -> Tuple[pd.DataFrame, List[str]]:
    if not FEATURES.exists():
        raise FileNotFoundError(
            f"{FEATURES} not found — run pipeline/p2_dataset.py first")
    df = pd.read_parquet(FEATURES)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    # Guard against accidental non-numeric / constant columns.
    feat_cols = [c for c in feat_cols
                 if pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols


class Winsorize(BaseEstimator, TransformerMixin):
    """Clip each feature to a fitted [low, high] percentile band. Tames
    physically-unbounded outliers (e.g. a ball that left the frame yields
    a huge 'rebound') so the standardised logistic regression stays
    numerically stable and its coefficients remain interpretable. Fitted
    on TRAIN only — no leakage."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.p, axis=0)
        self.hi_ = np.nanpercentile(X, 100.0 - self.p, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)


def _make_models() -> Dict[str, Pipeline]:
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorize(p=1.0)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear", penalty="l2", max_iter=5000,
            class_weight="balanced", C=0.5, random_state=SEED)),
    ])
    hgb = Pipeline([
        # HGB handles NaN natively but impute keeps the pipeline uniform.
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05,
            l2_regularization=1.0, random_state=SEED)),
    ])
    return {"logreg": logreg, "hgb": hgb}


def _metrics(y: np.ndarray, pred: np.ndarray,
             prob: np.ndarray) -> Dict[str, float]:
    out = {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)),
                           4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y, prob)), 4)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return out


def _best_threshold(y: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Pick the threshold maximising F1 on the validation set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, round(best_f1, 4)


def _group_cv(df: pd.DataFrame, feat_cols: List[str],
              model: Pipeline) -> Dict[str, float]:
    """GroupKFold by game over train+val (stability estimate, no leakage:
    whole games per fold)."""
    sub = df[df["split"].isin(["train", "val"])]
    X = sub[feat_cols].to_numpy()
    y = sub["label"].to_numpy()
    groups = sub["game_id"].to_numpy()
    n_games = len(np.unique(groups))
    k = min(5, n_games)
    gkf = GroupKFold(n_splits=k)
    accs, f1s, aucs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        m = _clone(model)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, zero_division=0))
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return {
        "folds": k,
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_std": round(float(np.std(accs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "auc_std": round(float(np.std(aucs)), 4) if aucs else None,
    }


def _clone(model: Pipeline) -> Pipeline:
    from sklearn.base import clone
    return clone(model)


def _coeffs(model: Pipeline, feat_cols: List[str],
            X: np.ndarray, y: np.ndarray, kind: str) -> List[Tuple]:
    """Interpretability: standardised logistic coefficients, or HGB
    permutation importances."""
    if kind == "logreg":
        clf = model.named_steps["clf"]
        coef = clf.coef_[0]
        pairs = sorted(zip(feat_cols, coef),
                       key=lambda p: -abs(p[1]))
        return [(f, round(float(c), 4)) for f, c in pairs]
    r = permutation_importance(
        model, X, y, n_repeats=10, random_state=SEED,
        scoring="f1")
    pairs = sorted(zip(feat_cols, r.importances_mean),
                   key=lambda p: -p[1])
    return [(f, round(float(v), 5)) for f, v in pairs]


def _subset_metrics(meta: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                    prob: np.ndarray) -> Dict[str, Dict]:
    """Test metrics overall, on v1_in_scope=True (apples-to-apples vs the
    v1 85.7/79.5/89.2 baseline), and on 4PT_MAKE-only shots."""
    out: Dict[str, Dict] = {}
    out["overall"] = _metrics(y, pred, prob)
    insc = meta["v1_in_scope"].to_numpy().astype(bool)
    out["v1_in_scope"] = _metrics(y[insc], pred[insc], prob[insc])
    is4 = (meta["classification"] == "4PT_MAKE").to_numpy()
    if is4.sum() > 0:
        out["4pt_make_only"] = {
            "n": int(is4.sum()),
            "recall_as_make": round(
                float((pred[is4] == 1).mean()), 4),
        }
    return out


def _by_game(meta: pd.DataFrame, y: np.ndarray,
             pred: np.ndarray, prob: np.ndarray) -> Dict[str, Dict]:
    res: Dict[str, Dict] = {}
    for gid in sorted(meta["game_id"].unique()):
        mask = (meta["game_id"] == gid).to_numpy()
        res[gid] = _metrics(y[mask], pred[mask], prob[mask])
    return res


def _fmt_metric_block(name: str, m: Dict) -> List[str]:
    cm = m.get("confusion_matrix", {})
    return [
        f"### {name}",
        f"- n={m.get('n')}  acc={m.get('accuracy')}  "
        f"prec={m.get('precision')}  rec={m.get('recall')}  "
        f"f1={m.get('f1')}  auc={m.get('roc_auc')}",
        f"- confusion: TN={cm.get('tn')} FP={cm.get('fp')} "
        f"FN={cm.get('fn')} TP={cm.get('tp')}",
        "",
    ]


def _write_doc(ctx: Dict) -> None:
    L: List[str] = ["# P3 Results — interpretable make/miss model\n"]
    L.append(f"Seed={SEED}. Whole-game splits (no game spans splits, no "
             f"leakage). Test opened once.\n")
    L.append("## Split sizes")
    for s, n in ctx["split_sizes"].items():
        L.append(f"- {s}: {n} shots")
    L.append("")
    L.append("## Model selection (tuned on VAL)")
    for name, v in ctx["val_selection"].items():
        L.append(f"- {name}: val_f1={v['val_f1']} "
                 f"val_acc={v['val_acc']} thr*={v['threshold']}")
    L.append(f"\n**Selected model: `{ctx['selected']}` "
             f"(threshold={ctx['threshold']})**\n")
    L.append("## GroupKFold CV (train+val, grouped by game)")
    cv = ctx["cv"]
    L.append(f"- folds={cv['folds']}  "
             f"acc={cv['acc_mean']}±{cv['acc_std']}  "
             f"f1={cv['f1_mean']}±{cv['f1_std']}  "
             f"auc={cv['auc_mean']}±{cv['auc_std']}\n")
    L.append("## HELD-OUT TEST METRICS\n")
    for key, label in (("overall", "Overall (make/miss, incl 4PT_MAKE)"),
                       ("v1_in_scope",
                        "v1_in_scope=True (apples-to-apples vs v1 "
                        "85.7/79.5/89.2)"),
                       ("4pt_make_only", "4PT_MAKE only")):
        if key in ctx["test"]:
            if key == "4pt_make_only":
                m = ctx["test"][key]
                L.append(f"### {label}")
                L.append(f"- n={m['n']}  recall_as_make="
                         f"{m['recall_as_make']}\n")
            else:
                L += _fmt_metric_block(label, ctx["test"][key])
    L.append("### v1 baseline comparison")
    b = ctx["test"]["v1_in_scope"]
    L.append("| metric | v1 baseline | v2 (v1_in_scope) | target |")
    L.append("|---|---|---|---|")
    L.append(f"| accuracy | 0.857 | {b['accuracy']} | >=0.92 |")
    L.append(f"| precision | 0.795 | {b['precision']} | >=0.90 |")
    L.append(f"| recall | 0.892 | {b['recall']} | >=0.90 |")
    L.append("")
    L.append("## Per-game test breakdown")
    for gid, m in ctx["by_game"].items():
        tag = "  <-- ANCHOR" if gid == ANCHOR else ""
        L.append(f"- `{gid}`{tag}: n={m['n']} acc={m['accuracy']} "
                 f"prec={m['precision']} rec={m['recall']} "
                 f"f1={m['f1']}")
    L.append("")
    L.append("## Interpretability — top 20 features")
    L.append(f"({'logistic coefficients (standardised)' if ctx['selected']=='logreg' else 'permutation importance (F1)'})\n")
    for f, v in ctx["importances"][:20]:
        L.append(f"- `{f}`: {v}")
    L.append("")
    if "logreg_coefs" in ctx:
        lc = ctx["logreg_coefs"]
        L.append("## Logistic-regression coefficients (standardised, "
                 "signed)")
        L.append("Always reported for interpretability. Positive => "
                 "pushes toward MAKE, negative => toward MISS. Top 15 by "
                 "|coef|:\n")
        for f, v in lc[:15]:
            arrow = "MAKE+" if v > 0 else "MISS-"
            L.append(f"- `{f}`: {v}  ({arrow})")
        L.append("")
    OUT_DOC.write_text("\n".join(L))


def main() -> None:
    np.random.seed(SEED)
    df, feat_cols = _load()
    eprint(f"[p3] {len(df)} shots, {len(feat_cols)} features")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if train.empty or val.empty or test.empty:
        raise RuntimeError("a split is empty — check manifest/p2 output")

    Xtr, ytr = train[feat_cols].to_numpy(), train["label"].to_numpy()
    Xva, yva = val[feat_cols].to_numpy(), val["label"].to_numpy()
    Xte, yte = test[feat_cols].to_numpy(), test["label"].to_numpy()

    models = _make_models()
    val_selection: Dict[str, Dict] = {}
    fitted: Dict[str, Pipeline] = {}
    for name, mdl in models.items():
        m = _clone(mdl)
        m.fit(Xtr, ytr)
        fitted[name] = m
        pva = m.predict_proba(Xva)[:, 1]
        thr, vf1 = _best_threshold(yva, pva)
        vacc = accuracy_score(yva, (pva >= thr).astype(int))
        val_selection[name] = {
            "val_f1": vf1, "val_acc": round(float(vacc), 4),
            "threshold": thr,
        }
        eprint(f"[p3] {name}: val_f1={vf1} val_acc={round(vacc,4)} "
               f"thr={thr}")

    selected = max(val_selection, key=lambda k: val_selection[k]["val_f1"])
    threshold = val_selection[selected]["threshold"]
    eprint(f"[p3] selected={selected} threshold={threshold}")

    # Refit selected model on train+val (more data; still no test leak),
    # keeping the val-tuned threshold.
    final = _clone(models[selected])
    trval = pd.concat([train, val])
    Xtv = trval[feat_cols].to_numpy()
    ytv = trval["label"].to_numpy()
    final.fit(Xtv, ytv)

    cv = _group_cv(df, feat_cols, models[selected])
    eprint(f"[p3] CV acc={cv['acc_mean']}±{cv['acc_std']} "
           f"f1={cv['f1_mean']}±{cv['f1_std']}")

    prob_te = final.predict_proba(Xte)[:, 1]
    pred_te = (prob_te >= threshold).astype(int)

    test_metrics = _subset_metrics(test, yte, pred_te, prob_te)
    by_game = _by_game(test, yte, pred_te, prob_te)
    importances = _coeffs(final, feat_cols, Xte, yte, selected)

    # Always expose standardised logistic coefficients (signed, directly
    # interpretable) even when HGB is the deployed model — refit logreg
    # on train+val for an apples-to-apples readout.
    logreg_tv = _clone(models["logreg"])
    logreg_tv.fit(Xtv, ytv)
    logreg_coefs = _coeffs(logreg_tv, feat_cols, Xte, yte, "logreg")

    # Per-shot test predictions (immutable artifact).
    preds = test[["game_id", "play_id", "label", "classification",
                  "v1_in_scope"]].copy()
    preds["pred"] = pred_te
    preds["prob"] = np.round(prob_te, 6)
    preds["correct"] = (preds["pred"] == preds["label"]).astype(int)
    preds = preds.sort_values(["game_id", "play_id"]).reset_index(
        drop=True)
    preds.to_parquet(OUT_PREDS, index=False)

    dump({
        "model": final,
        "selected": selected,
        "threshold": threshold,
        "feat_cols": feat_cols,
        "seed": SEED,
    }, OUT_MODEL)

    ctx = {
        "split_sizes": {
            "train": int(len(train)), "val": int(len(val)),
            "test": int(len(test))},
        "val_selection": val_selection,
        "selected": selected,
        "threshold": threshold,
        "cv": cv,
        "test": test_metrics,
        "by_game": by_game,
        "importances": importances,
        "logreg_coefs": logreg_coefs,
    }
    _write_doc(ctx)

    print(json.dumps({
        "selected": selected,
        "threshold": threshold,
        "cv": cv,
        "test_overall": test_metrics["overall"],
        "test_v1_in_scope": test_metrics["v1_in_scope"],
        "test_4pt": test_metrics.get("4pt_make_only"),
        "anchor": by_game.get(ANCHOR),
        "top_features": importances[:15],
        "logreg_coefs_top": logreg_coefs[:15],
    }, indent=2))


if __name__ == "__main__":
    main()
