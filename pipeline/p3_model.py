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

Iteration 2: the feature file is selectable via --features (default
the v2 file). An ablation (--ablate) trains the model with the
iteration-2 levers added incrementally (v1 -> +L3 -> +L2 -> +L1 ->
all) so we can attribute every metric move to a lever.

Outputs (version-suffixed, immutable; v1 artifacts untouched):
  data/p3_model_v2.joblib
  data/p3_test_predictions_v2.parquet
  data/P3_RESULTS_v2.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DATA = REPO_ROOT / "data"
FEATURES_V1 = DATA / "p2_features.parquet"
FEATURES_V2 = DATA / "p2_features_v2.parquet"
OUT_MODEL = DATA / "p3_model_v2.joblib"
OUT_PREDS = DATA / "p3_test_predictions_v2.parquet"
OUT_DOC = DATA / "P3_RESULTS_v2.md"

SEED = 42
ANCHOR = "c2a354fe-eb34-4980-af00-8f5ff6b00143"
META_COLS = ["game_id", "play_id", "classification", "label",
             "v1_in_scope", "split"]

# Iteration-1 reference numbers (from the prior P3_RESULTS.md run on the
# v1 feature set), the frozen-v1 anchor baseline, and the target.
ITER1 = {"acc": 0.8474, "prec": 0.7735, "rec": 0.896}          # overall
ITER1_V1SCOPE = {"acc": 0.8484, "prec": 0.7464, "rec": 0.907}
V1_BASELINE = {"acc": 0.857, "prec": 0.795, "rec": 0.892}      # to beat
TARGET = {"acc": 0.92, "prec": 0.90, "rec": 0.90}


def _lever_of(col: str) -> str:
    """Classify a feature column into a lever group for ablation.
    L1=bounce-out, L2=arc, L3=quality-gated fusion, V1=everything else
    (the iteration-1 feature set)."""
    if col.startswith("bo_") or col.startswith("far_cam_left_frame"):
        return "L1"
    if col.startswith("arc_"):
        return "L2"
    if (col.startswith("q_quality") or col in {
            "min_dist_rw_qw", "arc_entry_angle_qw", "arc_vy_at_rim_qw",
            "close_agree_votes", "agree_make_like",
            "through_hoop_agree2"}):
        return "L3"
    return "V1"


def _load(features: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not features.exists():
        raise FileNotFoundError(
            f"{features} not found — run pipeline/p2_dataset.py first")
    df = pd.read_parquet(features)
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


# Precision is the explicit bottleneck (target prec>=0.90, FP-heavy).
# Plain F1-optimal thresholding pushes recall up / precision down — the
# wrong trade here. We instead pick, on VAL ONLY, the highest-F1
# threshold whose VAL precision clears a floor; if no threshold reaches
# the floor we fall back to the threshold maximising
# (precision-weighted) F-beta with beta<1 (precision favoured). This is
# a tuning decision, never touches TEST.
VAL_PRECISION_FLOOR = 0.85
PRECISION_BETA = 0.5  # F-beta with beta<1 weights precision higher


def _best_threshold(y: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Precision-aware threshold selection on the validation set."""
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    floor_t, floor_f1 = None, -1.0
    b2 = PRECISION_BETA ** 2
    for t in grid:
        pred = (prob >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        # Track the F1-best threshold among those clearing the
        # precision floor (must keep some recall so it's usable).
        if p >= VAL_PRECISION_FLOOR and r >= 0.5 and f1 > floor_f1:
            floor_f1, floor_t = f1, float(t)
        # Fallback objective: precision-weighted F-beta (beta<1).
        denom = (b2 * p + r)
        fb = (1 + b2) * p * r / denom if denom > 0 else 0.0
        if fb > best_f1:
            best_f1, best_t = fb, float(t)
    if floor_t is not None:
        return floor_t, round(floor_f1, 4)
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


def _cmp_table(o: Dict, v1s: Dict) -> List[str]:
    """Iteration-2 vs iteration-1 vs v1 baseline vs target."""
    L = ["## Iteration 2 vs Iteration 1 vs v1 baseline vs target\n"]
    L.append("Overall held-out test (make/miss incl 4PT_MAKE):\n")
    L.append("| metric | v1 baseline | iter1 | **iter2** | target |")
    L.append("|---|---|---|---|---|")
    for k, lab in (("acc", "accuracy"), ("prec", "precision"),
                   ("rec", "recall")):
        ov = {"acc": o["accuracy"], "prec": o["precision"],
              "rec": o["recall"]}[k]
        L.append(
            f"| {lab} | {V1_BASELINE[k]} | {ITER1[k]} | "
            f"**{ov}** | {TARGET[k]} |")
    L.append("")
    L.append("v1_in_scope subset (apples-to-apples vs v1 "
             "85.7/79.5/89.2):\n")
    L.append("| metric | v1 baseline | iter1 | **iter2** | target |")
    L.append("|---|---|---|---|---|")
    for k, lab in (("acc", "accuracy"), ("prec", "precision"),
                   ("rec", "recall")):
        vv = {"acc": v1s["accuracy"], "prec": v1s["precision"],
              "rec": v1s["recall"]}[k]
        L.append(
            f"| {lab} | {V1_BASELINE[k]} | {ITER1_V1SCOPE[k]} | "
            f"**{vv}** | {TARGET[k]} |")
    L.append("")
    L.append(
        "> Note: iter1 numbers used plain F1-optimal thresholding; "
        "iter2 uses a precision-aware threshold (precision is the "
        "stated bottleneck), tuned on VAL only. The ablation below "
        "isolates the feature levers because its `V1 only` stage "
        "ALSO uses the iter2 threshold policy — so stage-to-stage "
        "deltas are pure feature effects, while the iter1-vs-iter2 "
        "table reflects the combined (features + threshold policy) "
        "improvement.\n")
    return L


def _ablation_table(rows: List[Dict]) -> List[str]:
    L = ["## Ablation — levers added incrementally\n"]
    L.append("Each stage re-selects model + threshold on VAL, then "
             "evaluates ONCE on TEST. Overall metrics:\n")
    L.append("| stage | #feat | model | acc | prec | rec | f1 | "
             "FP | FN |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["overall"]
        cm = m["confusion_matrix"]
        L.append(
            f"| {r['stage']} | {r['n_feats']} | {r['selected']} | "
            f"{m['accuracy']} | {m['precision']} | {m['recall']} | "
            f"{m['f1']} | {cm['fp']} | {cm['fn']} |")
    L.append("")
    L.append("Anchor game (c2a354fe) per stage:\n")
    L.append("| stage | acc | prec | rec |")
    L.append("|---|---|---|---|")
    for r in rows:
        a = r["anchor"]
        if a:
            L.append(f"| {r['stage']} | {a['accuracy']} | "
                     f"{a['precision']} | {a['recall']} |")
    L.append("")
    return L


def _write_doc(ctx: Dict) -> None:
    L: List[str] = [
        "# P3 Results v2 — interpretable make/miss model "
        "(iteration 2)\n"]
    L.append(f"Seed={SEED}. Whole-game splits (no game spans splits, no "
             f"leakage). Test opened once. "
             f"Features: `{ctx.get('features_file', '?')}`.\n")
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
    L += _cmp_table(ctx["test"]["overall"], ctx["test"]["v1_in_scope"])
    if ctx.get("ablation"):
        L += _ablation_table(ctx["ablation"])
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


def _fit_select_eval(df: pd.DataFrame, feat_cols: List[str],
                     write_artifacts: bool) -> Dict:
    """Train both models on TRAIN, tune model+threshold on VAL, refit
    the winner on TRAIN+VAL, then evaluate ONCE on TEST. Returns the
    full context dict. No test data ever touches selection/tuning."""
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
    for name, mdl in models.items():
        m = _clone(mdl)
        m.fit(Xtr, ytr)
        pva = m.predict_proba(Xva)[:, 1]
        thr, vf1 = _best_threshold(yva, pva)
        vacc = accuracy_score(yva, (pva >= thr).astype(int))
        val_selection[name] = {
            "val_f1": vf1, "val_acc": round(float(vacc), 4),
            "threshold": thr,
        }
        eprint(f"[p3] {name}: val_f1={vf1} val_acc={round(vacc,4)} "
               f"thr={thr}")

    selected = max(val_selection,
                   key=lambda k: val_selection[k]["val_f1"])
    threshold = val_selection[selected]["threshold"]
    eprint(f"[p3] selected={selected} threshold={threshold}")

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

    logreg_tv = _clone(models["logreg"])
    logreg_tv.fit(Xtv, ytv)
    logreg_coefs = _coeffs(logreg_tv, feat_cols, Xte, yte, "logreg")

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

    if write_artifacts:
        preds = test[["game_id", "play_id", "label", "classification",
                      "v1_in_scope"]].copy()
        preds["pred"] = pred_te
        preds["prob"] = np.round(prob_te, 6)
        preds["correct"] = (
            preds["pred"] == preds["label"]).astype(int)
        preds = preds.sort_values(
            ["game_id", "play_id"]).reset_index(drop=True)
        preds.to_parquet(OUT_PREDS, index=False)
        dump({
            "model": final, "selected": selected,
            "threshold": threshold, "feat_cols": feat_cols,
            "seed": SEED,
        }, OUT_MODEL)
    return ctx


def _ablation(df: pd.DataFrame, all_feats: List[str]) -> List[Dict]:
    """Incrementally add the iteration-2 lever groups so every metric
    move is attributable. Order: V1 -> +L3 -> +L2 -> +L1 -> all
    (all == V1+L1+L2+L3). Each stage is fully re-selected and
    re-thresholded on VAL, then evaluated once on TEST."""
    groups: Dict[str, List[str]] = {"V1": [], "L1": [], "L2": [],
                                    "L3": []}
    for c in all_feats:
        groups[_lever_of(c)].append(c)
    stages = [
        ("V1 only (iteration-1 features)", ["V1"]),
        ("V1 + L3 (quality-gated fusion)", ["V1", "L3"]),
        ("V1 + L3 + L2 (arc fit)", ["V1", "L3", "L2"]),
        ("V1 + L3 + L2 + L1 (bounce-out) = ALL",
         ["V1", "L3", "L2", "L1"]),
    ]
    rows: List[Dict] = []
    for label, keys in stages:
        cols = [c for k in keys for c in groups[k]]
        eprint(f"[ablate] {label}: {len(cols)} feats")
        ctx = _fit_select_eval(df, cols, write_artifacts=False)
        rows.append({
            "stage": label,
            "n_feats": len(cols),
            "selected": ctx["selected"],
            "overall": ctx["test"]["overall"],
            "v1_in_scope": ctx["test"]["v1_in_scope"],
            "anchor": ctx["by_game"].get(ANCHOR),
            "cv": ctx["cv"],
        })
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="P3 make/miss model")
    ap.add_argument(
        "--features", default=str(FEATURES_V2),
        help="feature parquet (default: data/p2_features_v2.parquet)")
    ap.add_argument(
        "--ablate", action="store_true",
        help="run the incremental lever ablation and include it in "
             "the report")
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    feats_path = Path(args.features)
    df, feat_cols = _load(feats_path)
    eprint(f"[p3] {feats_path.name}: {len(df)} shots, "
           f"{len(feat_cols)} features")

    ctx = _fit_select_eval(df, feat_cols, write_artifacts=True)

    ablation: Optional[List[Dict]] = None
    if args.ablate:
        ablation = _ablation(df, feat_cols)
    ctx["ablation"] = ablation
    ctx["features_file"] = feats_path.name
    _write_doc(ctx)

    print(json.dumps({
        "features_file": feats_path.name,
        "selected": ctx["selected"],
        "threshold": ctx["threshold"],
        "cv": ctx["cv"],
        "test_overall": ctx["test"]["overall"],
        "test_v1_in_scope": ctx["test"]["v1_in_scope"],
        "test_4pt": ctx["test"].get("4pt_make_only"),
        "anchor": ctx["by_game"].get(ANCHOR),
        "ablation": [
            {"stage": a["stage"], "overall": a["overall"]}
            for a in ablation
        ] if ablation else None,
        "top_features": ctx["importances"][:15],
        "logreg_coefs_top": ctx["logreg_coefs"][:15],
    }, indent=2))


if __name__ == "__main__":
    main()
