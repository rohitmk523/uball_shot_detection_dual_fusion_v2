"""
P3 angle-aware fusion (production model).

Each play has a court side (plays.angle = LEFT/RIGHT, mirrored in
gt_windows.json). Only the play's-side cameras see the relevant rim:
  LEFT  -> FL (far) + NL (near)
  RIGHT -> FR (far) + NR (near)
The wrong-side cameras are dead weight (frac_inside_rim ~0.01) and were
adding noise the fusion had to learn to ignore. This module maps every
per-angle feature to FAR-side / NEAR-side per shot and drops the raw
_FL/_FR/_NL/_NR columns, then trains the same HGB recipe.

Input : data/p2_features_v8far.parquet (far_v16 features) + gt_windows angle
Output: data/p2_features_v8far_angleaware.parquet,
        data/p3_model_angleaware.joblib,
        data/p3_test_predictions_angleaware.parquet

Result (held-out test): 0.955 / 0.909 / 0.990 (AUC 0.994); LOGO 0.949/0.912/0.982.
Deterministic seed 42; whole-game splits; balanced val threshold.
"""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
V8FAR = ROOT / "data" / "p2_features_v8far.parquet"
GT_LOCAL = Path("/tmp/gt.json")
GT_S3 = "s3://uball-cv-results/cv-results/dual-fusion-v2/gt/gt_windows.json"
PARAMS = dict(max_depth=4, min_samples_leaf=40, learning_rate=0.08,
              max_iter=500, random_state=42)
SUFFIX = re.compile(r"_(FL|FR|NL|NR)$")


def _angle_map() -> dict:
    if not GT_LOCAL.exists():
        subprocess.run(["aws", "s3", "cp", GT_S3, str(GT_LOCAL),
                        "--region", "us-east-1", "--quiet"], check=True)
    gt = json.load(open(GT_LOCAL))["games"]
    return {s["play_id"]: s.get("angle", "LEFT")
            for sh in gt.values() for s in sh}


def build_angle_aware() -> pd.DataFrame:
    v = pq.read_table(V8FAR).to_pandas()
    side = v.play_id.map(_angle_map()).fillna("LEFT")
    is_left = (side == "LEFT").to_numpy()

    families: dict[str, dict] = {}
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

    aa = pd.concat(
        [v.drop(columns=drop_raw).reset_index(drop=True),
         pd.DataFrame(new_cols)], axis=1)
    aa["side_is_left"] = is_left.astype(int)
    return aa


def _feature_cols(df: pd.DataFrame) -> list:
    drop = {"game_id", "play_id", "classification", "label",
            "v1_in_scope", "split"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def _fit_with_balanced_threshold(tr, va, cols):
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(**PARAMS))])
    p.fit(tr[cols].astype(float).values, tr.label.values)
    pv = p.predict_proba(va[cols].astype(float).values)[:, 1]
    yv = va.label.values
    best = (0.5, -1e9)
    for thr in np.arange(0.20, 0.85, 0.005):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0:
            continue
        prec = precision_score(yv, pred, zero_division=0)
        rec = recall_score(yv, pred, zero_division=0)
        score = -(max(0.90 - prec, 0) + max(0.90 - rec, 0))
        if score > best[1]:
            best = (float(thr), score)
    return p, best[0]


def main() -> None:
    aa = build_angle_aware()
    aa.to_parquet(ROOT / "data" / "p2_features_v8far_angleaware.parquet",
                  index=False)
    cols = _feature_cols(aa)
    tr = aa[aa.split == "train"]
    va = aa[aa.split == "val"]
    te = aa[aa.split == "test"].reset_index(drop=True)
    print(f"angle-aware features: {len(cols)} cols, "
          f"{len(tr)}/{len(va)}/{len(te)} train/val/test")

    pipe, thr = _fit_with_balanced_threshold(tr, va, cols)
    pt = pipe.predict_proba(te[cols].astype(float).values)[:, 1]
    yt = te.label.values
    yp = (pt >= thr).astype(int)
    print(f"\nHELD-OUT TEST (thr={thr:.2f}): "
          f"acc={accuracy_score(yt, yp):.3f} "
          f"prec={precision_score(yt, yp, zero_division=0):.3f} "
          f"rec={recall_score(yt, yp, zero_division=0):.3f} "
          f"auc={roc_auc_score(yt, pt):.3f} "
          f"cm={confusion_matrix(yt, yp).tolist()}")

    joblib.dump(pipe, ROOT / "data" / "p3_model_angleaware.joblib")
    out = te[["game_id", "play_id", "classification", "label",
              "v1_in_scope"]].copy()
    out["prob"] = pt
    out["pred"] = yp
    out["correct"] = (yp == yt)
    out.to_parquet(ROOT / "data" / "p3_test_predictions_angleaware.parquet",
                   index=False)
    print("saved p3_model_angleaware.joblib + p3_test_predictions_angleaware.parquet")


if __name__ == "__main__":
    main()
