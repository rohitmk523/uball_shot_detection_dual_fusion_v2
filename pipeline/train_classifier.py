#!/usr/bin/env python3
"""End-to-end multi-camera MAKE/MISS classifier.

Trains a gradient boosting classifier on the per-shot features from
extract_features.py. Validation: leave-one-game-out -- train on 2 games,
predict the 3rd. Compares vs the rule-based ensemble pipeline.
"""
from __future__ import annotations
import csv, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/client_report/triangulation_test/shot_features.csv"

FEATURES = [
    "n_samples", "apex_r", "apex_z", "cross_r", "z_min", "bounce_cm",
    "tri_make", "tri_miss", "tri_und",
    "fr_strength", "fr_is_make", "fr_is_miss",
    "fr_n_deep", "fr_rebound_px", "fr_max_cy",
    "nr_strength", "nr_is_make", "nr_is_miss",
    "nr_n_deep", "nr_rebound_px", "nr_max_cy",
    "tri_has_rim_out", "tri_has_rim_bounce", "tri_has_gap_stop",
    "tri_has_smooth_descent", "tri_has_pass_through",
    "tri_has_clean_clean", "tri_has_no_clear",
]


def load_rows() -> list[dict]:
    rows = []
    with open(CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([int(r["y"]) for r in rows])
    names = [r["name"] for r in rows]
    return X, y, names


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier

    rows = load_rows()
    print(f"loaded {len(rows)} shots")
    print(f"  G1={sum(1 for r in rows if r['game']=='G1')}  "
          f"G2={sum(1 for r in rows if r['game']=='G2')}  "
          f"G3={sum(1 for r in rows if r['game']=='G3')}")

    # Leave-one-game-out evaluation
    print("\n=== Leave-One-Game-Out Cross-Validation ===")
    for test_game in ["G1", "G2", "G3"]:
        train = [r for r in rows if r["game"] != test_game]
        test = [r for r in rows if r["game"] == test_game]
        if not test: continue
        Xtr, ytr, _ = to_xy(train)
        Xte, yte, names = to_xy(test)
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42)
        clf.fit(Xtr, ytr)
        # use 0.5 threshold; could tune
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        # Compare to rule-based ensemble "final" verdict from CSV — load again
        tp = int(((pred == 1) & (yte == 1)).sum())
        tn = int(((pred == 0) & (yte == 0)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        n = len(yte); dec = tp+tn+fp+fn
        acc = 100*(tp+tn)/dec if dec else 0
        print(f"\n  Train={','.join(g for g in ['G1','G2','G3'] if g != test_game)}  "
              f"Test={test_game}  N={n}")
        print(f"    TP={tp} TN={tn} FP={fp} FN={fn}  "
              f"acc={acc:.1f}%")
        # Show top features (only on G3 hold-out — most interesting)
        if test_game == "G3":
            imp = clf.feature_importances_
            order = np.argsort(-imp)[:10]
            print(f"    Top features:")
            for j in order:
                print(f"      {FEATURES[j]:25s}  {imp[j]:.3f}")
            # Show G3 errors
            print(f"\n    G3 hold-out errors:")
            for i, name in enumerate(names):
                if pred[i] != yte[i]:
                    print(f"      {name:18s} gt={'MAKE' if yte[i] else 'MISS'} "
                          f"pred={'MAKE' if pred[i] else 'MISS'} prob={prob[i]:.2f}")

    # Train on ALL and report training accuracy (sanity check)
    X, y, _ = to_xy(rows)
    clf_all = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=42)
    clf_all.fit(X, y)
    pred = clf_all.predict(X)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y); dec = tp+tn+fp+fn
    print(f"\n  All-in training fit (sanity): "
          f"TP={tp} TN={tn} FP={fp} FN={fn}  acc={100*(tp+tn)/dec:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
