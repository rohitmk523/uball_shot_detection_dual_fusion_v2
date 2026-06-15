#!/usr/bin/env python3
"""Re-test fusion with the NEW near-angle CNN. Joins the far/fusion model's
out-of-fold predictions with the new near CNN's LOGO predictions (per shot),
then tries several fusion rules and reports whether fusing the new (strong) near
signal beats far-alone. All honest: simple blends are param-free; learned
fusion uses leave-one-game-out so weights never see the test game.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def metrics(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return acc, prec, rec, int(fp), int(fn)


def auc(p, y):
    order = np.argsort(p); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(len(p)); n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    nr = json.loads((REPO / "data/near_rimcrop/cache/logo_results.json").read_text())
    rows = []
    for fold in nr["folds"]:
        for nm, p, yv in zip(fold["names"], fold["probs"], fold["ys"]):
            rows.append((nm.split("_")[-1], float(p), int(yv)))
    near = pd.DataFrame(rows, columns=["pid8", "prob_near", "y_near"])
    far = pd.read_parquet(REPO / "data/p3_logo_oof_predictions.parquet")
    far["pid8"] = far.play_id.str[:8]
    far = far.rename(columns={"prob": "prob_far"})
    m = far[["game_id", "pid8", "label", "prob_far"]].merge(near, on="pid8", how="inner")
    m = m.drop_duplicates("pid8")
    y = m.label.values.astype(int)
    pf, pn = m.prob_far.values, m.prob_near.values
    print(f"common shots: {len(m)} ({y.sum()} make / {len(y)-y.sum()} miss), "
          f"{m.game_id.nunique()} games\n")

    def show(tag, p):
        a, pr, rc, fp, fn = metrics(p, y)
        print(f"  {tag:28} acc={a:.4f} prec={pr:.3f} rec={rc:.3f} "
              f"AUC={auc(p,y):.4f} FP={fp} FN={fn}")

    print("=== components (out-of-fold, same shots) ===")
    show("FAR / fusion (current)", pf)
    show("NEW near CNN", pn)

    print("\n=== fusion rules ===")
    show("mean blend", (pf + pn) / 2)
    # max-confidence: take the more confident model
    mc = np.where(np.abs(pf - 0.5) >= np.abs(pn - 0.5), pf, pn)
    show("max-confidence", mc)
    # gate: use near when far is uncertain
    gate = np.where(np.abs(pf - 0.5) < 0.2, pn, pf)
    show("gate (near if far unsure)", gate)
    # product / noisy-or style (geometric mean of agreement)
    show("geometric-mean", np.sqrt(np.clip(pf, 1e-4, 1) * np.clip(pn, 1e-4, 1)))

    # learned: leave-one-game-out logistic on [pf, pn]
    from numpy.linalg import lstsq
    games = m.game_id.values
    pstack = np.full(len(y), np.nan)
    for g in np.unique(games):
        tr = games != g; te = games == g
        X = np.column_stack([np.log(np.clip(pf[tr], 1e-4, 1-1e-4)/(1-np.clip(pf[tr],1e-4,1-1e-4))),
                             np.log(np.clip(pn[tr],1e-4,1-1e-4)/(1-np.clip(pn[tr],1e-4,1-1e-4))),
                             np.ones(tr.sum())])
        w, *_ = lstsq(X, y[tr], rcond=None)
        Xte = np.column_stack([np.log(np.clip(pf[te],1e-4,1-1e-4)/(1-np.clip(pf[te],1e-4,1-1e-4))),
                              np.log(np.clip(pn[te],1e-4,1-1e-4)/(1-np.clip(pn[te],1e-4,1-1e-4))),
                              np.ones(te.sum())])
        pstack[te] = Xte @ w
    show("LOGO logistic stack", pstack)

    # best fixed weight (report, but note it's lightly tuned)
    best = (0.5, 0)
    for w in np.arange(0, 1.01, 0.05):
        a = metrics(w*pf + (1-w)*pn, y)[0]
        if a > best[1]:
            best = (w, a)
    print(f"\n  [ref] best fixed far-weight w={best[0]:.2f} -> acc={best[1]:.4f} "
          f"(lightly tuned, upper bound)")


if __name__ == "__main__":
    main()
