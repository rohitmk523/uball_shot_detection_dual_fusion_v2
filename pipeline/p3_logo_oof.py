#!/usr/bin/env python3
"""Out-of-fold (leave-one-game-out) fusion predictions for the training games.

The stacker needs UNBIASED fusion probs on fusion's own training games —
the stored in-sample predictions would leak. Same recipe as p3_angleaware
(HGB, seed 42); each game predicted by a model that never saw it.

Output: data/p3_logo_oof_predictions.parquet
        (game_id, play_id, classification, label, prob, pred)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from p3_angleaware import build_angle_aware, _feature_cols, PARAMS  # noqa: E402


def main() -> int:
    aa = build_angle_aware()
    cols = _feature_cols(aa)
    games = aa.game_id.astype(str).unique()
    print(f"{len(aa)} shots, {len(games)} games, {len(cols)} features")
    oof = np.full(len(aa), np.nan)
    for g in games:
        va = (aa.game_id.astype(str) == g).to_numpy()
        m = HistGradientBoostingClassifier(**PARAMS)
        m.fit(aa.loc[~va, cols], aa.loc[~va, "label"])
        oof[va] = m.predict_proba(aa.loc[va, cols])[:, 1]
        acc = ((oof[va] >= 0.5) == aa.loc[va, "label"]).mean()
        print(f"  {g[:8]}: n={int(va.sum()):3d}  oof acc={100*acc:.1f}%")
    out = aa[["game_id", "play_id", "classification", "label"]].copy()
    out["prob"] = oof
    out["pred"] = (oof >= 0.5).astype(int)
    p = ROOT / "data/p3_logo_oof_predictions.parquet"
    out.to_parquet(p)
    print(f"overall OOF acc: {100*(out.pred==out.label).mean():.2f}%  -> {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
