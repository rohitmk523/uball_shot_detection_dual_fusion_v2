#!/usr/bin/env python3
"""YOLO vs RF-DETR FUSION comparison (the CLIENT_REPORT 0.973 metric).

Both detectors' tracks (tracks_<det>_<game>.parquet, from track_extract.py) go
through the IDENTICAL downstream: _shot_rows box features + build_geometry g_*
+ cached nm_ (detector-independent) -> angle-aware collapse -> leave-one-game-out
HistGradientBoosting -> prob_far -> mean-blend with the cached near rim-crop CNN
prob_near. So the YOLO-recon validates against ~0.961/0.973 and the RF-DETR delta
is trustworthy (same code, only the detector differs).

  python near_v0/fusion_rfdetr.py --tracks /tmp/tracks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from p2_dataset import _shot_rows          # noqa: E402
from geometry_features import build_geometry  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer    # noqa: E402
from sklearn.pipeline import Pipeline       # noqa: E402

PARAMS = dict(max_depth=4, min_samples_leaf=40, learning_rate=0.08,
              max_iter=500, random_state=42)
GAMES = ["29b51d57", "2c490f1a", "74c4f686", "8dcb1330", "922bff3b",
         "9eb51980", "d0a9faef", "d446fe8c", "f66eb3b2"]
META = {"game_id", "play_id", "classification", "label", "pid8", "side",
        "v1_in_scope", "split", "game8"}


def metrics(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = float((pred == y).mean())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return acc, prec, rec, fp, fn


def angle_aware(feat, side):
    """Collapse _FL/_FR/_NL/_NR -> _FARSIDE/_NEARSIDE per shot's court side."""
    is_left = (feat["pid8"].map(side).fillna("RIGHT") == "LEFT").to_numpy()
    bases = set()
    for c in feat.columns:
        for suf in ("_FL", "_FR", "_NL", "_NR"):
            if c.endswith(suf):
                bases.add(c[:-3])
    new, drop = {}, []
    for base in sorted(bases):
        fl, fr = feat.get(base+"_FL"), feat.get(base+"_FR")
        nl, nr = feat.get(base+"_NL"), feat.get(base+"_NR")
        if fl is not None and fr is not None:
            new[base+"_FARSIDE"] = np.where(is_left, fl, fr); drop += [base+"_FL", base+"_FR"]
        if nl is not None and nr is not None:
            new[base+"_NEARSIDE"] = np.where(is_left, nl, nr); drop += [base+"_NL", base+"_NR"]
    return pd.concat([feat.drop(columns=drop), pd.DataFrame(new, index=feat.index)], axis=1)


def build_prob_far(detector, tdir, v8, side):
    box_rows, geos = [], []
    for g in GAMES:
        tr = pd.read_parquet(f"{tdir}/tracks_{detector}_{g}.parquet")
        box_rows += _shot_rows(g, tr)
        geos.append(build_geometry({g: tr}))
    box = pd.DataFrame(box_rows); geo = pd.concat(geos, ignore_index=True)
    box["pid8"] = box.play_id.astype(str).str[:8]
    geo["pid8"] = geo.play_id.astype(str).str[:8]
    feat = box.merge(geo.drop(columns=[c for c in ("game_id", "play_id") if c in geo]),
                     on="pid8", how="inner")
    nm = [c for c in v8.columns if c.startswith("nm_")]
    feat = feat.merge(v8[["pid8"] + nm].drop_duplicates("pid8"), on="pid8", how="left")
    feat = angle_aware(feat, side)
    cols = [c for c in feat.columns if c not in META and pd.api.types.is_numeric_dtype(feat[c])]
    prob = np.full(len(feat), np.nan)
    for g in feat.game_id.unique():
        tr = feat.game_id != g; te = feat.game_id == g
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", HistGradientBoostingClassifier(**PARAMS))])
        pipe.fit(feat.loc[tr, cols].astype(float).values, feat.loc[tr, "label"].values)
        prob[te.values] = pipe.predict_proba(feat.loc[te, cols].astype(float).values)[:, 1]
    out = feat[["game_id", "pid8", "label"]].copy(); out["prob_far"] = prob
    print(f"  [{detector}] {len(out)} shots, {len(cols)} features", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="/tmp/tracks")
    a = ap.parse_args()

    v8 = pd.read_parquet(REPO / "data/p2_features_v8far.parquet")
    v8["pid8"] = v8.play_id.astype(str).str[:8]
    side = {}
    for g in GAMES:
        for s in json.loads((REPO / f"data/near_detector/demo_data2/{g}.json").read_text())["shots"]:
            side[s["pid8"]] = s["basket"]

    nr = json.loads((REPO / "data/near_rimcrop/cache/logo_results.json").read_text())
    near = pd.DataFrame([(nm.split("_")[-1], float(p)) for fold in nr["folds"]
                         for nm, p in zip(fold["names"], fold["probs"])],
                        columns=["pid8", "prob_near"]).drop_duplicates("pid8")

    print("=== YOLO vs RF-DETR FUSION (mean blend of prob_far + cached near CNN) ===\n")
    res = {}
    for det in ("yolo", "rfdetr"):
        far = build_prob_far(det, a.tracks, v8, side)
        m = far.merge(near, on="pid8", how="inner").drop_duplicates("pid8")
        y = m.label.values.astype(int); pf, pn = m.prob_far.values, m.prob_near.values
        blend = (pf + pn) / 2
        res[det] = (m, y, pf, pn, blend)
        fa = metrics(pf, y); ba = metrics(blend, y)
        print(f"[{det.upper():6}] far-alone acc={fa[0]:.4f} (FP={fa[3]} FN={fa[4]})  |  "
              f"FUSION mean-blend acc={ba[0]:.4f} prec={ba[1]:.3f} rec={ba[2]:.3f} "
              f"FP={ba[3]} FN={ba[4]}  (n={len(m)})", flush=True)

    # per-game fusion table + delta
    print("\n=== per-game fusion (mean blend) ===")
    print(f"{'game':10} {'n':>4} {'YOLO':>7} {'RF-DETR':>8} {'Δ':>7}")
    my, *_ = res["yolo"]; mr, *_ = res["rfdetr"]
    for g in GAMES:
        gy = res["yolo"][0]; gr = res["rfdetr"][0]
        sy = gy[gy.game_id == g]; sr = gr[gr.game_id == g]
        if not len(sy):
            continue
        ay = ((sy.prob_far.values + sy.prob_near.values)/2 >= 0.5).astype(int)
        ar = ((sr.prob_far.values + sr.prob_near.values)/2 >= 0.5).astype(int)
        accy = float((ay == sy.label.values).mean()); accr = float((ar == sr.label.values).mean())
        print(f"{g:10} {len(sy):>4} {accy:>7.3f} {accr:>8.3f} {accr-accy:>+7.3f}")
    oy = metrics(res["yolo"][4], res["yolo"][1])[0]
    orr = metrics(res["rfdetr"][4], res["rfdetr"][1])[0]
    print(f"\nOVERALL fusion:  YOLO {oy:.4f}   RF-DETR {orr:.4f}   delta {orr-oy:+.4f}")
    print("(reference: cached YOLO mean-blend = 0.9727)")


if __name__ == "__main__":
    main()
