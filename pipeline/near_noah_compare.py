#!/usr/bin/env python3
"""NEAR-only make/miss: OLD near features vs NEW (+ Noah-style depth cue).

Near detection is NOT the limiter (ball seen at rim in 100% of shots), so the
near angle's residual error is geometric: from an oblique near view a ball
passing IN FRONT of the rim looks like it went through (depth illusion). Noah
resolves this with a known-ball-size scale. We replicate that monocular cue on
the existing near detections:

    depth_ratio = ball_width_px / (rim_width_px * 9.5/18)
      ~1  ball at the rim plane (through/at)        <1  behind the rim
      >1  ball IN FRONT of the rim  (the illusion -> usually a MISS)

We compute nd_* features per shot from the play's near camera (LEFT->NL,
RIGHT->NR), then train two NEAR-ONLY HGB models on the 18 training games and
evaluate OUT-OF-SAMPLE on the 5 fresh games:
    OLD  = existing near-side features (_NEARSIDE + g_near_*)
    NEW  = OLD + nd_* (Noah depth cue)
Reports whether the depth cue cuts the near angle's false positives. CPU only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import score_fresh as SF                                   # noqa: E402

TRAIN_TRACKS = Path("/tmp/p1tracks_v16far")
FRESH_TRACKS = Path("/tmp/p1tracks_fresh")
AA_TRAIN = ROOT / "data" / "p2_features_v8far_angleaware.parquet"
EXP = 9.5 / 18.0
PARAMS = dict(max_depth=4, min_samples_leaf=40, learning_rate=0.08,
              max_iter=500, random_state=42)
ND_KEYS = ["nd_atcross_n", "nd_atcross_ratio_med", "nd_atcross_ratio_max",
           "nd_atcross_ratio_closest", "nd_inside_frac", "nd_vcross",
           "nd_below_after_inside", "nd_dr_closest"]


def nd_features(g: pd.DataFrame) -> dict:
    """Monocular Noah-style depth cue at the rim-crossing moment.

    The discriminative signal (validated AUC~0.60) is the ball's apparent size
    relative to the rim AT the frames where the ball is horizontally inside the
    rim AND vertically at rim level — NOT a min over the whole approach (that
    just grabs the small/far apex frames). dr uses the 9.5/18 scale only for the
    'closest' frame; the at-cross features use the raw ball/rim size ratio.
    """
    out = {k: np.nan for k in ND_KEYS}
    out["nd_atcross_n"] = 0; out["nd_vcross"] = 0; out["nd_below_after_inside"] = 0
    both = g[g.ball_x.notna() & g.rim_x.notna()]
    if len(both) == 0:
        return out
    bw = both.ball_w.to_numpy(float)
    rw = both.rim_w.to_numpy(float)
    rh = both.rim_h.to_numpy(float)
    bcx = (both.ball_x + both.ball_w / 2).to_numpy(float)
    bcy = (both.ball_y + both.ball_h / 2).to_numpy(float)
    rcx = (both.rim_x + both.rim_w / 2).to_numpy(float)
    rcy = (both.rim_y + both.rim_h / 2).to_numpy(float)
    ratio = bw / (rw + 1e-6)               # apparent ball/rim size ratio
    dx = bcx - rcx
    dy = bcy - rcy                          # >0 = ball below rim center
    near = (np.abs(dx) < 1.5 * rw) & (np.abs(dy) < 2.0 * rh)
    inside = np.abs(dx) < 0.6 * rw
    atcross = inside & (np.abs(dy) < 0.6 * rh)   # over hoop AND at rim level
    if near.sum() >= 1:
        out["nd_inside_frac"] = float(inside[near].mean())
        out["nd_dr_closest"] = float((bw / (rw * EXP + 1e-6))[near][
            np.argmin(np.abs(dy[near]))])
    if atcross.sum() >= 1:
        r = ratio[atcross]
        out["nd_atcross_n"] = int(atcross.sum())
        out["nd_atcross_ratio_med"] = float(np.nanmedian(r))
        out["nd_atcross_ratio_max"] = float(np.nanmax(r))
        out["nd_atcross_ratio_closest"] = float(
            ratio[atcross][np.argmin(np.abs(dy[atcross]))])
    if inside.sum() >= 2:
        dyi = dy[inside]; thr = 0.2 * rh[inside].mean()
        out["nd_vcross"] = int(dyi.min() < -thr and dyi.max() > thr)
        out["nd_below_after_inside"] = int(dyi.max() > 0.5 * rh[inside].mean())
    return out


def build_nd(tracks_dir: Path, side_of: dict, games: list) -> pd.DataFrame:
    rows = []
    for gid in games:
        pq = tracks_dir / f"{gid}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        for pid, gp in df.groupby("play_id"):
            side = side_of.get(pid, "LEFT")
            ncam = "NL" if side == "LEFT" else "NR"
            f = nd_features(gp[gp.angle == ncam].sort_values("frame_idx"))
            f["game_id"] = gid; f["play_id"] = str(pid)
            rows.append(f)
    return pd.DataFrame(rows)


def near_cols(df: pd.DataFrame) -> list:
    keep = [c for c in df.columns
            if c.endswith("_NEARSIDE") or c in ("g_near_dmin", "g_near_inside")]
    return keep


def fit_eval(tr, va, te, cols, tag):
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
        s = -(max(0.90 - precision_score(yv, pred, zero_division=0), 0) +
              max(0.90 - recall_score(yv, pred, zero_division=0), 0))
        if s > best[1]:
            best = (float(thr), s)
    thr = best[0]
    pt = p.predict_proba(te[cols].astype(float).values)[:, 1]
    yt = te.label.values
    yp = (pt >= thr).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    print(f"  {tag:<28} acc={accuracy_score(yt,yp):.3f} "
          f"prec={precision_score(yt,yp,zero_division=0):.3f} "
          f"rec={recall_score(yt,yp,zero_division=0):.3f} "
          f"auc={roc_auc_score(yt,pt):.3f} FP={cm[0,1]} FN={cm[1,0]} (thr={thr:.2f})")
    return dict(acc=accuracy_score(yt, yp), prec=precision_score(yt, yp, zero_division=0),
                rec=recall_score(yt, yp, zero_division=0), fp=int(cm[0, 1]),
                fn=int(cm[1, 0]), thr=thr, prob=pt, pred=yp)


def main():
    # ---- training angle-aware features (+ side from side_is_left) ----
    aa = pd.read_parquet(AA_TRAIN)
    aa["play_id"] = aa.play_id.astype(str)
    tr_side = {r.play_id: ("LEFT" if r.side_is_left == 1 else "RIGHT")
               for r in aa.itertuples()}
    train_games = sorted(aa.game_id.unique())

    # ---- fresh angle-aware features (reuse score_fresh) ----
    fresh_games = SF.fresh_game_ids()
    SF.pull_artifacts(fresh_games)
    amap = SF.angle_map()
    fv = SF.build_fresh_v8far(fresh_games)
    faa = SF.apply_angle_aware(fv, amap)
    faa["play_id"] = faa.play_id.astype(str)
    faa["split"] = "fresh"
    fr_side = {pid: amap.get(pid, "LEFT") for pid in faa.play_id}

    # ---- Noah depth features for train + fresh ----
    nd_tr = build_nd(TRAIN_TRACKS, tr_side, train_games)
    nd_fr = build_nd(FRESH_TRACKS, fr_side, fresh_games)
    print(f"[nd] train depth-feat rows={len(nd_tr)}  fresh rows={len(nd_fr)}")
    # quick separation check on fresh: depth ratio over-hoop, makes vs misses
    j = faa[["game_id", "play_id", "label"]].merge(nd_fr, on=["game_id", "play_id"], how="left")
    for lab, name in [(1, "MAKE"), (0, "MISS")]:
        s = j[j.label == lab]["nd_atcross_ratio_med"].dropna()
        print(f"  fresh {name}: nd_atcross_ratio_med median={s.median():.3f} "
              f"(ball/rim size at rim crossing)  n={len(s)}")

    aa = aa.merge(nd_tr, on=["game_id", "play_id"], how="left")
    faa = faa.merge(nd_fr, on=["game_id", "play_id"], how="left")

    old = near_cols(aa)
    new = old + ND_KEYS
    print(f"\nnear-only features: OLD={len(old)}  NEW={len(new)} (+{len(ND_KEYS)} nd_)")
    tr = aa[aa.split == "train"]; va = aa[aa.split == "val"]
    print(f"train={len(tr)} val={len(va)} fresh(test)={len(faa)} "
          f"(makes={int(faa.label.sum())})\n")
    print("=== NEAR-ONLY, evaluated OUT-OF-SAMPLE on 5 fresh games ===")
    ro = fit_eval(tr, va, faa, old, "OLD near (existing feats)")
    rn = fit_eval(tr, va, faa, new, "NEW near (+ Noah depth cue)")
    print(f"\nΔ from Noah depth cue:  acc {ro['acc']:.3f}->{rn['acc']:.3f} "
          f"({rn['acc']-ro['acc']:+.3f})   FP {ro['fp']}->{rn['fp']} "
          f"({rn['fp']-ro['fp']:+d})   FN {ro['fn']}->{rn['fn']} ({rn['fn']-ro['fn']:+d})")
    print("(reference: full far+near fusion on these games = 0.951 acc)")


if __name__ == "__main__":
    main()
