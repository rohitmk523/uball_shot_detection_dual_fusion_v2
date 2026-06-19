#!/usr/bin/env python3
"""Can the NEAR depth cue rescue the fusion's false positives?

The far angle already gives depth, so naively adding the near cue to the fusion
was redundant. Here we test smarter combinations, all tuned on the 18-game
train/val split and evaluated OUT-OF-SAMPLE on the 5 fresh games:

  base   : production fusion (210 feats)                      -> 0.951 ref
  blend  : w*prob_fus + (1-w)*prob_near        (w tuned on val)
  gate   : if prob_fus in uncertain band -> use prob_near  (band tuned on val)
  flip   : if fusion says MAKE but near depth cue says clearly IN-FRONT -> MISS

Enhanced near depth features add the size *trend* on approach (ball growing =
moving toward the near camera = in front of rim = miss) + pass-geometry. CPU.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import score_fresh as SF                                    # noqa: E402
import p3_angleaware as P3                                  # noqa: E402
import near_noah_compare as NN                              # noqa: E402

TRAIN_TRACKS = Path("/tmp/p1tracks_v16far")
FRESH_TRACKS = Path("/tmp/p1tracks_fresh")
AA = ROOT / "data" / "p2_features_v8far_angleaware.parquet"
PARAMS = NN.PARAMS
END = ["nd2_atcross_ratio_med", "nd2_atcross_ratio_max", "nd2_ratio_trend",
       "nd2_ratio_above", "nd2_ratio_below", "nd2_min_centerdist",
       "nd2_y_span_inside", "nd2_n_atcross", "nd2_inside_frac", "nd2_vcross"]


def nd2(g: pd.DataFrame) -> dict:
    out = {k: np.nan for k in END}
    out["nd2_n_atcross"] = 0; out["nd2_vcross"] = 0
    both = g[g.ball_x.notna() & g.rim_x.notna()]
    if len(both) == 0:
        return out
    bw = both.ball_w.to_numpy(float); rw = both.rim_w.to_numpy(float)
    rh = both.rim_h.to_numpy(float); fr = both.frame_idx.to_numpy(float)
    bcx = (both.ball_x + both.ball_w / 2).to_numpy(float)
    bcy = (both.ball_y + both.ball_h / 2).to_numpy(float)
    rcx = (both.rim_x + both.rim_w / 2).to_numpy(float)
    rcy = (both.rim_y + both.rim_h / 2).to_numpy(float)
    ratio = bw / (rw + 1e-6)
    dx = bcx - rcx; dy = bcy - rcy
    near = (np.abs(dx) < 1.5 * rw) & (np.abs(dy) < 2.0 * rh)
    inside = np.abs(dx) < 0.6 * rw
    atcross = inside & (np.abs(dy) < 0.6 * rh)
    if near.sum() >= 1:
        out["nd2_inside_frac"] = float(inside[near].mean())
        out["nd2_min_centerdist"] = float(
            np.min(np.sqrt(dx[near] ** 2 + dy[near] ** 2) / rw[near]))
    if near.sum() >= 3:                       # size trend over approach
        x = fr[near]; y = ratio[near]
        out["nd2_ratio_trend"] = float(np.polyfit(x - x.mean(), y, 1)[0])
    if atcross.sum() >= 1:
        out["nd2_n_atcross"] = int(atcross.sum())
        out["nd2_atcross_ratio_med"] = float(np.nanmedian(ratio[atcross]))
        out["nd2_atcross_ratio_max"] = float(np.nanmax(ratio[atcross]))
    ab = near & (dy < 0); bl = near & (dy > 0)
    if ab.sum() >= 1:
        out["nd2_ratio_above"] = float(np.nanmedian(ratio[ab]))
    if bl.sum() >= 1:
        out["nd2_ratio_below"] = float(np.nanmedian(ratio[bl]))
    if inside.sum() >= 2:
        dyi = dy[inside]; thr = 0.2 * rh[inside].mean()
        out["nd2_vcross"] = int(dyi.min() < -thr and dyi.max() > thr)
        out["nd2_y_span_inside"] = float((dyi.max() - dyi.min()) / rh[inside].mean())
    return out


def build_nd2(tracks_dir, side_of, games):
    rows = []
    for gid in games:
        pq = tracks_dir / f"{gid}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        for pid, gp in df.groupby("play_id"):
            side = side_of.get(str(pid), "LEFT")
            ncam = "NL" if side == "LEFT" else "NR"
            f = nd2(gp[gp.angle == ncam].sort_values("frame_idx"))
            f["game_id"] = gid; f["play_id"] = str(pid)
            rows.append(f)
    return pd.DataFrame(rows)


def fit_probs(tr, cols, *frames):
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(**PARAMS))])
    p.fit(tr[cols].astype(float).values, tr.label.values)
    return [p.predict_proba(f[cols].astype(float).values)[:, 1] for f in frames]


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(acc=accuracy_score(y, pred), prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0), fp=int(cm[0, 1]), fn=int(cm[1, 0]))


def best_thr(y, prob):
    b = (0.5, -1)
    for t in np.arange(0.15, 0.85, 0.005):
        a = accuracy_score(y, (prob >= t).astype(int))
        if a > b[1]:
            b = (float(t), a)
    return b[0]


def show(tag, y, pred, ref=0.951):
    m = metrics(y, pred)
    print(f"  {tag:<34} acc={m['acc']:.3f}  prec={m['prec']:.3f}  rec={m['rec']:.3f}  "
          f"FP={m['fp']}  FN={m['fn']}  ({m['acc']-ref:+.3f} vs fusion)")
    return m


def main():
    aa = pd.read_parquet(AA); aa["play_id"] = aa.play_id.astype(str)
    tr_side = {r.play_id: ("LEFT" if r.side_is_left == 1 else "RIGHT") for r in aa.itertuples()}
    tg = sorted(aa.game_id.unique())
    fresh = SF.fresh_game_ids(); SF.pull_artifacts(fresh); amap = SF.angle_map()
    faa = SF.apply_angle_aware(SF.build_fresh_v8far(fresh), amap)
    faa["play_id"] = faa.play_id.astype(str); faa["split"] = "fresh"
    fr_side = {pid: amap.get(pid, "LEFT") for pid in faa.play_id}

    nt = build_nd2(TRAIN_TRACKS, tr_side, tg)
    nf = build_nd2(FRESH_TRACKS, fr_side, fresh)
    aa = aa.merge(nt, on=["game_id", "play_id"], how="left")
    faa = faa.merge(nf, on=["game_id", "play_id"], how="left")
    for c in END:
        if c not in faa.columns:
            faa[c] = np.nan

    fus_cols = [c for c in P3._feature_cols(pd.read_parquet(AA)) if c in aa.columns]
    near_cols = NN.near_cols(aa) + END
    tr, va = aa[aa.split == "train"], aa[aa.split == "val"]
    y = faa.label.to_numpy(); yv = va.label.to_numpy()

    fus_v, fus_f = fit_probs(tr, fus_cols, va, faa)
    near_v, near_f = fit_probs(tr, near_cols, va, faa)

    # base fusion
    tf = best_thr(yv, fus_v)
    print("=== strategies, OUT-OF-SAMPLE on 5 fresh games (ref=fusion 0.951) ===")
    base = show("base fusion (210)", y, (fus_f >= tf).astype(int))

    # blend
    bestw = max(np.arange(0, 1.01, 0.1),
                key=lambda w: accuracy_score(yv, (w * fus_v + (1 - w) * near_v >= best_thr(yv, w * fus_v + (1 - w) * near_v)).astype(int)))
    bp_v = bestw * fus_v + (1 - bestw) * near_v
    tb = best_thr(yv, bestw * fus_v + (1 - bestw) * near_v)
    bp_f = bestw * fus_f + (1 - bestw) * near_f
    show(f"blend (w_fus={bestw:.1f})", y, (bp_f >= tb).astype(int))

    # gate: fusion uncertain band -> near decision
    tn = best_thr(yv, near_v)
    best = (-1, None)
    for lo in np.arange(0.15, 0.5, 0.05):
        for hi in np.arange(0.45, 0.8, 0.05):
            predv = np.where((fus_v >= lo) & (fus_v <= hi),
                             (near_v >= tn).astype(int), (fus_v >= tf).astype(int))
            a = accuracy_score(yv, predv)
            if a > best[0]:
                best = (a, (lo, hi))
    lo, hi = best[1]
    gate_f = np.where((fus_f >= lo) & (fus_f <= hi),
                      (near_f >= tn).astype(int), (fus_f >= tf).astype(int))
    show(f"gate (near if fus in [{lo:.2f},{hi:.2f}])", y, gate_f)

    # targeted flip: fusion=MAKE but near depth cue clearly IN-FRONT -> MISS
    cut_best = (-1, None)
    for cut in np.arange(0.20, 0.45, 0.01):
        predv = (fus_v >= tf).astype(int).copy()
        flip = (predv == 1) & (va["nd2_atcross_ratio_med"].to_numpy() < cut)
        predv[flip] = 0
        a = accuracy_score(yv, predv)
        if a > cut_best[0]:
            cut_best = (a, cut)
    cut = cut_best[1]
    flip_f = (fus_f >= tf).astype(int).copy()
    flip_f[(flip_f == 1) & (faa["nd2_atcross_ratio_med"].to_numpy() < cut)] = 0
    show(f"flip (make->miss if ratio<{cut:.2f})", y, flip_f)


if __name__ == "__main__":
    main()
