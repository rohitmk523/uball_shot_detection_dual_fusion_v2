#!/usr/bin/env python3
"""Sync-aware cross-view fusion.

For each game we derive the per-side FAR<->NEAR frame offset from MAKE shots
(use the rim-crossing instant as a synced real-world event), then for every
shot build cross-view "do the two sides agree at the SAME real moment?"
features. The depth illusion's signature is: near sees ball-at-rim, far at the
SAME instant sees ball-not-at-rim. With proper sync, that disagreement is now
measurable.

Phase A: derive per-game per-side offsets (LEFT side uses FL<->NL, RIGHT uses
         FR<->NR). User measured +10 on game b3c1f62c (RIGHT) - we should
         reproduce that.
Phase B: build sync-aware features for train (18 games) + fresh (5 games),
         retrain HGB on top of the production 210 features, evaluate
         out-of-sample. Compare vs fusion baseline (0.951).
CPU only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import score_fresh as SF
import p3_angleaware as P3
import near_noah_compare as NN

TRAIN_TRACKS = Path("/tmp/p1tracks_v16far")
FRESH_TRACKS = Path("/tmp/p1tracks_fresh")
AA = ROOT / "data" / "p2_features_v8far_angleaware.parquet"
MAKES = {"FG_MAKE", "3PT_MAKE", "4PT_MAKE", "FREE_THROW_MAKE"}
PARAMS = NN.PARAMS

SA_KEYS = [
    "sa_far_event", "sa_near_event",
    "sa_near_inside_at_far_sync", "sa_far_inside_at_near_sync",
    "sa_near_dx_at_far_sync", "sa_near_dy_at_far_sync",
    "sa_far_dx_at_near_sync", "sa_far_dy_at_near_sync",
    "sa_disagree_far_to_near", "sa_disagree_near_to_far",
]


def event_frame(g: pd.DataFrame) -> int | None:
    """frame_idx where the ball is at the rim plane (inside horizontally, at
    rim level vertically). None if no such frame in this slice."""
    both = g[g.ball_x.notna() & g.rim_x.notna()]
    if len(both) == 0:
        return None
    bcx = (both.ball_x + both.ball_w/2).to_numpy(float)
    bcy = (both.ball_y + both.ball_h/2).to_numpy(float)
    rcx = (both.rim_x + both.rim_w/2).to_numpy(float)
    rcy = (both.rim_y + both.rim_h/2).to_numpy(float)
    rw  = both.rim_w.to_numpy(float); rh = both.rim_h.to_numpy(float)
    fr  = both.frame_idx.to_numpy(int)
    dx, dy = bcx-rcx, bcy-rcy
    mask = (np.abs(dx) < 0.6*rw) & (np.abs(dy) < 0.6*rh)
    if mask.sum() == 0:
        return None
    return int(fr[mask][np.argmin(np.abs(dy[mask]))])


def at_frame(g: pd.DataFrame, fr: int) -> dict | None:
    """ball/rim geometry at the row whose frame_idx is closest to `fr` (within
    one frame). None if no row within ±1 frame or geometry missing."""
    s = g[g.ball_x.notna() & g.rim_x.notna()]
    if len(s) == 0: return None
    i = (s.frame_idx - fr).abs().idxmin()
    if abs(int(s.loc[i, "frame_idx"]) - fr) > 1: return None
    r = s.loc[i]
    rcx, rcy = r.rim_x + r.rim_w/2, r.rim_y + r.rim_h/2
    bcx, bcy = r.ball_x + r.ball_w/2, r.ball_y + r.ball_h/2
    return dict(dx=float((bcx-rcx)/r.rim_w), dy=float((bcy-rcy)/r.rim_h),
                inside=int(abs(bcx-rcx) < 0.6*r.rim_w and abs(bcy-rcy) < 0.6*r.rim_h))


def derive_offsets(df: pd.DataFrame, side_of: dict) -> dict:
    """{LEFT: median(f_NL - f_FL), RIGHT: median(f_NR - f_FR), n_LEFT, n_RIGHT}"""
    diffs = {"LEFT": [], "RIGHT": []}
    for pid, gp in df.groupby("play_id"):
        cls = gp.classification.iloc[0]
        if cls not in MAKES: continue
        side = side_of.get(str(pid))
        if side not in ("LEFT", "RIGHT"): continue
        fcam = "FL" if side == "LEFT" else "FR"
        ncam = "NL" if side == "LEFT" else "NR"
        f_far  = event_frame(gp[gp.angle == fcam].sort_values("frame_idx"))
        f_near = event_frame(gp[gp.angle == ncam].sort_values("frame_idx"))
        if f_far is not None and f_near is not None:
            diffs[side].append(f_near - f_far)
    out = {}
    for s in ("LEFT", "RIGHT"):
        out[f"n_{s}"] = len(diffs[s])
        out[s] = int(np.median(diffs[s])) if diffs[s] else None
        out[f"std_{s}"] = float(np.std(diffs[s])) if len(diffs[s]) >= 2 else None
    return out


def sa_features(gp: pd.DataFrame, side: str, off_left: int | None,
                off_right: int | None) -> dict:
    out = {k: np.nan for k in SA_KEYS}
    for k in ("sa_far_event", "sa_near_event",
              "sa_near_inside_at_far_sync", "sa_far_inside_at_near_sync",
              "sa_disagree_far_to_near", "sa_disagree_near_to_far"):
        out[k] = 0
    off = off_left if side == "LEFT" else off_right
    if off is None: return out
    fcam = "FL" if side == "LEFT" else "FR"
    ncam = "NL" if side == "LEFT" else "NR"
    far_g  = gp[gp.angle == fcam].sort_values("frame_idx")
    near_g = gp[gp.angle == ncam].sort_values("frame_idx")
    f_far  = event_frame(far_g)
    f_near = event_frame(near_g)
    if f_far is not None:
        out["sa_far_event"] = 1
        near_at = at_frame(near_g, f_far + off)
        if near_at:
            out["sa_near_dx_at_far_sync"] = abs(near_at["dx"])
            out["sa_near_dy_at_far_sync"] = abs(near_at["dy"])
            out["sa_near_inside_at_far_sync"] = near_at["inside"]
            out["sa_disagree_far_to_near"] = int(near_at["inside"] == 0)
    if f_near is not None:
        out["sa_near_event"] = 1
        far_at = at_frame(far_g, f_near - off)
        if far_at:
            out["sa_far_dx_at_near_sync"] = abs(far_at["dx"])
            out["sa_far_dy_at_near_sync"] = abs(far_at["dy"])
            out["sa_far_inside_at_near_sync"] = far_at["inside"]
            out["sa_disagree_near_to_far"] = int(far_at["inside"] == 0)
    return out


def build_sa(tracks_dir: Path, side_of: dict, games: list, offsets: dict) -> pd.DataFrame:
    rows = []
    for gid in games:
        pq = tracks_dir / f"{gid}.parquet"
        if not pq.exists(): continue
        df = pd.read_parquet(pq)
        oL = offsets.get(gid, {}).get("LEFT")
        oR = offsets.get(gid, {}).get("RIGHT")
        for pid, gp in df.groupby("play_id"):
            side = side_of.get(str(pid))
            if side not in ("LEFT", "RIGHT"): continue
            f = sa_features(gp, side, oL, oR)
            f["game_id"] = gid; f["play_id"] = str(pid)
            rows.append(f)
    return pd.DataFrame(rows)


def fit_eval(tr, va, te, cols, tag, base_acc=None):
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", HistGradientBoostingClassifier(**PARAMS))])
    p.fit(tr[cols].astype(float).values, tr.label.values)
    pv = p.predict_proba(va[cols].astype(float).values)[:, 1]
    yv = va.label.values
    best = (0.5, -1e9)
    for thr in np.arange(0.20, 0.85, 0.005):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0: continue
        s = -(max(0.90 - precision_score(yv, pred, zero_division=0), 0) +
              max(0.90 - recall_score(yv, pred, zero_division=0), 0))
        if s > best[1]: best = (float(thr), s)
    thr = best[0]
    pt = p.predict_proba(te[cols].astype(float).values)[:, 1]
    yt = te.label.values
    yp = (pt >= thr).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    note = f"  ({accuracy_score(yt,yp)-base_acc:+.3f} vs fusion)" if base_acc else ""
    print(f"  {tag:<34} acc={accuracy_score(yt,yp):.3f}  prec={precision_score(yt,yp,zero_division=0):.3f}  "
          f"rec={recall_score(yt,yp,zero_division=0):.3f}  FP={cm[0,1]}  FN={cm[1,0]} (thr={thr:.2f}){note}")
    return accuracy_score(yt, yp), int(cm[0,1]), int(cm[1,0])


def main():
    # ---- 1. side maps ----
    aa = pd.read_parquet(AA); aa["play_id"] = aa.play_id.astype(str)
    tr_side = {r.play_id: ("LEFT" if r.side_is_left == 1 else "RIGHT") for r in aa.itertuples()}
    tg = sorted(aa.game_id.unique())
    fresh = SF.fresh_game_ids()
    amap = json.loads(Path("/tmp/plays_meta.json").read_text()) if Path("/tmp/plays_meta.json").exists() else {}
    SF.pull_artifacts(fresh)
    full_amap = SF.angle_map()  # falls back to gt_fresh
    fr_side = {pid: full_amap.get(pid, "LEFT") for pid in full_amap}

    # ---- PHASE A: derive per-game offsets ----
    print("=== PHASE A: per-game FAR<->NEAR offsets (median(f_near - f_far) across makes) ===")
    offsets: dict[str, dict] = {}
    for tag, tdir, games, sides in [("TRAIN", TRAIN_TRACKS, tg, tr_side),
                                     ("FRESH", FRESH_TRACKS, fresh, fr_side)]:
        for gid in games:
            pq = tdir / f"{gid}.parquet"
            if not pq.exists(): continue
            df = pd.read_parquet(pq)
            o = derive_offsets(df, sides)
            offsets[gid] = o
            sL = f"L={o['LEFT']:+d}" if o["LEFT"] is not None else "L=  -"
            sR = f"R={o['RIGHT']:+d}" if o["RIGHT"] is not None else "R=  -"
            stdL = (f" σ={o['std_LEFT']:.1f}" if o['std_LEFT'] else "")
            stdR = (f" σ={o['std_RIGHT']:.1f}" if o['std_RIGHT'] else "")
            print(f"  [{tag}] {gid[:8]}  {sL} ({o['n_LEFT']}{stdL})   {sR} ({o['n_RIGHT']}{stdR})")
    # show distribution
    rvals = [o["RIGHT"] for o in offsets.values() if o.get("RIGHT") is not None]
    lvals = [o["LEFT"] for o in offsets.values() if o.get("LEFT") is not None]
    print(f"  summary: RIGHT offsets median={int(np.median(rvals))}  range=[{min(rvals)},{max(rvals)}]")
    print(f"           LEFT  offsets median={int(np.median(lvals))}  range=[{min(lvals)},{max(lvals)}]")

    # ---- PHASE B: build SA features + retrain + eval on fresh ----
    print("\n=== PHASE B: sync-aware features in fusion ===")
    sa_tr = build_sa(TRAIN_TRACKS, tr_side, tg, offsets)
    sa_fr = build_sa(FRESH_TRACKS, fr_side, fresh, offsets)
    print(f"[sa] train rows={len(sa_tr)}  fresh rows={len(sa_fr)}")

    # quick separation check on fresh: disagree feature, make vs miss
    pred = pd.read_parquet(ROOT / "data" / "p3_fresh_predictions.parquet")[["game_id","play_id","label"]]
    pred["play_id"] = pred.play_id.astype(str)
    j = pred.merge(sa_fr, on=["game_id","play_id"], how="left")
    for lab, name in [(1,"MAKE"),(0,"MISS")]:
        s = j[j.label==lab]
        rate = s["sa_disagree_far_to_near"].mean()
        print(f"  fresh {name}: sa_disagree_far_to_near rate={rate:.2f}  (high = depth illusion signal)")

    fresh_amap = SF.angle_map()
    faa = SF.apply_angle_aware(SF.build_fresh_v8far(fresh), fresh_amap)
    faa["play_id"] = faa.play_id.astype(str); faa["split"] = "fresh"
    aa = aa.merge(sa_tr, on=["game_id","play_id"], how="left")
    faa = faa.merge(sa_fr, on=["game_id","play_id"], how="left")

    fus = [c for c in P3._feature_cols(pd.read_parquet(AA)) if c in aa.columns]
    fus_sa = fus + SA_KEYS
    for c in SA_KEYS:
        if c not in faa.columns: faa[c] = np.nan
    tr, va = aa[aa.split == "train"], aa[aa.split == "val"]
    print(f"train={len(tr)} val={len(va)} fresh={len(faa)} (makes={int(faa.label.sum())})\n")
    print("=== OUT-OF-SAMPLE on 5 fresh games (ref=fusion 0.951) ===")
    a0, fp0, fn0 = fit_eval(tr, va, faa, fus, "base fusion (210)")
    a1, fp1, fn1 = fit_eval(tr, va, faa, fus_sa, "fusion + sync-aware (210+sa)", base_acc=a0)
    print(f"\nΔ  acc {a0:.3f}->{a1:.3f} ({a1-a0:+.3f})   FP {fp0}->{fp1} ({fp1-fp0:+d})   FN {fn0}->{fn1} ({fn1-fn0:+d})")


if __name__ == "__main__":
    main()
