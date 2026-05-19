#!/usr/bin/env python3
"""
P2 — per-shot make/miss feature dataset.

For every (game_id, play_id) we build ONE feature row from the buffered
GT window of per-frame ball/rim tracks across the 4 camera angles
(FL, FR, NL, NR). Features are physically meaningful and interpretable:
they describe the geometry of the ball relative to the (near-static) rim
over time, per angle, then fuse the 4 angles.

Coordinates: (x, y) is the box top-left, (w, h) the box size, pixels in a
1920x1080 frame. Ball/rim conf is NaN when the object was not detected
that frame.

Output (idempotent):
  data/p2_features.parquet   one row per shot
  data/P2_FEATURES.md        feature dictionary + rationale + coverage

Local CPU only. No AWS, no GPU. Reads cached parquet from
/tmp/p1tracks/<game_id>.parquet (P1 artifacts).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ANGLES,
    SHOT_MAKE_CLASSES,
    SHOT_MISS_CLASSES,
    V1_ALL_SHOT_CLASSES,
    eprint,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKS_CACHE = Path("/tmp/p1tracks")
OUT_PARQUET = REPO_ROOT / "data" / "p2_features.parquet"
OUT_DOC = REPO_ROOT / "data" / "P2_FEATURES.md"

# A "through hoop" event requires the ball center to be horizontally within
# this many rim-widths of the rim center while crossing the rim plane.
THROUGH_HOOP_X_TOL_RW = 1.0
# Distance (in rim-widths) under which we call the ball "at the rim".
NEAR_RIM_DIST_RW = 1.5

NEAR_ANGLES = ("NL", "NR")
FAR_ANGLES = ("FL", "FR")

# Per-angle features computed by _angle_features (suffixed with _<angle>).
_PER_ANGLE_KEYS = [
    "n_frames",
    "ball_frac",          # fraction of window frames with a ball detection
    "ball_conf_mean",
    "ball_conf_max",
    "rim_frac",           # fraction with a rim detection
    "rim_detected",       # 1 if rim ever detected (=> rim ref usable)
    "min_dist_rw",        # min ball-rim center dist / rim_width (lower=closer)
    "min_dist_frac_time", # when closest approach happened (0..1 in window)
    "enters_rim_box",     # ball center ever inside rim box (1/0)
    "frac_inside_rim",    # fraction of ball frames with center in rim box
    "through_hoop",       # clean above->below crossing within x-tol (1/0)
    "through_hoop_count", # # of such crossing frames
    "through_hoop_conf",  # mean ball conf on crossing frames
    "vy_sign_change_near_rim",  # ball vy flips sign while near rim (1/0)
    "min_dist_conf",      # ball conf at closest approach
    "approach_drop",      # how much dist drops from start->min (rim-widths)
    "post_min_rebound",   # dist increase after min (miss bounce-out proxy)
]

# Imputation defaults for per-angle features when an angle is unusable.
_ANGLE_IMPUTE = {
    "n_frames": 0.0,
    "ball_frac": 0.0,
    "ball_conf_mean": 0.0,
    "ball_conf_max": 0.0,
    "rim_frac": 0.0,
    "rim_detected": 0.0,
    "min_dist_rw": 10.0,            # "very far" sentinel
    "min_dist_frac_time": 0.5,
    "enters_rim_box": 0.0,
    "frac_inside_rim": 0.0,
    "through_hoop": 0.0,
    "through_hoop_count": 0.0,
    "through_hoop_conf": 0.0,
    "vy_sign_change_near_rim": 0.0,
    "min_dist_conf": 0.0,
    "approach_drop": 0.0,
    "post_min_rebound": 0.0,
}


def _robust_rim_ref(g: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Robust (median) rim box over the window. The rim is ~static, so the
    median over all detected frames is a stable reference. Returns None if
    the rim was never detected for this angle."""
    rim = g.dropna(subset=["rim_x", "rim_y", "rim_w", "rim_h"])
    if rim.empty:
        return None
    rx = float(np.median(rim["rim_x"]))
    ry = float(np.median(rim["rim_y"]))
    rw = float(np.median(rim["rim_w"]))
    rh = float(np.median(rim["rim_h"]))
    if rw <= 1.0 or rh <= 1.0:
        return None
    return {
        "rcx": rx + rw / 2.0,
        "rcy": ry + rh / 2.0,
        "rw": rw,
        "rh": rh,
        "rx0": rx,
        "ry0": ry,
        "rx1": rx + rw,
        "ry1": ry + rh,
    }


def _angle_features(g: pd.DataFrame) -> Dict[str, float]:
    """Compute interpretable ball-vs-rim features for one angle's window.

    g is already sorted by frame_idx and restricted to a single
    (play_id, angle). Always returns every _PER_ANGLE_KEYS key (imputed
    defaults when the signal is undefined) so the fused row is dense.
    """
    feats = dict(_ANGLE_IMPUTE)
    n = len(g)
    feats["n_frames"] = float(n)
    if n == 0:
        return feats

    rim = _robust_rim_ref(g)
    feats["rim_frac"] = float(g["rim_conf"].notna().mean())
    feats["rim_detected"] = 1.0 if rim is not None else 0.0

    ball = g.dropna(subset=["ball_x", "ball_y", "ball_w", "ball_h"]).copy()
    feats["ball_frac"] = float(len(ball)) / float(n)
    if len(ball):
        bc = g["ball_conf"].dropna()
        feats["ball_conf_mean"] = float(bc.mean()) if len(bc) else 0.0
        feats["ball_conf_max"] = float(bc.max()) if len(bc) else 0.0

    if rim is None or ball.empty:
        return feats

    rcx, rcy, rw = rim["rcx"], rim["rcy"], rim["rw"]
    bcx = ball["ball_x"].to_numpy() + ball["ball_w"].to_numpy() / 2.0
    bcy = ball["ball_y"].to_numpy() + ball["ball_h"].to_numpy() / 2.0
    bconf = ball["ball_conf"].fillna(0.0).to_numpy()
    fidx = ball["frame_idx"].to_numpy().astype(float)

    dist = np.hypot(bcx - rcx, bcy - rcy)
    dist_rw = dist / rw

    j = int(np.argmin(dist_rw))
    feats["min_dist_rw"] = float(dist_rw[j])
    feats["min_dist_conf"] = float(bconf[j])
    f0, f1 = fidx.min(), fidx.max()
    span = max(f1 - f0, 1.0)
    feats["min_dist_frac_time"] = float((fidx[j] - f0) / span)

    # Ball center inside the rim box.
    inside = (
        (bcx >= rim["rx0"]) & (bcx <= rim["rx1"])
        & (bcy >= rim["ry0"]) & (bcy <= rim["ry1"])
    )
    feats["enters_rim_box"] = 1.0 if inside.any() else 0.0
    feats["frac_inside_rim"] = float(inside.mean())

    # "Through hoop": consecutive detected ball frames where the center
    # moves from at/above the rim plane (bcy <= rcy) to below it
    # (bcy > rcy) while staying horizontally within X_TOL rim-widths of
    # the rim center. A made shot drops vertically through the hoop.
    xtol = THROUGH_HOOP_X_TOL_RW * rw
    horiz_ok = np.abs(bcx - rcx) <= xtol
    above = bcy <= rcy
    th_count = 0
    th_confs: List[float] = []
    for k in range(1, len(bcy)):
        if above[k - 1] and (not above[k]) and horiz_ok[k - 1] and horiz_ok[k]:
            th_count += 1
            th_confs.append(0.5 * (bconf[k - 1] + bconf[k]))
    feats["through_hoop_count"] = float(th_count)
    feats["through_hoop"] = 1.0 if th_count > 0 else 0.0
    feats["through_hoop_conf"] = (
        float(np.mean(th_confs)) if th_confs else 0.0
    )

    # Vertical-velocity sign change while the ball is near the rim — a
    # bounce/rim interaction. Misses often bounce (vy flips up) near rim.
    near = dist_rw <= NEAR_RIM_DIST_RW
    if near.sum() >= 3:
        vy = np.diff(bcy)
        near_pairs = near[1:] & near[:-1]
        sign_flip = False
        prev = 0
        for k in range(len(vy)):
            if not near_pairs[k]:
                continue
            s = np.sign(vy[k])
            if s == 0:
                continue
            if prev != 0 and s != prev:
                sign_flip = True
                break
            prev = s
        feats["vy_sign_change_near_rim"] = 1.0 if sign_flip else 0.0

    # Approach dynamics: how far the ball travelled toward the rim, and
    # whether the distance rebounds after the closest approach (a strong
    # miss / bounce-out cue vs. a clean make that just disappears).
    feats["approach_drop"] = float(max(dist_rw[0] - dist_rw[j], 0.0))
    if j < len(dist_rw) - 1:
        feats["post_min_rebound"] = float(
            max(dist_rw[j + 1:].max() - dist_rw[j], 0.0)
        )
    return feats


def _fuse(per_angle: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Fuse the 4 per-angle feature dicts into one shot-level row.

    Strategy: emit every per-angle feature flat (so the model can learn
    angle-specific weights), plus physically motivated cross-angle
    aggregates (max/mean of made-like signals, vote agreement, count of
    angles with a clean through-hoop, near vs far symmetry, confidence-
    weighted closest-approach distance)."""
    row: Dict[str, float] = {}
    for ang in ANGLES:
        fa = per_angle.get(ang, dict(_ANGLE_IMPUTE))
        for k in _PER_ANGLE_KEYS:
            row[f"{k}_{ang}"] = float(fa.get(k, _ANGLE_IMPUTE[k]))
        row[f"angle_usable_{ang}"] = (
            1.0 if (fa.get("rim_detected", 0.0) >= 1.0
                    and fa.get("ball_frac", 0.0) > 0.0) else 0.0
        )

    usable = [a for a in ANGLES if row[f"angle_usable_{a}"] >= 1.0]
    row["n_usable_angles"] = float(len(usable))

    def vals(key: str, angs=ANGLES) -> np.ndarray:
        return np.array([row[f"{key}_{a}"] for a in angs], dtype=float)

    def uvals(key: str, angs=ANGLES) -> np.ndarray:
        v = [row[f"{key}_{a}"] for a in angs if a in usable]
        return np.array(v, dtype=float)

    # Made-like signals: a make is best evidenced by a small min distance,
    # a clean through-hoop event, and the ball entering the rim box.
    md = uvals("min_dist_rw")
    row["min_dist_rw_best"] = float(md.min()) if md.size else 10.0
    row["min_dist_rw_mean"] = float(md.mean()) if md.size else 10.0
    row["through_hoop_any"] = float((vals("through_hoop") >= 1.0).any())
    row["through_hoop_votes"] = float(
        sum(row[f"through_hoop_{a}"] for a in usable)
    )
    row["through_hoop_count_max"] = float(vals("through_hoop_count").max())
    thc = uvals("through_hoop_conf")
    row["through_hoop_conf_max"] = float(thc.max()) if thc.size else 0.0
    row["enters_rim_votes"] = float(
        sum(row[f"enters_rim_box_{a}"] for a in usable)
    )
    fir = uvals("frac_inside_rim")
    row["frac_inside_rim_max"] = float(fir.max()) if fir.size else 0.0

    # Miss-like signals: bounce-out rebound and vy sign change near rim.
    pmr = uvals("post_min_rebound")
    row["post_min_rebound_max"] = float(pmr.max()) if pmr.size else 0.0
    row["vy_sign_change_votes"] = float(
        sum(row[f"vy_sign_change_near_rim_{a}"] for a in usable)
    )

    # Confidence-weighted closest approach: weight each angle's min-dist by
    # the ball conf at that closest frame; robust when some angles are noisy.
    if usable:
        w = np.array(
            [max(row[f"min_dist_conf_{a}"], 1e-3) for a in usable]
        )
        d = np.array([row[f"min_dist_rw_{a}"] for a in usable])
        row["min_dist_rw_confw"] = float(np.average(d, weights=w))
    else:
        row["min_dist_rw_confw"] = 10.0

    # Near vs far symmetry — near cameras (NL/NR) usually see the rim
    # plane better; keep both groups so the model can weight them.
    for grp, angs in (("near", NEAR_ANGLES), ("far", FAR_ANGLES)):
        gm = [row[f"min_dist_rw_{a}"] for a in angs
              if row[f"angle_usable_{a}"] >= 1.0]
        row[f"min_dist_rw_{grp}_best"] = (
            float(min(gm)) if gm else 10.0
        )
        row[f"through_hoop_{grp}_any"] = float(
            any(row[f"through_hoop_{a}"] >= 1.0 for a in angs
                if row[f"angle_usable_{a}"] >= 1.0)
        )

    # Overall detection quality (imputation-awareness for the model).
    row["ball_frac_mean"] = float(vals("ball_frac").mean())
    row["ball_conf_max_overall"] = float(vals("ball_conf_max").max())
    row["any_angle_usable"] = 1.0 if usable else 0.0
    return row


def _shot_rows(game_id: str, df: pd.DataFrame) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for play_id, gp in df.groupby("play_id", sort=True):
        classification = str(gp["classification"].iloc[0])
        if classification in SHOT_MAKE_CLASSES:
            label = 1
        elif classification in SHOT_MISS_CLASSES:
            label = 0
        else:
            continue  # not a make/miss shot class
        per_angle: Dict[str, Dict[str, float]] = {}
        for ang in ANGLES:
            ga = gp[gp["angle"] == ang].sort_values("frame_idx")
            per_angle[ang] = _angle_features(ga)
        fused = _fuse(per_angle)
        fused.update(
            game_id=game_id,
            play_id=str(play_id),
            classification=classification,
            label=label,
            v1_in_scope=classification in V1_ALL_SHOT_CLASSES,
        )
        rows.append(fused)
    return rows


def build() -> pd.DataFrame:
    games = {g.game_id: g for g in load_manifest()}
    all_rows: List[Dict[str, float]] = []
    for gid, game in games.items():
        pq = TRACKS_CACHE / f"{gid}.parquet"
        if not pq.exists():
            raise FileNotFoundError(
                f"missing cached tracks for {gid}: {pq} "
                f"(run the P1 download first)"
            )
        df = pd.read_parquet(pq)
        rows = _shot_rows(gid, df)
        for r in rows:
            r["split"] = game.split
        all_rows.extend(rows)
        eprint(f"[p2] {gid} split={game.split} shots={len(rows)}")

    out = pd.DataFrame(all_rows)
    lead = ["game_id", "play_id", "classification", "label",
            "v1_in_scope", "split"]
    feat_cols = [c for c in out.columns if c not in lead]
    out = out[lead + sorted(feat_cols)]
    out = out.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    return out


def _coverage(df: pd.DataFrame) -> Dict[str, object]:
    th_cols = [f"through_hoop_{a}" for a in ANGLES]
    has_th = (df[th_cols].sum(axis=1) > 0)
    return {
        "n_shots": int(len(df)),
        "n_makes": int((df["label"] == 1).sum()),
        "n_misses": int((df["label"] == 0).sum()),
        "n_v1_in_scope": int(df["v1_in_scope"].sum()),
        "n_4pt_make": int((df["classification"] == "4PT_MAKE").sum()),
        "shots_with_through_hoop_ge1_angle": int(has_th.sum()),
        "frac_with_through_hoop": round(float(has_th.mean()), 4),
        "mean_usable_angles": round(
            float(df["n_usable_angles"].mean()), 3),
        "shots_0_usable_angles": int((df["n_usable_angles"] == 0).sum()),
        "by_split": {
            s: int((df["split"] == s).sum())
            for s in ("train", "val", "test")
        },
        "through_hoop_make_rate": round(float(
            df.loc[has_th, "label"].mean()), 4) if has_th.any() else None,
        "no_through_hoop_make_rate": round(float(
            df.loc[~has_th, "label"].mean()), 4) if (~has_th).any() else None,
    }


def _write_doc(df: pd.DataFrame, cov: Dict[str, object]) -> None:
    feat_cols = [c for c in df.columns if c not in (
        "game_id", "play_id", "classification", "label",
        "v1_in_scope", "split")]
    lines: List[str] = []
    lines.append("# P2 Feature Dictionary\n")
    lines.append(
        "One row per shot `(game_id, play_id)`. Features describe the "
        "ball's geometry relative to the (near-static) rim over the "
        "buffered GT window, computed per camera angle (FL/FR/NL/NR) and "
        "fused. All coordinates are pixels in 1920x1080; (x,y)=box "
        "top-left. Distances are normalised by the robust median rim "
        "width so they are scale/zoom invariant.\n")
    lines.append("## Coverage\n")
    for k, v in cov.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Per-angle features (suffix `_FL/_FR/_NL/_NR`)\n")
    rationale = {
        "n_frames": "window length for this angle (context).",
        "ball_frac": "fraction of frames the ball was detected — "
                     "detection quality / imputation awareness.",
        "ball_conf_mean": "mean ball detector confidence.",
        "ball_conf_max": "peak ball confidence (clear sighting).",
        "rim_frac": "fraction of frames the rim was detected.",
        "rim_detected": "1 if a robust rim reference exists for this angle.",
        "min_dist_rw": "closest ball-center to rim-center distance over "
                       "the window, in rim-widths. Low => ball reached the "
                       "hoop (necessary for a make).",
        "min_dist_frac_time": "when (0..1 through window) closest approach "
                              "occurred.",
        "enters_rim_box": "1 if the ball center was ever inside the rim "
                          "bounding box.",
        "frac_inside_rim": "fraction of ball frames with center in rim box.",
        "through_hoop": "1 if a clean above->below rim-plane crossing "
                        "occurred within ~1 rim-width horizontally — the "
                        "core make cue.",
        "through_hoop_count": "number of such crossing frames.",
        "through_hoop_conf": "mean ball confidence on crossing frames.",
        "vy_sign_change_near_rim": "1 if the ball's vertical velocity "
                                   "flipped sign while near the rim — a "
                                   "rim bounce (miss cue).",
        "min_dist_conf": "ball confidence at the closest-approach frame.",
        "approach_drop": "how much the rim distance fell from window start "
                         "to the closest approach (rim-widths).",
        "post_min_rebound": "how much distance increased after the closest "
                            "approach — bounce-out (miss cue).",
        "angle_usable": "1 if rim ref exists AND ball seen at least once.",
    }
    for base, why in rationale.items():
        lines.append(f"- `{base}_*`: {why}")
    lines.append("")
    lines.append("## Fused cross-angle features\n")
    fused_doc = {
        "n_usable_angles": "# angles with rim ref + a ball detection.",
        "min_dist_rw_best": "min over usable angles of min_dist_rw "
                            "(strongest make-distance evidence).",
        "min_dist_rw_mean": "mean over usable angles.",
        "min_dist_rw_confw": "ball-conf-weighted mean of per-angle min "
                             "distance (robust fusion).",
        "through_hoop_any": "any angle saw a clean through-hoop.",
        "through_hoop_votes": "# usable angles with a through-hoop event.",
        "through_hoop_count_max": "max crossing-frame count across angles.",
        "through_hoop_conf_max": "best through-hoop confidence.",
        "enters_rim_votes": "# usable angles where ball entered rim box.",
        "frac_inside_rim_max": "max frac_inside_rim across angles.",
        "post_min_rebound_max": "max bounce-out rebound across angles "
                                "(miss cue).",
        "vy_sign_change_votes": "# angles with a near-rim vy sign flip.",
        "min_dist_rw_near_best": "best min-dist among near cams (NL/NR).",
        "min_dist_rw_far_best": "best min-dist among far cams (FL/FR).",
        "through_hoop_near_any": "through-hoop seen by a near cam.",
        "through_hoop_far_any": "through-hoop seen by a far cam.",
        "ball_frac_mean": "mean ball-detection fraction across angles.",
        "ball_conf_max_overall": "global peak ball confidence.",
        "any_angle_usable": "1 if at least one angle is usable.",
    }
    for k, why in fused_doc.items():
        lines.append(f"- `{k}`: {why}")
    lines.append("")
    lines.append(f"Total feature columns: **{len(feat_cols)}**.\n")
    lines.append(
        "### Sanity: through-hoop vs label\n"
        f"- Shots with a through-hoop on >=1 angle: make rate = "
        f"{cov['through_hoop_make_rate']} "
        f"(n={cov['shots_with_through_hoop_ge1_angle']})\n"
        f"- Shots with NO through-hoop: make rate = "
        f"{cov['no_through_hoop_make_rate']}\n")
    OUT_DOC.write_text("\n".join(lines))


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    if df.empty:
        raise RuntimeError("no shots produced — check input tracks")
    df.to_parquet(OUT_PARQUET, index=False)
    cov = _coverage(df)
    _write_doc(df, cov)
    eprint(f"[p2] wrote {OUT_PARQUET} ({len(df)} shots, "
           f"{df.shape[1]} cols)")
    eprint(f"[p2] wrote {OUT_DOC}")
    print("P2 COVERAGE:")
    for k, v in cov.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
