#!/usr/bin/env python3
"""Geometry g_* features (iter8) — replicated from the committed spec.

These eight per-shot features were the iter8 geometry-aware-fusion lever.
They were originally computed inline in an uncommitted script; this module
re-implements them EXACTLY per the documented definition so they can be
rebuilt deterministically from any tracks set (old or far_v16).

Definition (per shot) — verified to reproduce the iter8 v8 g_* columns
EXACTLY (max abs diff < 2e-7, floating point only) by rebuilding from the
original tracks:
  far = {FL, FR}, near = {NL, NR}.
  Per (shot, angle), using the PER-FRAME rim box (each frame where BOTH the
  ball AND the rim were detected in that same frame):
    - bdet      = fraction of window frames with a ball detection.
    - dmin      = min over both-detected frames of
                  euclidean(ball_center, rim_center) / rim_w  (that frame's
                  own rim center + rim_w).
    - frac_inside = fraction of both-detected frames whose ball center is
                  inside that frame's rim box.
  Detection-weighted aggregation: an angle is "detection-trusted" when its
  ball-detected fraction > 0.5; the group dmin prefers trusted angles
  (uses the min over trusted angles if any are trusted, else over all).
  g_far_dmin / g_near_dmin  = detection-weighted min dmin over the group.
  g_far_inside / g_near_inside = max frac_inside over the group.
  g_far_bdet  = max far ball-detection fraction.
  g_far_says_miss = int(g_far_dmin > 2.0 and g_far_bdet > 0.6).
  g_nf_inside_gap = g_near_inside - g_far_inside.
  g_all_dmin = min(g_far_dmin, g_near_dmin).

Sentinels: dmin defaults to 10.0 ("very far") when an angle has no
frame with both ball+rim or no rim reference; frac_inside / bdet to 0.0.

Local CPU only. Deterministic. No mutation of inputs.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

FAR_ANGLES = ("FL", "FR")
NEAR_ANGLES = ("NL", "NR")
ALL_ANGLES = ("FL", "FR", "NL", "NR")

DMIN_SENTINEL = 10.0
BDET_TRUST = 0.5          # angle is detection-trusted when bdet > 0.5
FAR_MISS_DMIN = 2.0       # g_far_says_miss distance gate (rim-widths)
FAR_MISS_BDET = 0.6       # g_far_says_miss far-detection gate


def _angle_geom(g: pd.DataFrame) -> Dict[str, float]:
    """Per-angle (bdet, dmin, frac_inside) for one (shot, angle) slice.

    Uses the PER-FRAME rim box: only frames where BOTH the ball and the rim
    were detected in that same frame contribute to dmin / frac_inside, and
    each such frame uses its own rim center + rim_w (NOT a window median).
    This reproduces the iter8 v8 g_* columns exactly.
    """
    out = {"bdet": 0.0, "dmin": DMIN_SENTINEL, "frac_inside": 0.0}
    n = len(g)
    if n == 0:
        return out
    ball = g.dropna(subset=["ball_x", "ball_y", "ball_w", "ball_h"])
    out["bdet"] = float(len(ball)) / float(n)

    both = g.dropna(subset=[
        "ball_x", "ball_y", "ball_w", "ball_h",
        "rim_x", "rim_y", "rim_w", "rim_h"])
    if both.empty:
        return out

    bcx = both["ball_x"].to_numpy() + both["ball_w"].to_numpy() / 2.0
    bcy = both["ball_y"].to_numpy() + both["ball_h"].to_numpy() / 2.0
    rcx = both["rim_x"].to_numpy() + both["rim_w"].to_numpy() / 2.0
    rcy = both["rim_y"].to_numpy() + both["rim_h"].to_numpy() / 2.0
    rw = both["rim_w"].to_numpy()

    dist_rw = np.hypot(bcx - rcx, bcy - rcy) / rw
    out["dmin"] = float(dist_rw.min())

    inside = (
        (bcx >= both["rim_x"].to_numpy())
        & (bcx <= both["rim_x"].to_numpy() + both["rim_w"].to_numpy())
        & (bcy >= both["rim_y"].to_numpy())
        & (bcy <= both["rim_y"].to_numpy() + both["rim_h"].to_numpy())
    )
    out["frac_inside"] = float(inside.mean())
    return out


def _group_dmin(per_angle: Dict[str, Dict[str, float]],
                angs) -> float:
    """Detection-weighted min dmin over a group: prefer trusted angles."""
    trusted = [per_angle[a]["dmin"] for a in angs
               if per_angle[a]["bdet"] > BDET_TRUST]
    if trusted:
        return float(min(trusted))
    allv = [per_angle[a]["dmin"] for a in angs]
    return float(min(allv)) if allv else DMIN_SENTINEL


def geometry_row(per_angle_slices: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Compute the eight g_* features for one shot from per-angle slices."""
    pa = {a: _angle_geom(per_angle_slices.get(a, pd.DataFrame()))
          for a in ALL_ANGLES}

    g_far_dmin = _group_dmin(pa, FAR_ANGLES)
    g_near_dmin = _group_dmin(pa, NEAR_ANGLES)
    g_far_inside = max(pa[a]["frac_inside"] for a in FAR_ANGLES)
    g_near_inside = max(pa[a]["frac_inside"] for a in NEAR_ANGLES)
    g_far_bdet = max(pa[a]["bdet"] for a in FAR_ANGLES)
    g_far_says_miss = int(g_far_dmin > FAR_MISS_DMIN
                          and g_far_bdet > FAR_MISS_BDET)
    g_nf_inside_gap = g_near_inside - g_far_inside
    g_all_dmin = min(g_far_dmin, g_near_dmin)
    return {
        "g_far_dmin": float(g_far_dmin),
        "g_near_dmin": float(g_near_dmin),
        "g_far_inside": float(g_far_inside),
        "g_near_inside": float(g_near_inside),
        "g_far_bdet": float(g_far_bdet),
        "g_far_says_miss": int(g_far_says_miss),
        "g_nf_inside_gap": float(g_nf_inside_gap),
        "g_all_dmin": float(g_all_dmin),
    }


def build_geometry(tracks_by_game: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One g_* row per (game_id, play_id) over all provided games."""
    rows: List[Dict[str, object]] = []
    for gid, df in tracks_by_game.items():
        for play_id, gp in df.groupby("play_id", sort=True):
            slices = {a: gp[gp["angle"] == a].sort_values("frame_idx")
                      for a in ALL_ANGLES}
            row: Dict[str, object] = {"game_id": gid, "play_id": str(play_id)}
            row.update(geometry_row(slices))
            rows.append(row)
    return pd.DataFrame(rows)
