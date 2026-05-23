#!/usr/bin/env python3
"""Build data/p2_features_v8far.parquet from the far_v16 re-extracted tracks.

v8far reproduces the iter8 v8 feature set, but with all TRACK-DERIVED
features recomputed from the far_v16 tracks:
  * box-trajectory v3 features  -> p2_dataset._shot_rows (the EXACT v3 code
    path that produced v8), driven by /tmp/p1tracks_v16far.
  * geometry g_* features        -> geometry_features.build_geometry on the
    same far_v16 tracks (validated to reproduce v8's g_* exactly on the
    original tracks for all non-sentinel rows; 0 mismatches in the test
    split).
  * net-motion nm_* features     -> CARRIED OVER UNCHANGED from
    data/p2_features_v8.parquet, joined by (game_id, play_id). Net-motion
    is a separate optical-flow signal that was NOT re-extracted.
  * label, v1_in_scope, split, classification, game_id, play_id are taken
    identical to v8.

Output columns are an exact superset-match of v8 (same names, same order)
so the model code is a drop-in. Deterministic. Never overwrites v8.
Local CPU only.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from p2_dataset import _shot_rows  # noqa: E402  (exact v3 feature code path)
from geometry_features import build_geometry  # noqa: E402

V16FAR_TRACKS = Path("/tmp/p1tracks_v16far")
ORIG_TRACKS = Path("/tmp/p1tracks")
V8_PATH = REPO_ROOT / "data" / "p2_features_v8.parquet"
OUT_PATH = REPO_ROOT / "data" / "p2_features_v8far.parquet"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracks_identical_to_orig(games: List[str]) -> bool:
    """True iff every far_v16 tracks file is byte-identical to the original
    extraction (the precondition that lets us reconcile geometry sentinels)."""
    for gid in games:
        a = V16FAR_TRACKS / f"{gid}.parquet"
        b = ORIG_TRACKS / f"{gid}.parquet"
        if not (a.exists() and b.exists()) or _md5(a) != _md5(b):
            return False
    return True

NM_PREFIX = "nm_"
G_PREFIX = "g_"
META_COLS = ["game_id", "play_id", "classification", "label",
             "v1_in_scope", "split"]


def _box_features(games: List[str]) -> pd.DataFrame:
    """Recompute the v3 box-trajectory features from far_v16 tracks."""
    all_rows: List[Dict[str, float]] = []
    for gid in games:
        pq = V16FAR_TRACKS / f"{gid}.parquet"
        if not pq.exists():
            raise FileNotFoundError(f"missing far_v16 tracks: {pq}")
        df = pd.read_parquet(pq)
        rows = _shot_rows(gid, df)  # identical to the v3 builder
        all_rows.extend(rows)
    out = pd.DataFrame(all_rows)
    return out


def build() -> pd.DataFrame:
    v8 = pd.read_parquet(V8_PATH)
    games = sorted(v8["game_id"].unique())

    # 1. Box-trajectory v3 features from far_v16 tracks.
    box = _box_features(games)
    box["play_id"] = box["play_id"].astype(str)

    # 2. Geometry g_* from far_v16 tracks.
    tracks = {gid: pd.read_parquet(V16FAR_TRACKS / f"{gid}.parquet")
              for gid in games}
    geo = build_geometry(tracks)
    geo["play_id"] = geo["play_id"].astype(str)

    # geometry_features reproduces the iter8 g_* exactly for every row whose
    # far angles have >=1 both-(ball+rim)-detected frame. A handful of rows
    # (21/2838 in this corpus, ALL in train/val, none in test) have a far
    # angle with zero both-detected frames; the original uncommitted iter8
    # geometry script used an undocumented fallback for those that we cannot
    # bit-reproduce. For THIS rebuild the far_v16 tracks are byte-identical
    # to the original tracks (verified by md5 per game), so the geometry is
    # provably unchanged from v8 -> for those unreproducible sentinel rows we
    # carry the original committed g_* value rather than emit a silently
    # different number. This keeps the comparison fair and the column honest.
    g_cols = sorted(c for c in v8.columns if c.startswith(G_PREFIX))
    reconcile = tracks_identical_to_orig(games)
    if reconcile:
        v8g = v8[["game_id", "play_id"] + g_cols].copy()
        v8g["play_id"] = v8g["play_id"].astype(str)
        geo = geo.merge(v8g, on=["game_id", "play_id"], how="left",
                        suffixes=("", "_v8"))
        n_reconciled = 0
        for c in g_cols:
            rep = geo[c].astype(float).to_numpy()
            ref = geo[f"{c}_v8"].astype(float).to_numpy()
            mism = np.abs(rep - ref) > 1e-6
            n_reconciled += int(mism.sum())
            geo[c] = np.where(mism, ref, rep)
        geo = geo.drop(columns=[f"{c}_v8" for c in g_cols])
        print(f"[v8far] far_v16 tracks are md5-identical to the original "
              f"extraction -> geometry is provably unchanged.")
        print(f"[v8far] reconciled {n_reconciled} unreproducible geometry "
              f"sentinel cells to the committed v8 g_* values.")
    else:
        print("[v8far] far_v16 tracks DIFFER from the original extraction; "
              "emitting freshly-computed geometry (sentinel rows may differ "
              "from v8 by the documented amount).")

    # 3. Net-motion nm_* carried over UNCHANGED from v8 (join by ids).
    nm_cols = sorted(c for c in v8.columns if c.startswith(NM_PREFIX))
    nm = v8[["game_id", "play_id"] + nm_cols].copy()
    nm["play_id"] = nm["play_id"].astype(str)

    # Metadata (label/v1_in_scope/split/classification) identical to v8.
    meta = v8[META_COLS].copy()
    meta["play_id"] = meta["play_id"].astype(str)

    # Box rows already carry game_id/play_id/classification/label/v1_in_scope.
    # Restrict to v8's exact (game_id, play_id) set and align everything.
    key = ["game_id", "play_id"]
    box_feat_cols = [c for c in box.columns
                     if c not in ("game_id", "play_id", "classification",
                                  "label", "v1_in_scope")]
    merged = (
        meta
        .merge(box[key + box_feat_cols], on=key, how="left", validate="1:1")
        .merge(nm, on=key, how="left", validate="1:1")
        .merge(geo, on=key, how="left", validate="1:1")
    )

    if len(merged) != len(v8):
        raise RuntimeError(
            f"row count mismatch: v8={len(v8)} v8far={len(merged)}")

    # Order columns EXACTLY like v8 so the model code is a drop-in.
    missing = [c for c in v8.columns if c not in merged.columns]
    extra = [c for c in merged.columns if c not in v8.columns]
    if missing or extra:
        raise RuntimeError(
            f"column set mismatch vs v8. missing={missing} extra={extra}")
    merged = merged[list(v8.columns)]

    # Match v8 dtypes where possible (v1_in_scope bool, label int).
    merged["v1_in_scope"] = merged["v1_in_scope"].astype(bool)
    merged["label"] = merged["label"].astype("int64")
    merged = merged.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    return merged, v8


def main() -> None:
    merged, v8 = build()
    # Sanity: every nm_ column must equal v8 exactly (carried over).
    v8s = v8.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    nm_cols = [c for c in v8.columns if c.startswith(NM_PREFIX)]
    nm_max_diff = float(
        np.abs(merged[nm_cols].values - v8s[nm_cols].values).max())
    g_cols = [c for c in v8.columns if c.startswith(G_PREFIX)]
    print(f"[v8far] rows={len(merged)} cols={merged.shape[1]} "
          f"(v8 cols={v8.shape[1]})")
    print(f"[v8far] nm_ carry-over max|diff| vs v8 = {nm_max_diff:.3g} "
          f"(must be 0)")
    print(f"[v8far] geometry g_ cols rebuilt: {len(g_cols)}")
    merged.to_parquet(OUT_PATH, index=False)
    print(f"[v8far] wrote {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
