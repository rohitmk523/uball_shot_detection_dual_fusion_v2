#!/usr/bin/env python3
"""Extract per-shot features from L1 results + L3 ensemble cache + GT.

Produces a CSV with one row per shot for the end-to-end multi-camera
MAKE/MISS classifier. Across the 185 labeled shots (G1=88, G2=22, G3=75)
this gives the training set.

Features:
  - tri-based: apex_r, apex_z, cross_r, z_min, has_bounce, has_rim_out,
               has_smooth_descent, has_gap_stop, n_samples
  - FR per-camera: n_deep, rebound_px, strength (0=UND, 1=weak, 2=strong)
  - NR per-camera: same
  - tri verdict bool: tri_make, tri_miss, tri_und
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

MAKE = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}

GAMES = [
    ("G1", ROOT / "data/client_report/triangulation_test/full_game/results",
           ROOT / "data/client_report/triangulation_test/full_game/shots_88.json",
           ROOT / "data/client_report/triangulation_test/full_game/ensemble_88.json"),
    ("G2", ROOT / "data/client_report/triangulation_test/game2_dc5f199e/results",
           ROOT / "data/client_report/triangulation_test/game2_dc5f199e/shots_usable.json",
           ROOT / "data/client_report/triangulation_test/game2_dc5f199e/ensemble_results.json"),
    ("G3", ROOT / "data/client_report/triangulation_test/game3_3398befc/results",
           ROOT / "data/client_report/triangulation_test/game3_3398befc/shots_right.json",
           ROOT / "data/client_report/triangulation_test/game3_3398befc/ensemble_results.json"),
]


def parse_num(s: str, pat: str) -> float | None:
    m = re.search(pat, s)
    return float(m.group(1)) if m else None


def extract(d: dict) -> dict:
    """From an L1 result.json compute trajectory features."""
    samples = d.get("samples", [])
    n_samples = len(samples)
    if n_samples < 3:
        return dict(n_samples=n_samples, apex_r=-1, apex_z=-1, cross_r=-1,
                    z_min=-1, bounce_cm=-1)
    zs = np.array([s["X_cm"][2] for s in samples])
    apex_i = int(np.argmax(zs))
    apex_z = float(zs[apex_i])
    apex_x, apex_y, _ = samples[apex_i]["X_cm"]
    apex_r = float(np.hypot(apex_x - 2008.7, apex_y - 713.2))
    z_min = float(zs[apex_i:].min()) if apex_i < len(zs)-1 else apex_z
    # parse verdict text for cross_r and bounce
    v = d.get("verdict", "")
    cross_r = parse_num(v, r"r=(\d+)cm") or -1
    bounce = parse_num(v, r"bounce=(\d+)cm") or 0
    return dict(n_samples=n_samples, apex_r=apex_r, apex_z=apex_z,
                cross_r=cross_r, z_min=z_min, bounce_cm=bounce,
                tri_verdict=v[:30])


def strength_n(v: str) -> int:
    if v == "UND": return 0
    if v.endswith("-weak"): return 1
    return 2


def main() -> int:
    rows = []
    for game_label, results_dir, manifest_p, ens_p in GAMES:
        manifest = json.loads(Path(manifest_p).read_text())
        gt_map = {s["name"]: s["gt"] for s in manifest}
        ens = {r["name"]: r for r in json.loads(Path(ens_p).read_text())}
        for s in manifest:
            name = s["name"]
            rp = Path(results_dir) / f"{name}.json"
            if not rp.exists(): continue
            try:
                d = json.loads(rp.read_text())
            except Exception:
                continue
            if not isinstance(d, dict) or "name" not in d: continue
            feats = extract(d)
            e = ens.get(name, {})
            fi = e.get("fr_info", {}) or {}
            ni = e.get("nr_info", {}) or {}
            tri = e.get("tri", "")
            fr_v = e.get("fr", "UND")
            nr_v = e.get("nr", "UND")
            row = dict(
                name=name, game=game_label, gt=gt_map[name],
                y=1 if gt_map[name] in MAKE else 0,
                **feats,
                tri_make=int(tri.startswith("MAKE")),
                tri_miss=int(tri.startswith("MISS")),
                tri_und=int(tri.startswith("UNDECIDED")),
                fr_strength=strength_n(fr_v),
                fr_is_make=int(fr_v.startswith("MAKE")),
                fr_is_miss=int(fr_v.startswith("MISS")),
                fr_n_deep=fi.get("n_deep", 0) or 0,
                fr_rebound_px=fi.get("rebound_px", 0) or 0,
                fr_max_cy=fi.get("max_cy", -1) or -1,
                nr_strength=strength_n(nr_v),
                nr_is_make=int(nr_v.startswith("MAKE")),
                nr_is_miss=int(nr_v.startswith("MISS")),
                nr_n_deep=ni.get("n_deep", 0) or 0,
                nr_rebound_px=ni.get("rebound_px", 0) or 0,
                nr_max_cy=ni.get("max_cy", -1) or -1,
                tri_has_rim_out=int("RIM-OUT" in tri),
                tri_has_rim_bounce=int("rim-bounce" in tri),
                tri_has_gap_stop=int("gap-stop" in tri),
                tri_has_smooth_descent=int("smooth descent" in tri),
                tri_has_pass_through=int("passed through" in tri or "rattled in" in tri),
                tri_has_clean_clean=int("CLEAN: " in tri),
                tri_has_no_clear=int("no clear make signal" in tri),
            )
            rows.append(row)

    import csv
    out = ROOT / "data/client_report/triangulation_test/shot_features.csv"
    if rows:
        with open(out, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    print(f"wrote {out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
