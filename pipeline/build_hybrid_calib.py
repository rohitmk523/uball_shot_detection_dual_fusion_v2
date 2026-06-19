#!/usr/bin/env python3
"""Build hybrid calibration: AUTO FR (auto-detected paint corners) + SAM3 NR
(existing, known-good).

Why hybrid? Auto-calibration on NR is underconstrained (only 5 landmarks vs
SAM3's 10+) and regressed game 4692eb2b from 74.5% to 50.0% decided in the
2026-06-08 session. But AUTO FR has lower reproj (10-12 px vs SAM3's 22 px)
and better floor cross-check (5 cm vs 21 cm). The hybrid keeps the proven NR
pose for rim accuracy and tests whether the FR improvement carries through.

Outputs: ``calibration_june_<id>_hybrid.json`` shaped like the SAM3 file so
the CALIB_JUNE_SAM3 loader in triangulate_shot.py reads it without changes.

Usage:
  python pipeline/build_hybrid_calib.py --all
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ("4692eb2b", "72c08cb7", "e74164e6", "454da9cf")


def hybrid(gid: str) -> Path:
    auto_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_auto.json"
    sam3_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_sam3.json"
    if not auto_path.exists():
        raise FileNotFoundError(f"missing AUTO json: {auto_path.name}")
    if not sam3_path.exists():
        raise FileNotFoundError(f"missing SAM3 json: {sam3_path.name}")

    auto = json.loads(auto_path.read_text())
    sam3 = json.loads(sam3_path.read_text())

    # Pull FR from AUTO, NR from SAM3, into the SAM3 schema so CALIB_JUNE_SAM3
    # reads it transparently
    fr_auto = auto['FR']
    out = dict(sam3)  # copy SAM3 schema (fov, hoop_bbox, sam3_ellipse, etc.)
    out['FR'] = dict(
        K=fr_auto['K'], rvec=fr_auto['rvec'], tvec=fr_auto['tvec'],
        cam_cm=sam3['FR'].get('cam_cm'),
        reproj_mean=fr_auto['reproj_mean'],
    )
    # NR stays exactly as SAM3
    # Mirror top-level metric fields the loader prints
    out['cross_check_mean_cm'] = sam3.get('cross_check_mean_cm', float('nan'))
    out['rim_center_cross_check_cm'] = sam3.get('rim_center_cross_check_cm',
                                                 float('nan'))
    out['method'] = "hybrid: AUTO FR (paint corners) + SAM3 NR"
    out['source_auto_json'] = auto_path.name
    out['source_sam3_json'] = sam3_path.name
    out['auto_FR_reproj'] = fr_auto['reproj_mean']
    out['sam3_NR_reproj'] = sam3['NR']['reproj_mean']

    out_path = ROOT / f"data/client_report/triangulation_test/calibration_june_{gid}_hybrid.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  {gid}: AUTO FR reproj={fr_auto['reproj_mean']:.1f}px  "
          f"SAM3 NR reproj={sam3['NR']['reproj_mean']:.1f}px  "
          f"-> {out_path.name}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.game_id and not args.all:
        print("specify --game-id or --all"); return 1
    target = GAMES if args.all else (args.game_id,)
    for gid in target:
        hybrid(gid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
