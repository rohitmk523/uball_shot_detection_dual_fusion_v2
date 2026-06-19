#!/usr/bin/env python3
"""Per-camera ensemble layer for game-2 (dc5f199e).

Cameras identical to game-1 -> reuse v4 calibration + same hoop bboxes
+ same single_camera_verdict / ensemble_vote logic.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from per_camera_verdict import (   # noqa: E402
    detect_per_frame, single_camera_verdict, ensemble_vote,
    FR_WEIGHTS, NR_WEIGHTS,
    FR_HX1, FR_HY1, FR_HX2, FR_HY2,
    NR_HX1, NR_HY1, NR_HX2, NR_HY2,
    MAKE_LABELS, MISS_LABELS,
)

G2 = ROOT / "data/client_report/triangulation_test/game2_dc5f199e"
CLIPS = G2 / "clips"
MAIN_RESULTS = G2 / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-sep shot names to run")
    ap.add_argument("--out", default=str(G2 / "ensemble_results.json"))
    args = ap.parse_args()

    manifest = json.loads((G2 / "shots_usable.json").read_text())
    gt_map = {s["name"]: s["gt"] for s in manifest}

    print("[setup] loading YOLO models ...")
    fr_model = YOLO(str(FR_WEIGHTS))
    nr_model = YOLO(str(NR_WEIGHTS))

    sel = set(args.only.split(",")) if args.only else None
    rows: list[dict] = []

    def cat(g: str, v: str) -> str:
        if v.startswith("UNDECIDED"): return "UND"
        if g in MAKE_LABELS and v.startswith("MAKE"): return "TP"
        if g in MISS_LABELS and v.startswith("MISS"): return "TN"
        if g in MISS_LABELS and v.startswith("MAKE"): return "FP"
        if g in MAKE_LABELS and v.startswith("MISS"): return "FN"
        return "?"

    n_done = 0
    for s in manifest:
        name = s["name"]
        if sel and name not in sel: continue
        fr_clip = CLIPS / f"{name}_FR.mp4"
        nr_clip = CLIPS / f"{name}_NR.mp4"
        if not fr_clip.exists() or not nr_clip.exists():
            print(f"[skip] {name}: missing clip"); continue

        main_path = MAIN_RESULTS / f"{name}.json"
        tri_v = "UNDECIDED"
        if main_path.exists():
            tri_v = json.loads(main_path.read_text()).get("verdict", "UNDECIDED")

        fr_dets = detect_per_frame(fr_model, fr_clip, conf=0.10)
        nr_dets = detect_per_frame(nr_model, nr_clip, conf=0.10)
        fr_v, fr_text, fr_info = single_camera_verdict(
            fr_dets, "FR", FR_HX1, FR_HY1, FR_HX2, FR_HY2, rebound_px=20.0)
        nr_v, nr_text, nr_info = single_camera_verdict(
            nr_dets, "NR", NR_HX1, NR_HY1, NR_HX2, NR_HY2, rebound_px=100.0)

        final_v, reason = ensemble_vote(tri_v, fr_v, nr_v)
        gt = gt_map[name]
        before = cat(gt, tri_v); after = cat(gt, final_v)

        rows.append(dict(name=name, gt=gt,
                         tri=tri_v, fr=fr_v, nr=nr_v,
                         fr_info=fr_info, nr_info=nr_info,
                         final=final_v, reason=reason,
                         before=before, after=after))
        n_done += 1
        change = f"  [{before}→{after}]" if before != after else ""
        print(f"  [{n_done:>3}] {name:18s} GT={gt:18s} "
              f"tri={tri_v[:30]:30s} FR={fr_v:10s} NR={nr_v:10s} "
              f"→ final={final_v[:25]:25s}{change}")

    Path(args.out).write_text(json.dumps(rows, indent=2))

    roll: dict[str, int] = defaultdict(int)
    for r in rows: roll[r["after"]] += 1
    n = sum(roll.values()); dec = roll["TP"]+roll["TN"]+roll["FP"]+roll["FN"]
    acc = 100*(roll["TP"]+roll["TN"])/dec if dec else 0
    print(f"\n=== GAME-2 ENSEMBLE FINAL ===")
    print(f"  TP={roll['TP']} TN={roll['TN']} FP={roll['FP']} "
          f"FN={roll['FN']} UND={roll['UND']}  acc={acc:.1f}%  "
          f"({dec}/{n} decided)")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
