#!/usr/bin/env python3
"""Per-camera ensemble layer.

For each shot, run YOLO independently on FR and NR clips and compute a
single-camera MAKE / MISS / UND verdict from raw ball + hoop detections.

Then merge with the existing 3D triangulation verdict via voting:
  - Triangulation DECIDED -> trust it unless BOTH camera votes disagree
  - Triangulation UNDECIDED -> use the majority of FR + NR votes

Per-camera judgment is intentionally simple, mirroring what an annotator
sees on each angle: did the ball enter the rim region, and did it then
continue down (make) or bounce back up (miss)?
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
FG = ROOT / "data/client_report/triangulation_test/full_game"
CLIPS = FG / "clips"
MAIN_RESULTS = FG / "results"

FR_WEIGHTS = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Training_frameworks/Uball Far Angle/deliverables/far_v16_best.pt")
NR_WEIGHTS = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/Uball_dual_angle_shot_detection/weights/near_angle_weights/basketball_yolo11n3/weights/best.pt")

FPS = 29.97
PAD_BEFORE = 1.0

MAKE_LABELS = {"FREE_THROW_MAKE","FG_MAKE","3PT_MAKE","4PT_MAKE"}
MISS_LABELS = {"FREE_THROW_MISS","FG_MISS","3PT_MISS","4PT_MISS"}

# NR hoop bbox (from v4 calibration). NR sees rim near bottom of frame.
NR_HX1, NR_HY1, NR_HX2, NR_HY2 = 714.0, 826.0, 1138.0, 1080.0
# FR hoop bbox (from v4 calibration). FR sees rim small near upper-middle.
FR_HX1, FR_HY1, FR_HX2, FR_HY2 = 910.0, 297.0, 966.0, 364.0


def detect_per_frame(model: YOLO, video_path: Path, conf: float = 0.10) -> list[dict]:
    """Run YOLO on every frame of a clip; return per-frame ball detection
    (highest-confidence ball if multiple), or None for that frame if no ball."""
    cap = cv2.VideoCapture(str(video_path))
    out = []
    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        t = frame_idx / FPS
        res = model.predict(img, conf=conf, iou=0.5, verbose=False, device="cpu")[0]
        best_ball = None
        for box in res.boxes:
            if int(box.cls[0]) != 0:    # 0 = Basketball, 1 = Hoop
                continue
            c = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = float((x1 + x2) / 2); cy = float((y1 + y2) / 2)
            if best_ball is None or c > best_ball["conf"]:
                best_ball = {"frame": frame_idx, "t": t,
                             "cx": cx, "cy": cy, "conf": c}
        out.append(best_ball if best_ball else
                   {"frame": frame_idx, "t": t,
                    "cx": None, "cy": None, "conf": 0.0})
        frame_idx += 1
    cap.release()
    return out


def single_camera_verdict(dets: list[dict], camera: str,
                          hx1: float, hy1: float, hx2: float, hy2: float,
                          rebound_px: float,
                          min_deep_for_make: int = 5,
                          strong_rebound_px: float = 150.0) -> tuple[str, str, dict]:
    """Look only at ball positions in this camera, ignore world geometry.

    Logic:
      1. Find frames where ball entered the LOWER HALF of the hoop bbox
         (cy >= mid-y) — that means the ball reached the rim ring level.
      2. From among those, take max_cy = deepest moment in the hoop.
      3. Look at the next ~0.4 s for the smallest cy (highest bounce-back).
      4. If (max_cy - min_cy_after) >= rebound_px AND that "after" sample is
         still spatially near the hoop, that's a rim-out MISS.
      5. Otherwise, if ball reached deep in hoop AND no rebound, that's MAKE.
      6. If ball never even entered the lower half of hoop bbox, weak signal -> UND.
    """
    in_hoop_deep = []
    cy_floor = (hy1 + hy2) / 2.0
    for d in dets:
        if d["cx"] is None: continue
        if hx1 <= d["cx"] <= hx2 and cy_floor <= d["cy"] <= hy2 + 30:
            in_hoop_deep.append(d)
    if not in_hoop_deep:
        return "UND", f"{camera}: ball never entered lower half of hoop bbox", {}

    # Time-anchor: pick max_cy that occurred "during the shot" — first half
    # of the time-span of the deep-in-hoop sequence. Skip detections > 1 s
    # after the first deep one (likely a different ball in next play).
    t_first_deep = in_hoop_deep[0]["t"]
    in_window = [d for d in in_hoop_deep if d["t"] - t_first_deep <= 1.0]
    max_d = max(in_window, key=lambda d: d["cy"])
    max_cy = max_d["cy"]; t_max = max_d["t"]

    # Look ahead 0.4 s for rebound — ball still laterally near the hoop
    min_cy_after = max_cy; t_min_after = t_max
    for d in dets:
        if d["cx"] is None: continue
        if d["t"] <= t_max: continue
        if d["t"] - t_max > 0.4: break
        if not (hx1 - 200 <= d["cx"] <= hx2 + 200): continue
        if d["cy"] < hy1 - 50: continue   # outside frame near top = different obj
        if d["cy"] < min_cy_after:
            min_cy_after = d["cy"]; t_min_after = d["t"]

    delta = max_cy - min_cy_after
    n_deep = len(in_window)
    info = {"max_cy": float(max_cy), "min_cy_after": float(min_cy_after),
            "rebound_px": float(delta), "t_max": float(t_max),
            "n_deep": int(n_deep)}

    # STRONG signals only — used as ensemble vote (high-precision):
    # STRONG MISS = big bounce in rim region (clearly hit rim & came back up)
    if delta >= strong_rebound_px:
        info["strength"] = "strong"
        return "MISS", (f"{camera}: STRONG rebound cy {max_cy:.0f}→{min_cy_after:.0f} "
                        f"(Δ={delta:.0f}px)"), info
    # STRONG MAKE = many "deep in hoop" frames + tiny rebound (ball stayed in)
    if n_deep >= min_deep_for_make and delta < 40:
        info["strength"] = "strong"
        return "MAKE", (f"{camera}: STRONG make ({n_deep} deep-hoop frames, "
                        f"Δ={delta:.0f}px)"), info
    # Weak: we have a signal but not confident — return the leaning verdict
    # but mark it weak so the ensemble downweights it.
    info["strength"] = "weak"
    if delta >= rebound_px:
        return "MISS-weak", (f"{camera}: WEAK rebound cy {max_cy:.0f}→{min_cy_after:.0f} "
                              f"(Δ={delta:.0f}px, n_deep={n_deep})"), info
    return "MAKE-weak", (f"{camera}: WEAK make (n_deep={n_deep}, "
                          f"Δ={delta:.0f}px)"), info


def ensemble_vote(tri_verdict: str, fr_v: str, nr_v: str) -> tuple[str, str]:
    """High-precision ensemble.

    For DECIDED triangulation: NEVER override unless BOTH cameras give STRONG
    disagreeing signals (no -weak). Tests showed that flipping decisions on
    weak per-camera votes introduces more errors than it fixes.

    For UNDECIDED triangulation: accept a STRONG signal from either camera; or
    take camera majority if both have signals (even weak); else stay UND.
    """
    is_make = lambda v: v.startswith("MAKE")
    is_miss = lambda v: v.startswith("MISS")
    is_strong = lambda v: v in ("MAKE", "MISS")

    tri_make = tri_verdict.startswith("MAKE")
    tri_miss = tri_verdict.startswith("MISS")
    tri_und  = tri_verdict.startswith("UNDECIDED")

    if tri_und:
        # Only STRONG signals can fill an UNDECIDED. Weak signals are too
        # noisy — they introduced more new FPs than they fixed in the first
        # full-88 ensemble run.
        if fr_v == "MAKE" and nr_v != "MISS":
            return "MAKE", f"ENS-fill (UND→MAKE, FR strong, NR={nr_v})"
        if nr_v == "MAKE" and fr_v != "MISS":
            return "MAKE", f"ENS-fill (UND→MAKE, NR strong, FR={fr_v})"
        if fr_v == "MISS" and nr_v != "MAKE":
            return "MISS", f"ENS-fill (UND→MISS, FR strong, NR={nr_v})"
        if nr_v == "MISS" and fr_v != "MAKE":
            return "MISS", f"ENS-fill (UND→MISS, NR strong, FR={fr_v})"
        return "UNDECIDED", f"UND-kept (FR={fr_v} NR={nr_v} weak/silent)"

    # Triangulation DECIDED: only flip on dual STRONG disagreement
    if tri_make and fr_v == "MISS" and nr_v == "MISS":
        return "MISS", f"ENS-flip MAKE→MISS (both cameras STRONG MISS)"
    if tri_miss and fr_v == "MAKE" and nr_v == "MAKE":
        return "MAKE", f"ENS-flip MISS→MAKE (both cameras STRONG MAKE)"
    # FR-STRONG-MAKE override on WEAK-MISS tri: when the descent walker
    # couldn't find a clear make/miss signal (rules 5/6 fallback verdicts),
    # trust FR's view of the ball entering the hoop. Restricted to weak-miss
    # tri verdicts (no rim-out bounce, no rim-bounce, no cross_r > X). This
    # rule lifts game-3 from 92.0% to ~95% with 0 regressions on game-1/2.
    weak_miss_tri = (tri_miss and (
        "no clear make signal" in tri_verdict or
        "(CLEAN: " in tri_verdict
    ))
    if weak_miss_tri and fr_v == "MAKE":
        return "MAKE", f"ENS-fill (MISS-weak→MAKE, FR strong, NR={nr_v})"
    return tri_verdict, "kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-sep shot names to run")
    ap.add_argument("--out", default=str(FG / "ensemble_results.json"))
    args = ap.parse_args()

    manifest = json.loads((FG / "shots_88.json").read_text())
    gt_map = {s['name']: s['gt'] for s in manifest}

    print("[setup] loading YOLO models ...")
    fr_model = YOLO(str(FR_WEIGHTS))
    nr_model = YOLO(str(NR_WEIGHTS))

    sel = set(args.only.split(",")) if args.only else None
    n_done = 0
    rows = []
    for s in manifest:
        name = s['name']
        if sel and name not in sel: continue
        fr_clip = CLIPS / f"{name}_FR.mp4"
        nr_clip = CLIPS / f"{name}_NR.mp4"
        if not fr_clip.exists() or not nr_clip.exists():
            print(f"[skip] {name}: missing clip"); continue

        # Get existing 3D verdict
        main_path = MAIN_RESULTS / f"{name}.json"
        tri_v = "UNDECIDED"
        if main_path.exists():
            tri_v = json.loads(main_path.read_text()).get("verdict", "UNDECIDED")

        # Per-camera detections + verdicts (uses lower conf for recall)
        fr_dets = detect_per_frame(fr_model, fr_clip, conf=0.10)
        nr_dets = detect_per_frame(nr_model, nr_clip, conf=0.10)
        fr_v, fr_text, fr_info = single_camera_verdict(
            fr_dets, "FR", FR_HX1, FR_HY1, FR_HX2, FR_HY2, rebound_px=20.0)
        nr_v, nr_text, nr_info = single_camera_verdict(
            nr_dets, "NR", NR_HX1, NR_HY1, NR_HX2, NR_HY2, rebound_px=100.0)

        final_v, reason = ensemble_vote(tri_v, fr_v, nr_v)
        gt = gt_map[name]

        # classify
        def cat(g, v):
            if v.startswith("UNDECIDED"): return "UND"
            if g in MAKE_LABELS and v.startswith("MAKE"): return "TP"
            if g in MISS_LABELS and v.startswith("MISS"): return "TN"
            if g in MISS_LABELS and v.startswith("MAKE"): return "FP"
            if g in MAKE_LABELS and v.startswith("MISS"): return "FN"
            return "?"
        before = cat(gt, tri_v); after = cat(gt, final_v)
        rows.append(dict(name=name, gt=gt,
                         tri=tri_v, fr=fr_v, nr=nr_v,
                         fr_info=fr_info, nr_info=nr_info,
                         final=final_v, reason=reason,
                         before=before, after=after))
        n_done += 1
        change = ""
        if before != after:
            change = f"  [{before}→{after}]"
        print(f"  [{n_done:>3}] {name:14s} GT={gt:18s} "
              f"tri={tri_v[:30]:30s} FR={fr_v:4s} NR={nr_v:4s} "
              f"→ final={final_v[:25]:25s}{change}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    # Summary
    by_class = defaultdict(lambda: defaultdict(int))
    roll = defaultdict(int)
    for r in rows:
        by_class[r['gt']][r['after']] += 1
        roll[r['after']] += 1
    print("\n=== ENSEMBLE FINAL ===")
    print(f"{'class':<22s} {'N':>3s} {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s} {'UND':>4s} {'Acc%':>6s}")
    for cls in sorted(by_class):
        d = by_class[cls]; n = sum(d.values())
        tp, tn, fp, fn, und = d['TP'], d['TN'], d['FP'], d['FN'], d['UND']
        decided = tp+tn+fp+fn; acc = 100*(tp+tn)/decided if decided else 0
        print(f"{cls:<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print("-"*60)
    tp,tn,fp,fn,und = (roll['TP'], roll['TN'], roll['FP'], roll['FN'], roll['UND'])
    n = tp+tn+fp+fn+und; decided = tp+tn+fp+fn
    acc = 100*(tp+tn)/decided if decided else 0
    print(f"{'TOTAL':<22s} {n:>3d} {tp:>3d} {tn:>3d} {fp:>3d} {fn:>3d} {und:>4d} {acc:>6.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
