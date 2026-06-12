#!/usr/bin/env python3
"""Phase 0: dense rim-zoom film strips for visual inspection.

For each clip: crop the (per-era fixed) rim region, locate the rim event as
the peak of frame-diff motion energy inside the GT shot window, then tile
stride-2 frames spanning the event. Detector-independent on purpose --
this is how we LOOK at the data; YOLO boxes are overlaid only as a probe.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
P0 = REPO / "data/client_report/near_angle/phase0"

# rim crop per era (hand-located from gridded frames, fixed camera mount)
CROP_JUNE = (640, 1330, 700, 1080)   # x1,x2,y1,y2
CROP_OLD = (650, 1250, 750, 1080)
COLS, MAX_CELLS, CELL_W = 3, 36, 440
PAGE = 12  # cells per output image -- keeps each PNG under API downscale size


def load_windows():
    win = {}
    m = json.loads((P0 / "phase0_manifest.json").read_text())
    for s in m:
        win[Path(s["clip"]).stem] = (s["t_start_clip"], s["t_end_clip"])
    sel = json.loads((P0 / "old_era_selection.json").read_text())
    for s in sel:
        stem = f"{s['game']}_{s['name']}_{s['gt']}"
        t0 = max(0.0, s["t_start"] - 1.0)
        win[stem] = (s["t_start"] - t0, s["t_end"] - t0)
    return win


def rim_strip(clip: Path, crop, t_lo, t_hi, dets, out_png: Path):
    x1, x2, y1, y2 = crop
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    if n == 0:
        return None

    # motion energy in rim crop, search inside GT window (+2s descent buffer)
    lo = max(1, int(t_lo * fps))
    hi = min(n - 1, int((t_hi + 2.0) * fps))
    prev = None
    energy = np.zeros(n)
    for i in range(max(0, lo - 1), hi + 1):
        g = cv2.cvtColor(frames[i][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is not None:
            energy[i] = float(np.mean(cv2.absdiff(g, prev)))
        prev = g
    peak = int(np.argmax(energy[lo:hi + 1])) + lo if hi > lo else n // 2

    half = int(2.5 * fps)
    idxs = list(range(max(0, peak - half), min(n, peak + half + 1), 2))
    if len(idxs) > MAX_CELLS:
        idxs = [idxs[int(k * (len(idxs) - 1) / (MAX_CELLS - 1))]
                for k in range(MAX_CELLS)]
    _render(frames, dets, crop, idxs, peak, out_png)

    # second strip: uniform over the whole GT window (+2s) as fallback
    # coverage in case the motion peak centered on the wrong event
    widxs = list(range(max(0, lo - int(0.5 * fps)), hi + 1))
    if len(widxs) > MAX_CELLS:
        widxs = [widxs[int(k * (len(widxs) - 1) / (MAX_CELLS - 1))]
                 for k in range(MAX_CELLS)]
    _render(frames, dets, crop, widxs, peak,
            out_png.with_name(out_png.stem + "_win.png"))
    return peak


def _render(frames, dets, crop, idxs, peak, out_png: Path):
    x1, x2, y1, y2 = crop
    cells = []
    ch = int(CELL_W * (y2 - y1) / (x2 - x1))
    for i in idxs:
        f = frames[i].copy()
        for d in dets[i] if i < len(dets) else []:
            bx1, by1, bx2, by2 = [int(v) for v in d["xyxy"]]
            color = (0, 200, 255) if d["cls"] == 0 else (255, 120, 0)
            cv2.rectangle(f, (bx1, by1), (bx2, by2), color, 2)
        c = cv2.resize(f[y1:y2, x1:x2], (CELL_W, ch))
        tag = f"f{i}" + (" *PEAK*" if abs(i - peak) <= 1 else "")
        cv2.putText(c, tag, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cells.append(c)
    for p in range(0, len(cells), PAGE):
        page_cells = cells[p:p + PAGE]
        rows = []
        for r in range(0, len(page_cells), COLS):
            row = page_cells[r:r + COLS]
            while len(row) < COLS:
                row.append(np.zeros_like(cells[0]))
            rows.append(np.hstack(row))
        out = out_png.with_name(f"{out_png.stem}_p{p // PAGE + 1}.png")
        cv2.imwrite(str(out), np.vstack(rows))


def main():
    win = load_windows()
    (P0 / "rimstrip").mkdir(exist_ok=True)
    peaks = {}
    jobs = [(c, CROP_JUNE) for c in sorted((P0 / "clips").glob("*.mp4"))] + \
           [(c, CROP_OLD) for c in sorted((P0 / "clips_old_era").glob("*.mp4"))]
    for clip, crop in jobs:
        stem = clip.stem
        det_path = P0 / "detections" / (stem + ".json")
        dets = (json.loads(det_path.read_text())["frames"]
                if det_path.exists() else [])
        t_lo, t_hi = win.get(stem, (1.0, 6.0))
        peak = rim_strip(clip, crop, t_lo, t_hi, dets,
                         P0 / "rimstrip" / (stem + ".png"))
        peaks[stem] = peak
        print(f"{stem}: peak f{peak}")
    (P0 / "rimstrip_peaks.json").write_text(json.dumps(peaks, indent=1))
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
