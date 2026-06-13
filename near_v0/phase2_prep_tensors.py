#!/usr/bin/env python3
"""Phase 2 prep: decode each usable rim-crop clip into a fixed tensor cache.

For each usable clip in dataset_index.json:
  - uniformly sample N_FRAMES frames across the clip
  - grayscale, resize to RES x RES, store uint8

Output (portable to AWS, fast to train on):
  data/near_rimcrop/cache/frames_u8.npy   [N, N_FRAMES, RES, RES] uint8
  data/near_rimcrop/cache/meta.json        parallel list of {name,game,cam,
                                            make,gt,flags,t1_game}

Grayscale-only on purpose: hue is not a load-bearing cue (taxonomy section 5);
frame-difference channels are derived on-the-fly in the trainer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RC = REPO / "data/near_rimcrop"
CACHE = RC / "cache"
N_FRAMES = 16
RES = 160


def sample_idx(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def load_clip(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, (RES, RES), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    idx = sample_idx(len(frames), N_FRAMES)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)


def main():
    idx = json.loads((RC / "dataset_index.json").read_text())
    recs = [r for r in idx["records"] if r["usable"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((len(recs), N_FRAMES, RES, RES), dtype=np.uint8)
    meta, ok = [], 0
    for i, r in enumerate(recs):
        clip = load_clip(REPO / r["path"])
        if clip is None:
            continue
        arr[ok] = clip
        rec = {k: r[k] for k in
               ("game", "cam", "make", "gt", "flags", "t1_game")}
        rec["name"] = Path(r["path"]).stem
        meta.append(rec)
        ok += 1
        if i % 200 == 0:
            print(f"{i}/{len(recs)}", flush=True)
    arr = arr[:ok]
    np.save(CACHE / "frames_u8.npy", arr)
    (CACHE / "meta.json").write_text(json.dumps(meta))
    print(f"cached {ok}/{len(recs)} clips -> {arr.shape} "
          f"({arr.nbytes/1e6:.0f} MB); makes={sum(m['make'] for m in meta)}")


if __name__ == "__main__":
    sys.exit(main())
