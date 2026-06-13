#!/usr/bin/env python3
"""Build a whole-game train/val/test split for the near-angle detector.

All 5,196 verified frames currently live in .../Uball Near Angle/data/train/.
Honest detector eval needs games NOT seen in training, so we hold out whole
games (never frames from a trained game -- near-identical frames would leak).

Holdout choices cover the distribution that matters:
  - val  = 9eb51980 (Apr SuperView, NL+NR)  -> early stopping
  - test = b3c1f62c (May SuperView) + 72c08cb7 (June WIDE)  -> honest report,
           both a SuperView and the clipped-rim Wide mode.
Everything else trains. Writes dataset.yaml.
"""
from __future__ import annotations

import shutil
from pathlib import Path

NA = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
          "Training_frameworks/Uball Near Angle/data")
SRC_IMG, SRC_LBL = NA / "train/images", NA / "train/labels"
VAL_GAMES = {"9eb51980"}
TEST_GAMES = {"b3c1f62c", "72c08cb7"}


def split_of(stem: str) -> str:
    g = stem.split("_")[0]
    if g in TEST_GAMES:
        return "test"
    if g in VAL_GAMES:
        return "val"
    return "train"


def main():
    # stage everything into a sibling split dir so the annotation source
    # (train/) stays intact and re-runnable.
    root = NA / "yolo_split"
    for s in ("train", "val", "test"):
        (root / s / "images").mkdir(parents=True, exist_ok=True)
        (root / s / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    for img in sorted(SRC_IMG.glob("*.jpg")):
        lbl = SRC_LBL / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        s = split_of(img.stem)
        shutil.copy(img, root / s / "images" / img.name)
        shutil.copy(lbl, root / s / "labels" / lbl.name)
        counts[s] += 1
    yaml = f"""# Near-angle ball+hoop detector. Whole-game holdout (no frame leakage).
# val={sorted(VAL_GAMES)}  test={sorted(TEST_GAMES)} (incl. June WIDE 72c08cb7)
path: {root}
train: train/images
val:   val/images
test:  test/images
names:
  0: Basketball
  1: Basketball Hoop
"""
    (root / "dataset.yaml").write_text(yaml)
    print(f"split counts: {counts}")
    print(f"dataset.yaml -> {root/'dataset.yaml'}")


if __name__ == "__main__":
    main()
