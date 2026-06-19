#!/usr/bin/env python3
"""Fine-tune YOLO11n for the near-angle overhead view (ball + hoop).

Warm-starts from the existing near-angle best.pt (same 2-class schema:
0=Basketball, 1=Basketball Hoop) so basketball features transfer; the 4,386
verified overhead frames adapt the hoop class to the top-down rim+net and
teach the teal ball. Whole-game val/test (no frame leakage).

--smoke: 2 epochs, imgsz 640, tiny -- validate the pipeline end to end.
Full run is GPU-heavy (-> AWS per the >1hr rule), checkpoints every 10 ep.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

NA = Path("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
          "Training_frameworks/Uball Near Angle/data")
DATA = NA / "yolo_split/dataset.yaml"
BASE = ("/Users/rohitkale/Cellstrat/GitHub_Repositories/"
        "Uball_dual_angle_shot_detection/weights/near_angle_weights/"
        "basketball_yolo11n3/weights/best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--project", default=str(NA.parent / "runs"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    base = args.base if Path(args.base).exists() else "yolo11n.pt"
    # Docker containers default to 64MB /dev/shm; YOLO's multi-worker
    # DataLoader OOMs there. workers=0 avoids shared-memory tensor passing
    # (set NW>0 only when running outside a shm-constrained container).
    import os
    nw = int(os.environ.get("NW", "0"))
    print(f"device={dev} base={base} data={args.data} workers={nw}")

    if args.resume:
        checkpoint_path = Path(args.project) / "near-det-v1/weights/last.pt"
        if not checkpoint_path.exists():
            checkpoint_path = Path(base)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))
        model.train(resume=True, workers=nw)
        print("TRAIN_DONE")
        return

    model = YOLO(base)
    common = dict(data=args.data, device=dev, project=args.project,
                  exist_ok=True, seed=42, workers=nw)
    if args.smoke:
        model.train(name="near-det-smoke", epochs=2, imgsz=640, batch=4,
                    augment=False, **common)
        print("SMOKE_OK")
        return

    model.train(
        name="near-det-v1", epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, patience=25,
        lr0=0.002, lrf=0.01, cos_lr=True, warmup_epochs=2.0,
        augment=True, mosaic=0.4, degrees=8, scale=0.3,
        hsv_h=0.05, hsv_s=0.7, hsv_v=0.5,        # strong color aug (teal ball)
        fliplr=0.5, translate=0.1,
        save_period=10, **common)
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
