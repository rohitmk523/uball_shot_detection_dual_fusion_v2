#!/usr/bin/env python3
"""Train the rim-crop make/miss classifier on ALL 17 games and save weights,
for the end-to-end shot-detection test on a FROZEN game (unseen by any model).

Reuses the model/dataset from phase2_train. Applies the 5 audited label
corrections. Saves state_dict + the config needed to run inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_train import (ClipDS, build_model, device, CACHE, N_FRAMES,
                          USE_DIFF, BATCH, LR, RES)

EPOCHS = 22
OUT = Path(__file__).resolve().parent / "weights/classifier_all17.pt"


def main():
    frames = np.load(CACHE / "frames_u8.npy", mmap_mode="r")
    meta = json.loads((CACHE / "meta.json").read_text())
    # apply audited label corrections
    corr_path = Path(__file__).resolve().parents[1] / \
        "data/near_rimcrop/label_corrections.json"
    corr = json.loads(corr_path.read_text())["corrected"] if corr_path.exists() else {}
    y = np.array([1 if (corr.get(m["name"], m["make"])) else 0 for m in meta])
    n_fix = sum(1 for m in meta if m["name"] in corr)

    dev = device()
    in_ch = N_FRAMES + (N_FRAMES - 1 if USE_DIFF else 0)
    print(f"device={dev} clips={len(meta)} corrections_applied={n_fix} in_ch={in_ch}")

    dl = DataLoader(ClipDS(frames, y, True), BATCH, shuffle=True,
                    num_workers=0, drop_last=True)
    model = build_model(in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    w = torch.tensor([1.0, (y == 0).sum() / max(1, (y == 1).sum())],
                     device=dev, dtype=torch.float32)
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
            loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"epoch {ep} loss {tot/len(dl):.3f}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_ch": in_ch,
                "n_frames": N_FRAMES, "res": RES, "use_diff": USE_DIFF},
               OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
