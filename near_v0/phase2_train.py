#!/usr/bin/env python3
"""Phase 2 v0: rim-crop make/miss classifier, leave-one-game-out (LOGO).

Model: ResNet18 with conv1 inflated to (N_FRAMES + N_DIFF) input channels.
Input per clip = N_FRAMES grayscale frames + (N_FRAMES-1) frame-difference
maps (color-independent motion; taxonomy section 5), stacked as channels.

Protocol: train on all games but one, test on the held-out game (honest
per-game accuracy). Run one fold (--fold GAME) or all (--all). Reports
decided accuracy, abstain rate (|p-0.5|<margin), and per-class.

Gate: beat 0.902 (old near+Noah). Frozen 6-game test pool never appears here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/near_rimcrop/cache"
N_FRAMES = 16
RES = 160
USE_DIFF = True
EPOCHS = 18
BATCH = 32
LR = 3e-4


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipDS(Dataset):
    def __init__(self, frames, labels, train: bool):
        self.frames = frames          # uint8 [N,16,RES,RES] (mmap ok)
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.frames[i].astype(np.float32) / 255.0   # [16,RES,RES]
        if self.train:
            if np.random.rand() < 0.5:                  # horizontal flip
                x = x[:, :, ::-1].copy()
            x = x * (0.8 + 0.4 * np.random.rand())       # brightness
            if np.random.rand() < 0.3:                   # synthetic blur
                k = np.random.choice([3, 5])
                x = np.stack([_blur(f, k) for f in x])
            x = np.clip(x, 0, 1)
        chans = [x]
        if USE_DIFF:
            d = np.abs(np.diff(x, axis=0))               # [15,RES,RES]
            chans.append(d)
        return torch.from_numpy(np.concatenate(chans, 0)), int(self.labels[i])


def _blur(f, k):
    import cv2
    return cv2.GaussianBlur(f, (k, k), 0)


def build_model(in_ch: int):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def run_fold(frames, meta, games, test_game, dev, epochs=EPOCHS, log=print):
    tr = [i for i, m in enumerate(meta) if m["game"] != test_game]
    te = [i for i, m in enumerate(meta) if m["game"] == test_game]
    y = np.array([1 if m["make"] else 0 for m in meta])
    in_ch = N_FRAMES + (N_FRAMES - 1 if USE_DIFF else 0)

    # num_workers=0: getitem just slices an mmap + cheap diff; avoids the
    # 64MB /dev/shm limit in Docker (shared-mem tensor passing OOMs there).
    nw = int(__import__("os").environ.get("NW", "0"))
    dl_tr = DataLoader(ClipDS(frames[tr], y[tr], True), BATCH, shuffle=True,
                       num_workers=nw, drop_last=True)
    dl_te = DataLoader(ClipDS(frames[te], y[te], False), BATCH, shuffle=False,
                       num_workers=nw)
    model = build_model(in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # class weights (more misses than makes)
    w = torch.tensor([1.0, (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())],
                     device=dev, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            p = F.softmax(model(xb.to(dev)), 1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(yb.numpy().tolist())
    probs, ys = np.array(probs), np.array(ys)
    pred = (probs >= 0.5).astype(int)
    acc = float((pred == ys).mean())
    names = [meta[i]["name"] for i in te]
    wrong = [{"name": names[k], "gt": int(ys[k]), "p": round(float(probs[k]), 3)}
             for k in range(len(ys)) if pred[k] != ys[k]]
    log(f"  {test_game}: n={len(ys)} acc={acc:.3f} "
        f"(makes={int(ys.sum())}/{len(ys)})")
    return {"game": test_game, "n": len(ys), "acc": acc,
            "probs": probs.tolist(), "ys": ys.tolist(), "names": names,
            "wrong": wrong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", help="single game to hold out")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(CACHE / "logo_results.json"))
    args = ap.parse_args()

    frames = np.load(CACHE / "frames_u8.npy", mmap_mode="r")
    meta = json.loads((CACHE / "meta.json").read_text())
    games = sorted({m["game"] for m in meta})
    dev = device()
    print(f"device={dev} clips={len(meta)} games={len(games)} "
          f"in_ch={N_FRAMES + (N_FRAMES-1 if USE_DIFF else 0)}")

    targets = [args.fold] if args.fold else (games if args.all else games[:1])
    results = [run_fold(frames, meta, games, g, dev, args.epochs)
               for g in targets]

    if len(results) > 1:
        all_p = np.concatenate([np.array(r["probs"]) for r in results])
        all_y = np.concatenate([np.array(r["ys"]) for r in results])
        micro = float(((all_p >= 0.5).astype(int) == all_y).mean())
        macro = float(np.mean([r["acc"] for r in results]))
        print(f"\nLOGO micro-acc={micro:.4f}  macro(per-game)={macro:.4f}  "
              f"vs gate 0.902")
        for r in sorted(results, key=lambda r: r["acc"]):
            print(f"  {r['game']}: {r['acc']:.3f} (n={r['n']})")
        Path(args.out).write_text(json.dumps(
            {"micro": micro, "macro": macro, "folds": results}, indent=0))
        print(f"saved -> {args.out}")
    else:
        Path(args.out).write_text(json.dumps({"folds": results}, indent=0))


if __name__ == "__main__":
    main()
