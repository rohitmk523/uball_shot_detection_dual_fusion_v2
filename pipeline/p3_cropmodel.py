#!/usr/bin/env python3
"""
P3 crop model — a SMALL spatio-temporal pixel model on rim-region crops.

Tests the hypothesis that a pixel model on rim-region crops (extract_crops.py)
beats the box-feature model for make/miss. Per shot the input is up to 4 angles
x [T=16, 64, 64] grayscale crops.

Architecture (a few hundred K params, plausibly real-time):
  * a small per-frame 2D-CNN encoder (shared weights across T and across the 4
    angles) -> a per-frame embedding,
  * temporal mean+max pool over T -> a per-angle embedding,
  * the 4 angle embeddings (zero-padded for missing angles) concatenated WITH
    a 4-d presence flag vector,
  * a small MLP head -> make/miss logit.

Data:
  * whole-game splits from data/games_manifest.json,
  * crops loaded from a local crops dir (default data/crops/<game_id>/crops.npz
    + crops_meta.json, overridable with --crops-dir; the same layout the S3
    pass writes, so you `aws s3 sync` it down first),
  * train on TRAIN games present in the crops dir, tune the decision threshold
    on VAL, evaluate ONCE on TEST.

Deterministic seed 42. Device auto-detect (cuda > mps > cpu).

Reports acc/prec/rec/AUC + confusion on TEST and SAVES per-shot test
predictions to data/p3_crops_test_predictions.parquet (schema compatible with
data/p3_test_predictions_v8.parquet so they can be compared / ensembled).

Optional ablation (--fuse-iter8 <parquet>): a simple late fusion of the crop
probability with the iter8 box-model probability (logistic regression on the
two probs, fit on VAL, evaluated on TEST). Robust + optional — skipped with a
clear message if the iter8 file or play overlap is unavailable.

Usage:
  python p3_cropmodel.py [--crops-dir data/crops] [--epochs 30]
  python p3_cropmodel.py --fuse-iter8 data/p3_test_predictions_v8.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

ANGLES = ("FL", "FR", "NL", "NR")
T = 16
SIZE = 64
SEED = 42


# ---------------------------------------------------------------------------
# Determinism + device.
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Data loading from the per-game npz + meta produced by extract_crops.py.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shot:
    game_id: str
    play_id: str
    classification: str
    label: int
    split: str
    # angle -> np.ndarray[T,SIZE,SIZE] uint8 (only present angles)
    crops: Dict[str, "object"]


def _manifest_splits() -> Dict[str, str]:
    """game_id -> split, from data/games_manifest.json."""
    data = json.loads((REPO_ROOT / "data" / "games_manifest.json").read_text())
    return {g["game_id"]: g.get("split", "train") for g in data["games"]}


def load_shots(crops_dir: Path) -> List[Shot]:
    """Load every shot from every <game_id>/crops.npz present under crops_dir.
    The npz is keyed f"{play_id}_{angle}"; crops_meta.json gives label /
    classification / split per play."""
    import numpy as np

    splits = _manifest_splits()
    shots: List[Shot] = []
    if not crops_dir.exists():
        return shots

    for game_dir in sorted(p for p in crops_dir.iterdir() if p.is_dir()):
        npz_path = game_dir / "crops.npz"
        meta_path = game_dir / "crops_meta.json"
        if not npz_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        game_id = meta.get("game_id", game_dir.name)
        split = splits.get(game_id, meta.get("split", "train"))
        plays = meta.get("plays", {})

        with np.load(npz_path) as npz:
            keys = list(npz.files)
            # Group keys by play_id (strip the trailing "_<angle>").
            by_play: Dict[str, Dict[str, object]] = {}
            for key in keys:
                for ang in ANGLES:
                    suffix = f"_{ang}"
                    if key.endswith(suffix):
                        pid = key[: -len(suffix)]
                        arr = np.asarray(npz[key])
                        by_play.setdefault(pid, {})[ang] = arr
                        break
            for pid, crops in by_play.items():
                pmeta = plays.get(pid, {})
                cls = pmeta.get("classification", "UNKNOWN")
                label = int(pmeta.get("label", 0))
                shots.append(Shot(
                    game_id=game_id, play_id=pid, classification=cls,
                    label=label, split=split, crops=crops,
                ))
    return shots


def shots_to_tensors(shots: List[Shot]):
    """Stack a list of Shots into:
      X:        float32 [N, 4, T, SIZE, SIZE]  (normalised to [0,1], zeros for
                missing angles)
      presence: float32 [N, 4]                 (1.0 if that angle is present)
      y:        float32 [N]
    Angle order is fixed = ANGLES.
    """
    import numpy as np

    n = len(shots)
    X = np.zeros((n, len(ANGLES), T, SIZE, SIZE), dtype="float32")
    presence = np.zeros((n, len(ANGLES)), dtype="float32")
    y = np.zeros((n,), dtype="float32")
    for i, s in enumerate(shots):
        y[i] = float(s.label)
        for a, ang in enumerate(ANGLES):
            arr = s.crops.get(ang)
            if arr is None:
                continue
            arr = np.asarray(arr)
            # Defensive: enforce [T,SIZE,SIZE]; pad/trim T if needed.
            if arr.shape != (T, SIZE, SIZE):
                fixed = np.zeros((T, SIZE, SIZE), dtype="float32")
                tt = min(T, arr.shape[0]) if arr.ndim == 3 else 0
                for k in range(tt):
                    fr = arr[k]
                    if fr.shape == (SIZE, SIZE):
                        fixed[k] = fr.astype("float32")
                arr = fixed
            X[i, a] = arr.astype("float32") / 255.0
            presence[i, a] = 1.0
    return X, presence, y


# ---------------------------------------------------------------------------
# Model. Small per-frame 2D-CNN encoder (shared) -> temporal mean+max pool ->
# per-angle embedding -> concat 4 angles + presence flags -> MLP head.
# ---------------------------------------------------------------------------

def build_model(emb_dim: int = 96, hidden: int = 192):
    import torch
    import torch.nn as nn

    class FrameEncoder(nn.Module):
        """Tiny 2D-CNN: 1x64x64 -> emb_dim. Shared across T and angles."""

        def __init__(self, emb: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 64 -> 32
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),  # 16 -> 8
                nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),                    # -> 48x1x1
            )
            self.proj = nn.Linear(48, emb)

        def forward(self, x):  # x: [B, 1, 64, 64]
            h = self.features(x).flatten(1)  # [B, 48]
            return self.proj(h)              # [B, emb]

    class CropModel(nn.Module):
        def __init__(self, emb: int, hid: int):
            super().__init__()
            self.encoder = FrameEncoder(emb)
            # per-angle embedding = mean-pool ++ max-pool over T -> 2*emb.
            # 4 angles concatenated -> 4 * 2 * emb, plus 4 presence flags.
            head_in = len(ANGLES) * 2 * emb + len(ANGLES)
            self.head = nn.Sequential(
                nn.Linear(head_in, hid),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hid, 1),
            )
            self.emb = emb

        def forward(self, x, presence):
            # x: [B, A, T, 64, 64]; presence: [B, A]
            b, a, t, h, w = x.shape
            flat = x.reshape(b * a * t, 1, h, w)
            emb = self.encoder(flat)                  # [B*A*T, emb]
            emb = emb.reshape(b, a, t, self.emb)
            mean_pool = emb.mean(dim=2)               # [B, A, emb]
            max_pool = emb.max(dim=2).values          # [B, A, emb]
            ang_emb = torch.cat([mean_pool, max_pool], dim=-1)  # [B, A, 2*emb]
            # Zero out absent angles so padding contributes nothing.
            ang_emb = ang_emb * presence.unsqueeze(-1)
            feat = torch.cat(
                [ang_emb.reshape(b, -1), presence], dim=-1
            )
            return self.head(feat).squeeze(-1)        # [B] logits

    return CropModel(emb_dim, hidden)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Metrics (no sklearn dependency for the core; sklearn used only if present
# for AUC, with a deterministic fallback).
# ---------------------------------------------------------------------------

def _auc(y_true: List[int], scores: List[float]) -> float:
    """ROC AUC via the rank (Mann-Whitney U) formula. Deterministic, handles
    ties. Returns 0.5 if a class is absent."""
    pairs = sorted(zip(scores, y_true), key=lambda p: p[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks (1-based) to handle ties.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_report(y_true: List[int], probs: List[float],
                          threshold: float) -> dict:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for t, p in zip(y_true, preds) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, preds) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, preds) if t == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(_auc(y_true, probs), 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": n,
    }


def best_threshold(y_true: List[int], probs: List[float]) -> float:
    """Pick the threshold (over a fine grid) that maximises accuracy on the
    given set (used on VAL). Ties broken toward 0.5."""
    grid = [i / 100.0 for i in range(1, 100)]
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        rep = classification_report(y_true, probs, t)
        acc = rep["accuracy"]
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc, best_t = acc, t
    return best_t


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def _iterate_batches(n: int, batch_size: int, shuffle: bool, rng):
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_model(model, device, X, presence, y, *, epochs: int,
                batch_size: int, lr: float, pos_weight: float):
    import torch
    import torch.nn as nn

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    rng = random.Random(SEED)
    n = X.shape[0]
    Xt = torch.from_numpy(X)
    Pt = torch.from_numpy(presence)
    yt = torch.from_numpy(y)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in _iterate_batches(n, batch_size, True, rng):
            bi = torch.tensor(batch, dtype=torch.long)
            xb = Xt[bi].to(device)
            pb = Pt[bi].to(device)
            yb = yt[bi].to(device)
            opt.zero_grad()
            logits = model(xb, pb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch)
        eprint(f"[train] epoch {ep + 1}/{epochs} loss={total / max(1, n):.4f}")
    return model


def predict_probs(model, device, X, presence, batch_size: int = 64) -> List[float]:
    import torch
    model.eval()
    n = X.shape[0]
    Xt = torch.from_numpy(X)
    Pt = torch.from_numpy(presence)
    out: List[float] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = Xt[i:i + batch_size].to(device)
            pb = Pt[i:i + batch_size].to(device)
            logits = model(xb, pb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            out.extend(probs)
    return out


# ---------------------------------------------------------------------------
# Optional iter8 late fusion (simple logistic regression on the two probs).
# ---------------------------------------------------------------------------

def _load_iter8_probs(path: Path) -> Dict[Tuple[str, str], float]:
    """(game_id, play_id) -> iter8 prob, from a p3_test_predictions parquet."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    d = table.to_pydict()
    out: Dict[Tuple[str, str], float] = {}
    for gid, pid, prob in zip(d["game_id"], d["play_id"], d["prob"]):
        out[(str(gid), str(pid))] = float(prob)
    return out


def _fit_logreg_2feat(feats: List[List[float]], labels: List[int]):
    """Fit a tiny 2-feature logistic regression by GD. Deterministic,
    dependency-free. Returns (w0, w1, b)."""
    import numpy as np
    Xf = np.asarray(feats, dtype="float64")
    yf = np.asarray(labels, dtype="float64")
    w = np.zeros(2)
    b = 0.0
    lr = 0.5
    for _ in range(2000):
        z = Xf @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xf.T @ (p - yf) / len(yf)
        grad_b = float(np.mean(p - yf))
        w -= lr * grad_w
        b -= lr * grad_b
    return float(w[0]), float(w[1]), float(b)


def _logreg_predict(w0, w1, b, feats: List[List[float]]) -> List[float]:
    import numpy as np
    Xf = np.asarray(feats, dtype="float64")
    z = Xf[:, 0] * w0 + Xf[:, 1] * w1 + b
    return (1.0 / (1.0 + np.exp(-z))).tolist()


# ---------------------------------------------------------------------------
# Persisting test predictions (schema-compatible with the box-model output).
# ---------------------------------------------------------------------------

def save_test_predictions(path: Path, shots: List[Shot], probs: List[float],
                          preds: List[int]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "game_id": [s.game_id for s in shots],
        "play_id": [s.play_id for s in shots],
        "classification": [s.classification for s in shots],
        "label": [int(s.label) for s in shots],
        "prob": [float(p) for p in probs],
        "pred": [int(p) for p in preds],
        "correct": [int(s.label) == int(pr) for s, pr in zip(shots, preds)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def run(crops_dir: Path, epochs: int, batch_size: int, lr: float,
        fuse_iter8: Optional[Path], out_pred: Path) -> int:
    set_seed(SEED)
    device = pick_device()
    eprint(f"[p3crop] device={device} seed={SEED}")

    shots = load_shots(crops_dir)
    if not shots:
        eprint(
            f"[p3crop] no crops found under {crops_dir}. Sync the P1c output "
            f"down first, e.g.\n"
            f"  aws s3 sync s3://uball-cv-results/cv-results/dual-fusion-v2/"
            f"crops {crops_dir}\n"
            f"Nothing to train on; exiting 0 (no-op)."
        )
        return 0

    by_split: Dict[str, List[Shot]] = {"train": [], "val": [], "test": []}
    for s in shots:
        by_split.setdefault(s.split, []).append(s)
    eprint(f"[p3crop] shots: train={len(by_split['train'])} "
           f"val={len(by_split['val'])} test={len(by_split['test'])}")

    if not by_split["train"]:
        eprint("[p3crop] no TRAIN shots present; cannot train. Exiting 0.")
        return 0

    Xtr, Ptr, ytr = shots_to_tensors(by_split["train"])
    n_pos = float(ytr.sum())
    n_neg = float(len(ytr) - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = build_model()
    n_params = count_params(model)
    eprint(f"[p3crop] model params: {n_params:,}")

    train_model(model, device, Xtr, Ptr, ytr, epochs=epochs,
                batch_size=batch_size, lr=lr, pos_weight=pos_weight)

    # Threshold tuning on VAL (fallback to 0.5 if no val present).
    if by_split["val"]:
        Xva, Pva, yva = shots_to_tensors(by_split["val"])
        val_probs = predict_probs(model, device, Xva, Pva)
        threshold = best_threshold([int(v) for v in yva.tolist()], val_probs)
        eprint(f"[p3crop] tuned threshold on VAL = {threshold}")
    else:
        threshold = 0.5
        eprint("[p3crop] no VAL shots; using threshold 0.5")

    if not by_split["test"]:
        eprint("[p3crop] no TEST shots present; skipping eval. Exiting 0.")
        return 0

    test_shots = by_split["test"]
    Xte, Pte, yte = shots_to_tensors(test_shots)
    y_test = [int(v) for v in yte.tolist()]
    test_probs = predict_probs(model, device, Xte, Pte)
    rep = classification_report(y_test, test_probs, threshold)
    test_preds = [1 if p >= threshold else 0 for p in test_probs]

    print("\n================ P3 CROP MODEL — TEST (crops-only) ============")
    print(f" model params      : {n_params:,}")
    print(f" device            : {device}")
    print(f" tuned threshold   : {rep['threshold']}")
    print(f" accuracy          : {rep['accuracy']}")
    print(f" precision         : {rep['precision']}")
    print(f" recall            : {rep['recall']}")
    print(f" f1                : {rep['f1']}")
    print(f" auc               : {rep['auc']}")
    c = rep["confusion"]
    print(f" confusion         : tp={c['tp']} tn={c['tn']} "
          f"fp={c['fp']} fn={c['fn']}  (n={rep['n']})")
    print("================================================================")

    save_test_predictions(out_pred, test_shots, test_probs, test_preds)
    print(f"[p3crop] wrote per-shot test predictions -> {out_pred}")

    # ---- Optional late fusion with iter8 box-model probabilities ----------
    if fuse_iter8 is not None:
        _run_fusion_ablation(
            fuse_iter8, device, model, by_split, threshold)

    return 0


def _run_fusion_ablation(fuse_iter8: Path, device, model,
                         by_split: Dict[str, List[Shot]],
                         crop_threshold: float) -> None:
    """Robust, optional crops+iter8 late fusion. Fits a 2-feature logistic
    regression (crop prob, iter8 prob) on VAL, evaluates on TEST. Anything
    missing -> a clear skip message, never an exception that aborts the run."""
    try:
        if not fuse_iter8.exists():
            print(f"[fusion] iter8 file {fuse_iter8} not found; skipping.")
            return
        iter8 = _load_iter8_probs(fuse_iter8)
    except Exception as e:
        print(f"[fusion] could not load iter8 probs ({e}); skipping.")
        return

    def overlap(shots: List[Shot]):
        keep = [s for s in shots if (s.game_id, s.play_id) in iter8]
        return keep

    val_shots = overlap(by_split.get("val", []))
    test_shots = overlap(by_split.get("test", []))
    if len(val_shots) < 10 or len(test_shots) < 10:
        print(f"[fusion] insufficient play overlap with iter8 "
              f"(val={len(val_shots)}, test={len(test_shots)}); skipping.")
        return

    Xva, Pva, yva = shots_to_tensors(val_shots)
    Xte, Pte, yte = shots_to_tensors(test_shots)
    val_crop = predict_probs(model, device, Xva, Pva)
    test_crop = predict_probs(model, device, Xte, Pte)

    val_feats = [[val_crop[i], iter8[(s.game_id, s.play_id)]]
                 for i, s in enumerate(val_shots)]
    test_feats = [[test_crop[i], iter8[(s.game_id, s.play_id)]]
                  for i, s in enumerate(test_shots)]
    yval = [int(v) for v in yva.tolist()]
    ytest = [int(v) for v in yte.tolist()]

    w0, w1, b = _fit_logreg_2feat(val_feats, yval)
    val_fused = _logreg_predict(w0, w1, b, val_feats)
    fuse_thr = best_threshold(yval, val_fused)
    test_fused = _logreg_predict(w0, w1, b, test_feats)
    rep = classification_report(ytest, test_fused, fuse_thr)

    # Crops-only baseline on the SAME overlap subset for an apples-to-apples
    # comparison.
    rep_crop = classification_report(ytest, test_crop, crop_threshold)
    rep_iter8 = classification_report(
        ytest, [iter8[(s.game_id, s.play_id)] for s in test_shots], 0.5)

    print("\n========= P3 FUSION ABLATION (crops + iter8, TEST overlap) =====")
    print(f" overlap n (test)  : {len(test_shots)}")
    print(f" logreg weights    : crop={w0:.3f} iter8={w1:.3f} bias={b:.3f}")
    print(f" crops-only acc/auc: {rep_crop['accuracy']} / {rep_crop['auc']}")
    print(f" iter8-only acc/auc: {rep_iter8['accuracy']} / {rep_iter8['auc']}")
    print(f" FUSED   acc/auc   : {rep['accuracy']} / {rep['auc']} "
          f"(thr={rep['threshold']})")
    c = rep["confusion"]
    print(f" fused confusion   : tp={c['tp']} tn={c['tn']} "
          f"fp={c['fp']} fn={c['fn']}")
    print("================================================================")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P3 crop (pixel) make/miss model")
    ap.add_argument("--crops-dir", default=str(REPO_ROOT / "data" / "crops"),
                    help="local dir of <game_id>/crops.npz + crops_meta.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fuse-iter8", default=None,
                    help="optional iter8 predictions parquet for late fusion")
    ap.add_argument(
        "--out-pred",
        default=str(REPO_ROOT / "data" / "p3_crops_test_predictions.parquet"),
    )
    args = ap.parse_args(argv)

    return run(
        crops_dir=Path(args.crops_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fuse_iter8=Path(args.fuse_iter8) if args.fuse_iter8 else None,
        out_pred=Path(args.out_pred),
    )


if __name__ == "__main__":
    sys.exit(main())
