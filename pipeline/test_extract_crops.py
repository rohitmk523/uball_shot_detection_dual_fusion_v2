"""
Local validation for the P1c rim-region crop pass + crop model — runs WITHOUT
AWS, without any real video, and without re-running YOLO.

Covers:
  * crop_roi_from_rim geometry (rim + approach + net band, clipped to frame)
  * closest_approach_index (ball nearest the rim; fallback to centre frame)
  * sample_window_indices (fixed length T=16, correct centring + clamping)
  * make/miss label mapping
  * _process_angle output shape via a fake tracks df fed through a stubbed cv2
    (asserts [T,64,64] uint8 and that the crop is centred on the
    closest-approach frame)
  * arg-parse smoke of run_crops_batch and p3_cropmodel
  * a tiny synthetic-tensor forward+backward pass of the crop model (proves it
    trains one step and shrinks the loss)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_crops as ec  # noqa: E402


# --------------------------------------------------------------------------
# Crop-ROI geometry
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_crop_roi_covers_rim_approach_and_net():
    # rim box: top-left (100,50), w=40, h=20 -> cx=120, top=50, bottom=70
    roi = ec.crop_roi_from_rim((100.0, 50.0, 40.0, 20.0), 1920, 1080)
    assert roi is not None
    rx, ry, rw, rh = roi
    # x spans cx +/- 1.5*w = 120 +/- 60 -> [60,180], w=120 (= 3.0 * rim_w)
    assert rx == 60 and rw == 120
    # y0 = rim_top - 1.5*h = 50 - 30 = 20 (approach zone above the iron)
    assert ry == 20
    # y1 = rim_bottom + 2.5*h = 70 + 50 = 120 -> h = 100 (= 5.0 * rim_h)
    assert rh == 100
    # crop straddles the rim: top above rim_top, bottom below rim_bottom.
    assert ry < 50 and (ry + rh) > 70


@pytest.mark.unit
def test_crop_roi_clipped_to_frame():
    roi = ec.crop_roi_from_rim((1890.0, 30.0, 40.0, 20.0), 1920, 1080)
    assert roi is not None
    rx, ry, rw, rh = roi
    assert rx >= 0 and ry >= 0
    assert rx + rw <= 1920
    assert ry + rh <= 1080
    assert rw >= 2 and rh >= 2


@pytest.mark.unit
def test_crop_roi_degenerate_returns_none():
    # rim entirely below the frame -> nothing left after clipping.
    assert ec.crop_roi_from_rim((10.0, 5000.0, 40.0, 20.0), 1920, 1080) is None


# --------------------------------------------------------------------------
# Closest-approach detection
# --------------------------------------------------------------------------

def _row(frame_idx, ball=None, rim=(100.0, 50.0, 40.0, 20.0)):
    r = {
        "play_id": "p", "angle": "FL", "frame_idx": frame_idx,
        "timestamp": float(frame_idx), "classification": "FG_MAKE",
        "rim_x": rim[0], "rim_y": rim[1], "rim_w": rim[2], "rim_h": rim[3],
        "ball_x": None, "ball_y": None, "ball_w": None, "ball_h": None,
    }
    if ball is not None:
        r.update(ball_x=ball[0], ball_y=ball[1], ball_w=ball[2], ball_h=ball[3])
    return r


@pytest.mark.unit
def test_closest_approach_picks_nearest_ball_frame():
    # rim centre = (120, 60). Ball gets progressively closer, nearest at i=2.
    rows = [
        _row(0, ball=(300.0, 300.0, 10.0, 10.0)),  # far
        _row(1, ball=(200.0, 150.0, 10.0, 10.0)),  # closer
        _row(2, ball=(115.0, 55.0, 10.0, 10.0)),   # nearest (centre ~120,60)
        _row(3, ball=(180.0, 140.0, 10.0, 10.0)),  # moving away
    ]
    idx, detected = ec.closest_approach_index(rows)
    assert detected is True
    assert idx == 2


@pytest.mark.unit
def test_closest_approach_fallback_to_center_when_no_ball():
    rows = [_row(i) for i in range(7)]  # no ball anywhere
    idx, detected = ec.closest_approach_index(rows)
    assert detected is False
    assert idx == 3  # centre of 7 frames


# --------------------------------------------------------------------------
# Fixed-window sampler
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_sample_window_centered_and_fixed_length():
    idxs = ec.sample_window_indices(center=20, n=40, t=16)
    assert len(idxs) == 16
    # window [12..27], centre 20 sits at position 8 (= half of 16).
    assert idxs[0] == 12 and idxs[-1] == 27
    assert idxs[8] == 20


@pytest.mark.unit
def test_sample_window_clamps_at_left_edge():
    idxs = ec.sample_window_indices(center=2, n=40, t=16)
    assert len(idxs) == 16
    assert idxs[0] == 0 and idxs[-1] == 15


@pytest.mark.unit
def test_sample_window_clamps_at_right_edge():
    idxs = ec.sample_window_indices(center=38, n=40, t=16)
    assert len(idxs) == 16
    assert idxs[-1] == 39 and idxs[0] == 24


@pytest.mark.unit
def test_sample_window_short_window_duplicates_last():
    idxs = ec.sample_window_indices(center=2, n=5, t=16)
    assert len(idxs) == 16
    assert max(idxs) == 4  # never indexes past the available 5 frames
    assert idxs[-1] == 4


# --------------------------------------------------------------------------
# Label mapping
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_make_label_mapping():
    assert ec.make_label("FG_MAKE") == 1
    assert ec.make_label("3PT_MAKE") == 1
    assert ec.make_label("FREE_THROW_MAKE") == 1
    assert ec.make_label("4PT_MAKE") == 1
    assert ec.make_label("FG_MISS") == 0
    assert ec.make_label("4PT_MISS") == 0
    assert ec.make_label(None) == 0


# --------------------------------------------------------------------------
# _process_angle output shape + centring, via a stubbed cv2.
# --------------------------------------------------------------------------

class _FakeCap:
    """Minimal cv2.VideoCapture stand-in. Each read() returns a constant-
    valued frame whose value == its frame index, so we can assert which
    frames were sampled into the crop stack."""

    def __init__(self, n_frames=60, w=1920, h=1080):
        self._n = n_frames
        self._w = w
        self._h = h
        self._cur = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: self._w,
            cv2.CAP_PROP_FRAME_HEIGHT: self._h,
            cv2.CAP_PROP_FRAME_COUNT: self._n,
        }.get(prop, 0)

    def set(self, prop, val):
        self._cur = int(val)
        return True

    def grab(self):
        if self._cur >= self._n:
            return False
        self._cur += 1
        return True

    def read(self):
        if self._cur >= self._n:
            return False, None
        # BGR frame whose every pixel == current frame index (mod 256).
        val = self._cur % 256
        frame = np.full((self._h, self._w, 3), val, dtype="uint8")
        self._cur += 1
        return True, frame

    def release(self):
        pass


@pytest.mark.unit
def test_process_angle_shape_and_centering(monkeypatch):
    import cv2  # available in CI image (opencv-python); skip if truly absent
    cap = _FakeCap(n_frames=80)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)

    # 40-frame window for one play; ball nearest rim at local index 20 ->
    # global frame_idx 20. Closest-approach should centre the T=16 sample
    # there, so the middle crop's pixel value == 20.
    rows = []
    for i in range(40):
        ball = (300.0, 300.0, 10.0, 10.0)
        if i == 20:
            ball = (115.0, 55.0, 10.0, 10.0)  # nearest the rim centre
        rows.append(_row(i, ball=ball))
    windows = {("p", "FL"): rows}

    arrays, meta = ec._process_angle(Path("fake.mp4"), "FL", windows)
    assert set(arrays) == {"p_FL"}
    stack = arrays["p_FL"]
    assert stack.shape == (ec.CROP_T, ec.CROP_SIZE, ec.CROP_SIZE)
    assert stack.dtype == np.dtype("uint8")
    assert meta["n_windows"] == 1
    assert meta["n_ball_detected"] == 1
    # centre frame of the sample window == closest-approach frame index 20.
    # The constant-valued frame means the whole 64x64 crop equals 20.
    assert int(stack[ec.CROP_T // 2].mean()) == 20


# --------------------------------------------------------------------------
# Arg-parse smoke tests (no AWS, no run)
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_run_crops_batch_arg_parse():
    import run_crops_batch as rb
    assert rb.PHASE == "P1c_crops"
    with pytest.raises(SystemExit):
        rb.main([])  # neither --split nor --games


@pytest.mark.unit
def test_run_batch_default_still_p1_phase():
    import run_batch as rb1
    assert rb1.PHASE == "P1_extract"


@pytest.mark.unit
def test_p3_cropmodel_arg_parse_noop_on_empty_dir(tmp_path):
    import p3_cropmodel as cm
    # An empty crops dir -> graceful no-op returning 0 (nothing to train).
    rc = cm.main(["--crops-dir", str(tmp_path), "--epochs", "1"])
    assert rc == 0


# --------------------------------------------------------------------------
# Crop model — geometry, param count, and a real one-step train.
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_crop_model_param_count_is_small():
    import p3_cropmodel as cm
    model = cm.build_model()
    n = cm.count_params(model)
    # "a few hundred K params" — assert it stays in that band.
    assert 50_000 < n < 800_000


@pytest.mark.unit
def test_crop_model_forward_and_backward_step():
    import p3_cropmodel as cm
    cm.set_seed(42)
    import torch

    model = cm.build_model()
    device = torch.device("cpu")
    model.to(device)

    b = 6
    x = torch.rand(b, len(cm.ANGLES), cm.T, cm.SIZE, cm.SIZE)
    presence = torch.ones(b, len(cm.ANGLES))
    presence[0, 2] = 0.0  # one missing angle -> zero-pad path exercised
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    logits0 = model(x, presence)
    assert logits0.shape == (b,)
    loss0 = loss_fn(logits0, y)
    opt.zero_grad()
    loss0.backward()
    # a gradient reached the encoder (proves the full graph is connected).
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in model.encoder.parameters())
    opt.step()

    # one more step should not increase the loss on the same batch.
    logits1 = model(x, presence)
    loss1 = loss_fn(logits1, y)
    assert float(loss1.item()) <= float(loss0.item()) + 1e-3


@pytest.mark.unit
def test_auc_and_threshold_helpers():
    import p3_cropmodel as cm
    # perfectly separable -> AUC 1.0
    assert cm._auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # single class -> 0.5 by convention
    assert cm._auc([1, 1, 1], [0.1, 0.5, 0.9]) == pytest.approx(0.5)
    thr = cm.best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    rep = cm.classification_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], thr)
    assert rep["accuracy"] == pytest.approx(1.0)
