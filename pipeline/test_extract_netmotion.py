"""
Local validation for the P1b net-motion pass — runs WITHOUT AWS, without
any real video, and without re-running YOLO.

Covers:
  * net_roi_from_rim geometry (placed below the iron, clipped to frame)
  * rim-box stabilisation (interpolation/hold + median smoothing, rim_ok)
  * net-motion metrics on synthetic ROIs: a moving bright blob yields high
    framediff_energy / flow_mag; a static ROI is ~0
  * arg-parse smoke test of BOTH batch entrypoints
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_netmotion as nm  # noqa: E402


# --------------------------------------------------------------------------
# ROI geometry
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_net_roi_below_rim_and_centered():
    # rim box: top-left (100,50), w=40, h=20  -> cx=120, bottom=70
    roi = nm.net_roi_from_rim((100.0, 50.0, 40.0, 20.0), 1920, 1080)
    assert roi is not None
    rx, ry, rw, rh = roi
    # x spans cx +/- 0.6*w = 120 +/- 24 -> [96,144], w=48
    assert rx == 96 and rw == 48
    # y starts at rim_bottom=70, extends 1.5*h=30 -> [70,100], h=30
    assert ry == 70 and rh == 30
    # ROI is strictly below the rim bottom edge
    assert ry >= 50 + 20


@pytest.mark.unit
def test_net_roi_clipped_to_frame():
    # rim near the right edge with the net band running off the bottom:
    # ROI must clip on both axes yet stay a valid (>=2px) box.
    roi = nm.net_roi_from_rim((1890.0, 1010.0, 40.0, 20.0), 1920, 1080)
    assert roi is not None
    rx, ry, rw, rh = roi
    assert rx + rw <= 1920          # clipped on x (cx+24 = 1934 -> 1920)
    assert ry + rh <= 1080          # clipped on y (band 1030..1060 ok)
    assert rw >= 2 and rh >= 2


@pytest.mark.unit
def test_net_roi_degenerate_returns_none():
    # rim entirely below the frame -> nothing left after clipping.
    assert nm.net_roi_from_rim((10.0, 2000.0, 40.0, 20.0), 1920, 1080) is None


# --------------------------------------------------------------------------
# Rim-box stabilisation
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_interp_hold_fills_gaps_and_edges():
    out = nm._interp_hold([None, None, 10.0, None, 20.0, None])
    assert out == [10.0, 10.0, 10.0, 15.0, 20.0, 20.0]


@pytest.mark.unit
def test_stabilized_rim_boxes_marks_observed_frames():
    rows = []
    for i in range(11):
        # middle frame has a null rim -> rim_ok False there, interpolated.
        has = i != 5
        rows.append({
            "play_id": "p", "angle": "FL", "frame_idx": i,
            "timestamp": float(i), "classification": "FG_MAKE",
            "rim_x": 100.0 if has else None,
            "rim_y": 50.0 if has else None,
            "rim_w": 40.0 if has else None,
            "rim_h": 20.0 if has else None,
        })
    boxes, rim_ok = nm.stabilized_rim_boxes(rows)
    assert len(boxes) == 11 and len(rim_ok) == 11
    assert rim_ok[0] is True and rim_ok[5] is False
    # smoothing of a constant box returns the same constant box.
    assert boxes[0] == pytest.approx((100.0, 50.0, 40.0, 20.0))


@pytest.mark.unit
def test_stabilized_rim_all_null_yields_none():
    rows = [{"play_id": "p", "angle": "FL", "frame_idx": i,
             "timestamp": float(i), "classification": "FG_MISS",
             "rim_x": None, "rim_y": None, "rim_w": None, "rim_h": None}
            for i in range(5)]
    boxes, rim_ok = nm.stabilized_rim_boxes(rows)
    assert boxes == [None] * 5
    assert rim_ok == [False] * 5


# --------------------------------------------------------------------------
# Net-motion metrics — moving blob vs static ROI
# --------------------------------------------------------------------------

def _blob_frame(cx: int, size: int = 80) -> np.ndarray:
    img = np.zeros((size, size), dtype="uint8")
    img[30:50, cx:cx + 12] = 255  # a bright bar
    return img.astype("float32")


@pytest.mark.unit
def test_static_roi_has_near_zero_energy():
    a = _blob_frame(10)
    fd, flow = nm.net_motion_metrics(a.copy(), a.copy())
    assert fd == pytest.approx(0.0, abs=1e-6)
    assert flow == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_moving_blob_has_high_energy():
    prev = _blob_frame(10)
    cur = _blob_frame(35)  # blob jumped 25 px
    fd, flow = nm.net_motion_metrics(prev, cur)
    static_fd, static_flow = nm.net_motion_metrics(prev, prev.copy())
    assert fd > 5.0
    assert fd > static_fd + 1.0
    assert flow > static_flow


@pytest.mark.unit
def test_roi_temporal_variance_zero_when_constant():
    assert nm.roi_temporal_variance([12.0, 12.0, 12.0]) == pytest.approx(0.0)
    assert nm.roi_temporal_variance([0.0, 10.0, 20.0]) > 0.0


@pytest.mark.unit
def test_downscale_roi_width_is_target():
    g = np.full((200, 320), 128, dtype="float32")
    small = nm._downscale_roi(g, target_w=64)
    assert small.shape[1] == 64
    assert small.dtype == np.dtype("float32")


# --------------------------------------------------------------------------
# Window grouping
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_group_windows_sorts_by_frame():
    rows = [
        {"play_id": "p1", "angle": "FL", "frame_idx": 3},
        {"play_id": "p1", "angle": "FL", "frame_idx": 1},
        {"play_id": "p1", "angle": "NR", "frame_idx": 2},
    ]
    g = nm._group_windows(rows)
    assert set(g) == {("p1", "FL"), ("p1", "NR")}
    assert [r["frame_idx"] for r in g[("p1", "FL")]] == [1, 3]


# --------------------------------------------------------------------------
# Arg-parse smoke tests for BOTH batch entrypoints (no AWS, no run)
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_run_netmotion_batch_arg_parse():
    import run_netmotion_batch as rb
    assert rb.PHASE == "P1b_netmotion"
    with pytest.raises(SystemExit):
        rb.main([])  # neither --split nor --games


@pytest.mark.unit
def test_run_batch_still_importable_and_p1_phase():
    import run_batch as rb1
    assert rb1.PHASE == "P1_extract"
    with pytest.raises(SystemExit):
        rb1.main([])
