# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable numerical contracts for whole-clip hand bbox smoothing."""

import copy

import numpy as np
import pytest

from gem.common.detectors.hand_boxes import (
    BBoxSmoothingConfig,
    smooth_bbox_track,
    smooth_hand_boxes,
)
from gem.common.detectors.hand_tracking import PAD, HandTrackResult


def test_stationary_track_is_tighter_and_empty_track_stays_empty():
    boxes = np.tile([200.0, 100.0, 100.0], (20, 1))
    actual = smooth_bbox_track(boxes, np.arange(20) / 20, np.ones(20, bool), BBoxSmoothingConfig())
    np.testing.assert_array_equal(actual, np.tile([200.0, 100.0, 80.0], (20, 1)))
    assert actual.dtype == np.float32
    empty = smooth_bbox_track(np.empty((0, 3)), [], [], BBoxSmoothingConfig())
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32


def test_jitter_is_reduced_with_bounded_lag_and_native_coverage():
    random = np.random.default_rng(12)
    boxes = np.tile([200.0, 100.0, 100.0], (120, 1))
    boxes[:, :2] += random.normal(0, 2, (120, 2))
    boxes[40, :2] += [9, 0]
    boxes[80, 2] = 125
    times = np.arange(len(boxes)) / 20
    observed = np.ones(len(boxes), bool)
    before = [array.copy() for array in (boxes, times, observed)]
    config = BBoxSmoothingConfig()
    actual = smooth_bbox_track(boxes, times, observed, config)

    assert np.std(actual[:, 0]) < np.std(boxes[:, 0])
    offsets = actual[:, :2] - boxes[:, :2]
    assert np.all(
        np.linalg.norm(offsets, axis=1) <= config.max_center_lag_fraction * boxes[:, 2] + 2e-5
    )
    required = (
        boxes[:, 2] / PAD
        + 2 * np.max(np.abs(offsets), axis=1)
        + 2 * config.coverage_margin_fraction * boxes[:, 2]
    )
    assert np.all(actual[:, 2] >= required - 2e-5)
    for original, unchanged in zip((boxes, times, observed), before):
        np.testing.assert_array_equal(original, unchanged)


def test_size_envelope_anticipates_growth_and_bounds_both_directions():
    times = np.cumsum(np.resize([0.025, 0.05, 0.08], 100))
    boxes = np.tile([200.0, 100.0, 100.0], (len(times), 1))
    boxes[40:65, 2] = 180
    config = BBoxSmoothingConfig()
    actual = smooth_bbox_track(boxes, times, np.ones(len(times), bool), config)
    rates = np.abs(np.diff(np.log(actual[:, 2].astype(np.float64)))) / np.diff(times)
    assert np.all(rates <= config.max_size_log_rate_per_second + 1e-5)
    assert actual[39, 2] > 100 * config.target_size_scale
    assert actual[65, 2] > 100 * config.target_size_scale


def test_ema_uses_elapsed_seconds_at_20_and_40_fps():
    config = BBoxSmoothingConfig(center_alpha_slow=0.5, center_alpha_fast=0.5)
    outputs = []
    for fps in (20, 40):
        times = np.arange(fps + 1) / fps
        boxes = np.tile([201.0, 100.0, 105.0], (len(times), 1))
        boxes[0] = [200, 100, 100]
        outputs.append(smooth_bbox_track(boxes, times, np.ones(len(times), bool), config))
    np.testing.assert_allclose(outputs[0], outputs[1][::2], rtol=0, atol=2e-5)


def test_interpolated_gap_stays_dense_and_reacquisition_resets_state():
    boxes = np.column_stack([200 + np.arange(9) * 2, np.full(9, 100), np.full(9, 100)])
    observed = np.array([True, False, False, False, False, False, True, True, True])
    actual = smooth_bbox_track(boxes, np.arange(9) / 20, observed, BBoxSmoothingConfig())
    assert np.isfinite(actual).all()
    assert actual[5, 0] < boxes[5, 0]
    np.testing.assert_array_equal(actual[6], [212, 100, 80])


def test_missing_boxes_split_envelope_and_reset_next_valid_box():
    boxes = np.array(
        [
            [200, 100, 100],
            [202, 100, 100],
            [np.nan, np.nan, np.nan],
            [250, 100, 200],
            [252, 100, 200],
            [260, 100, 0],
            [300, 100, 300],
        ]
    )
    actual = smooth_bbox_track(boxes, np.arange(7) / 20, np.ones(7, bool), BBoxSmoothingConfig())
    assert np.isnan(actual[[2, 5]]).all()
    np.testing.assert_array_equal(
        actual[[0, 3, 6]], [[200, 100, 80], [250, 100, 160], [300, 100, 240]]
    )


def test_large_center_jump_resets_instead_of_dragging_previous_crop():
    boxes = np.array([[200, 100, 100], [202, 100, 100], [400, 100, 100], [402, 100, 100]])
    actual = smooth_bbox_track(boxes, np.arange(4) / 20, np.ones(4, bool), BBoxSmoothingConfig())
    np.testing.assert_array_equal(actual[2], [400, 100, 80])
    assert actual[3, 0] < boxes[3, 0]


@pytest.mark.parametrize("times", [[0, 0], [0.1, 0], [0, np.nan], [0]])
def test_timestamp_boundary_rejects_unaligned_or_nonincreasing_times(times):
    with pytest.raises(ValueError, match="timestamp"):
        smooth_bbox_track(
            np.tile([200, 100, 100], (2, 1)), times, [True, True], BBoxSmoothingConfig()
        )


def make_tracks(left, right, *, left_present=None, right_present=None, sparse=False):
    boxes = {"left": np.asarray(left, np.float32), "right": np.asarray(right, np.float32)}
    count = len(left)
    present = {
        "left": np.ones(count, bool) if left_present is None else np.asarray(left_present, bool),
        "right": np.ones(count, bool) if right_present is None else np.asarray(right_present, bool),
    }
    emitted = {
        side: mask.copy() if sparse else np.full(count, mask.any())
        for side, mask in present.items()
    }
    return HandTrackResult(
        boxes, present, emitted, {"rescued": 2, "label_overridden": 1, "crossed": 0}
    )


def test_hand_filter_preserves_masks_stats_absent_side_and_inputs():
    left = [[200 + index * 2, 100, 100] for index in range(6)]
    tracks = make_tracks(
        left,
        np.zeros((6, 3)),
        left_present=[True, False, False, True, False, True],
        right_present=np.zeros(6, bool),
        sparse=True,
    )
    before = copy.deepcopy(tracks)
    actual = smooth_hand_boxes(tracks, np.arange(6) / 20)
    assert actual.stats == before.stats
    assert np.isfinite(actual.boxes_xys["left"]).all()
    assert actual.boxes_xys["left"][1, 0] < left[1][0]
    np.testing.assert_array_equal(actual.boxes_xys["right"], before.boxes_xys["right"])
    for side in ("left", "right"):
        np.testing.assert_array_equal(actual.raw_present[side], before.raw_present[side])
        np.testing.assert_array_equal(actual.emitted[side], before.emitted[side])
        np.testing.assert_array_equal(tracks.boxes_xys[side], before.boxes_xys[side])
        np.testing.assert_array_equal(tracks.raw_present[side], before.raw_present[side])
        np.testing.assert_array_equal(tracks.emitted[side], before.emitted[side])


def test_order_correction_moves_only_smoothed_crossing_centers():
    tracks = make_tracks([[100, 50, 100], [160, 50, 100]], [[120, 70, 300], [161, 70, 300]])
    times = [0, 0.05]
    unordered = smooth_hand_boxes(tracks, times, enforce_lr_order=False)
    assert unordered.boxes_xys["left"][1, 0] > unordered.boxes_xys["right"][1, 0]
    ordered = smooth_hand_boxes(tracks, times)
    for side in ("left", "right"):
        assert ordered.boxes_xys[side][1, 0] == 160
        np.testing.assert_array_equal(
            ordered.boxes_xys[side][:, 1:], unordered.boxes_xys[side][:, 1:]
        )
        np.testing.assert_array_equal(ordered.boxes_xys[side][0], unordered.boxes_xys[side][0])


def test_existing_native_crossing_keeps_identity():
    tracks = make_tracks([[100, 50, 100], [160, 50, 100]], [[200, 70, 100], [140, 70, 100]])
    ordered = smooth_hand_boxes(tracks, [0, 0.05])
    unordered = smooth_hand_boxes(tracks, [0, 0.05], enforce_lr_order=False)
    assert ordered.boxes_xys["left"][1, 0] > ordered.boxes_xys["right"][1, 0]
    for side in ("left", "right"):
        np.testing.assert_array_equal(ordered.boxes_xys[side], unordered.boxes_xys[side])
