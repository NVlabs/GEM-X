# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primary-track selection and crop policy used by the deployed GEM-X demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bytetrack import ByteTracker


@dataclass
class TrackResult:
    """Full timeline; valid_span gives inclusive first/last primary observation."""

    frame_ids: np.ndarray
    boxes_xyxy: np.ndarray
    boxes_xys: np.ndarray
    track_ids: np.ndarray
    observed_primary: np.ndarray
    valid_span: tuple[int, int]
    all_person_boxes: list[np.ndarray]
    other_person_boxes: list[np.ndarray]
    raw_boxes: list[np.ndarray]
    raw_scores: list[np.ndarray]
    selection_sources: list[str]


def smooth_boxes(boxes, window=5):
    """Original replicate-padded moving average, skipped for T<=window."""
    if window <= 1 or len(boxes) <= window:
        return boxes
    values = torch.as_tensor(boxes, dtype=torch.float32)
    kernel = torch.ones(1, 1, window, dtype=values.dtype) / window
    padded = torch.nn.functional.pad(
        values.T.unsqueeze(1), (window // 2, window // 2), mode="replicate"
    )
    return torch.nn.functional.conv1d(padded, kernel).squeeze(1).T.numpy()


def square_crops(boxes, enlarge=1.2):
    """Match the deployed3:4-fit crop: enlarge*max(height,width/0.75)."""
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    widths, heights = (boxes[:, 2:] - boxes[:, :2]).T
    size = np.maximum(heights, widths / 0.75) * enlarge
    return np.concatenate((centers, size[:, None]), axis=-1)


def detect_track(frames_rgb, detector, *, bbox_enlarge=1.2, tracker_options=None):
    """Detect every RGB frame, then apply original primary/gap/crop rules.

    Filled endpoints and gaps retain the selected ID, exactly as deployed.
    All/other person boxes contain active tracker outputs before smoothing.
    The caller owns video decoding/timing; detector.detect receives BGR.
    """
    if len(frames_rgb) == 0:
        raise ValueError("Detection needs at least one decoded RGB frame")
    height, width = frames_rgb[0].shape[:2]
    tracker = ByteTracker(**(tracker_options or {}))
    per_frame, raw_boxes, raw_scores = [], [], []
    for frame in frames_rgb:
        boxes, scores = detector.detect(frame[..., ::-1].copy())
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32)
        valid = (boxes[:, 2:] >= boxes[:, :2]).all(-1)
        boxes, scores = boxes[valid], scores[valid]
        raw_boxes.append(boxes)
        raw_scores.append(scores)
        per_frame.append(tracker.update(boxes, scores))
    if not any(per_frame):
        raise ValueError("No person track was detected in the decoded video")
    selected, ids = _select_primary_track(per_frame, len(per_frame), height, width)
    primary_id = ids[0]
    observed = np.asarray([any(tid == primary_id for _, tid, _ in rows) for rows in per_frame])
    valid_indices = np.flatnonzero(observed)
    all_boxes = [
        np.asarray([box for box, _, _ in rows], dtype=np.float32).reshape(-1, 4)
        for rows in per_frame
    ]
    other_boxes = [
        np.asarray([box for box, tid, _ in rows if tid != primary_id], dtype=np.float32).reshape(
            -1, 4
        )
        for rows in per_frame
    ]
    return TrackResult(
        np.arange(len(per_frame)),
        selected,
        square_crops(smooth_boxes(selected), bbox_enlarge),
        ids,
        observed,
        (int(valid_indices[0]), int(valid_indices[-1])),
        all_boxes,
        other_boxes,
        raw_boxes,
        raw_scores,
        ["primary_track" if seen else "primary_interpolated" for seen in observed],
    )


def _select_primary_track(
    per_frame: list[list[tuple[np.ndarray, int, float]]],
    T: int,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the dominant track and extract per-frame boxes."""
    from collections import defaultdict

    # Accumulate area per track ID
    track_area: dict = defaultdict(float)
    track_frames: dict = defaultdict(list)

    for t, results in enumerate(per_frame):
        for box, tid, score in results:
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            track_area[tid] += area
            track_frames[tid].append((t, box, score))

    # Primary track = largest cumulative area
    primary_id = max(track_area, key=track_area.get)
    primary_dict = {t: box for t, box, _ in track_frames[primary_id]}

    bbx_xyxy = np.zeros((T, 4), dtype=np.float32)
    track_ids = np.full(T, -1, dtype=np.int32)

    for t in range(T):
        if t in primary_dict:
            bbx_xyxy[t] = primary_dict[t]
            track_ids[t] = primary_id

    # Interpolate gaps (frames where primary track has no observation)
    _interpolate_gaps(bbx_xyxy, track_ids, primary_id)

    # Clamp
    bbx_xyxy[:, [0, 2]] = np.clip(bbx_xyxy[:, [0, 2]], 0, W - 1)
    bbx_xyxy[:, [1, 3]] = np.clip(bbx_xyxy[:, [1, 3]], 0, H - 1)

    return bbx_xyxy, track_ids


def _interpolate_gaps(
    bbx: np.ndarray,
    ids: np.ndarray,
    primary_id: int,
):
    """Fill gaps in the primary track via linear interpolation."""
    T = len(bbx)
    valid_idx = np.where(ids == primary_id)[0]

    # Fill leading frames with first valid box
    if valid_idx[0] > 0:
        bbx[: valid_idx[0]] = bbx[valid_idx[0]]
        ids[: valid_idx[0]] = primary_id

    # Fill trailing frames with last valid box
    if valid_idx[-1] < T - 1:
        bbx[valid_idx[-1] + 1 :] = bbx[valid_idx[-1]]
        ids[valid_idx[-1] + 1 :] = primary_id

    # Interpolate internal gaps
    for k in range(len(valid_idx) - 1):
        a, b = valid_idx[k], valid_idx[k + 1]
        if b - a > 1:
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                bbx[t] = (1 - alpha) * bbx[a] + alpha * bbx[b]
                ids[t] = primary_id
