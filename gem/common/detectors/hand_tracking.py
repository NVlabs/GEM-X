# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-clip RT-DETR hand association and crop stabilization.

The default screen-order prior is intended for egocentric footage: when both
hands are observed, the smaller-x detection becomes ``left``. This is not multi-person tracking.
External, already assigned hand tracks should bypass this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hand import HandDetection


THR = 0.4
THR_RESCUE = 0.15
RESCUE_IOU = 0.35
W_DET, W_IOU = 1.0, 1.5
PAD = 1.4
MIN_SIZE_FRAC = 0.16
MAX_CENTER_SPEED = 90.0
MAX_SIZE_RATIO = 1.12
CENTER_STILL = 8.0


@dataclass
class HandTrackResult:
    """Aligned full-timeline boxes, observation masks and emitted-frame masks.

    Each dictionary is keyed by ``left`` / ``right``. ``boxes_xys`` contains
    float32 ``[T,3]`` pixel boxes. A side never observed has zero boxes and an
    all-false ``emitted`` mask. Consumers must use ``emitted`` to select crops.
    ``raw_present`` includes detections rescued by continuity, before filling.
    """

    boxes_xys: dict[str, np.ndarray]
    raw_present: dict[str, np.ndarray]
    emitted: dict[str, np.ndarray]
    stats: dict[str, int]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    aw, ah = a[2], a[3]
    bw, bh = b[2], b[3]
    ax0, ay0, ax1, ay1 = a[0] - aw / 2, a[1] - ah / 2, a[0] + aw / 2, a[1] + ah / 2
    bx0, by0, bx1, by1 = b[0] - bw / 2, b[1] - bh / 2, b[0] + bw / 2, b[1] + bh / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _pool_candidates(record, previous_left, previous_right):
    pool = []
    # Class groups and stable score ordering deliberately match the source:
    # ties can affect the small assignment problem below.
    for key in ("L", "R"):
        for candidate in record[key]:
            duplicate = next((p for p in pool if _iou(p, candidate) > 0.9), None)
            if duplicate is None:
                pool.append(list(candidate))
            elif candidate[4] > duplicate[4]:
                # Preserve the original candidate geometry when merging labels.
                duplicate[4] = candidate[4]
                duplicate[5] = candidate[5]
    usable = [
        candidate
        for candidate in pool
        if candidate[4] > THR
        or (
            candidate[4] >= THR_RESCUE
            and (
                _iou(candidate, previous_left) >= RESCUE_IOU
                or _iou(candidate, previous_right) >= RESCUE_IOU
            )
        )
    ]
    usable.sort(key=lambda candidate: -candidate[4])
    return usable[:5]


def _assign(records):
    selected = {"left": ([], []), "right": ([], [])}
    previous_left = previous_right = None
    stats = {"rescued": 0, "label_overridden": 0, "crossed": 0}
    for record in records:
        usable = _pool_candidates(record, previous_left, previous_right)
        left = right = None
        if not usable:
            pass
        elif previous_left is None and previous_right is None:
            if len(usable) == 1:
                if usable[0][5] == 1:
                    left = usable[0]
                else:
                    right = usable[0]
            else:
                left, right = sorted(usable[:2], key=lambda candidate: candidate[0])
        else:
            best = None
            options = usable + [None]
            for a in options:
                for b in options:
                    if a is not None and b is not None and a is b:
                        continue
                    score = 0.0
                    if a is not None:
                        score += W_DET * a[4] + W_IOU * _iou(a, previous_left)
                    if b is not None:
                        score += W_DET * b[4] + W_IOU * _iou(b, previous_right)
                    if a is None and b is None:
                        score = -1e9
                    if best is None or score > best[0]:
                        best = (score, a, b)
            left, right = best[1], best[2]
        detector_left = record["L"][0] if record["L"] else None
        if left is not None and detector_left is not None and _iou(left, detector_left) < 0.5:
            stats["label_overridden"] += 1
        if left is not None and right is not None and left[0] > right[0]:
            stats["crossed"] += 1
        for side, candidate in (("left", left), ("right", right)):
            if candidate is None:
                selected[side][0].append([0.0, 0.0, 0.0])
                selected[side][1].append(False)
            else:
                if candidate[4] <= THR:
                    stats["rescued"] += 1
                selected[side][0].append([candidate[0], candidate[1], max(candidate[2:4]) * PAD])
                selected[side][1].append(True)
        if left is not None:
            previous_left = left
        if right is not None:
            previous_right = right
    return {
        side: (np.asarray(boxes, np.float32).reshape(-1, 3), np.asarray(present, bool))
        for side, (boxes, present) in selected.items()
    }, stats


def _stabilize(boxes, present, min_size):
    boxes = boxes.astype(np.float64).copy()
    indices = np.flatnonzero(present)
    if len(indices) == 0:
        return boxes.astype(np.float32)
    # All gaps, regardless of duration, use both adjacent observations. End
    # gaps hold the nearest observation. This requires the complete clip.
    boxes[: indices[0]] = boxes[indices[0]]
    boxes[indices[-1] + 1 :] = boxes[indices[-1]]
    for a, b in zip(indices, indices[1:]):
        for t in range(a + 1, b):
            weight = (t - a) / (b - a)
            boxes[t] = (1 - weight) * boxes[a] + weight * boxes[b]
    boxes[:, 2] = np.maximum(boxes[:, 2], min_size)
    anchor = boxes[0, :2].copy()
    for t in range(1, len(boxes)):
        previous_center = boxes[t - 1, :2].copy()
        previous_size = boxes[t - 1, 2]
        displacement = boxes[t, :2] - previous_center
        distance = float(np.hypot(displacement[0], displacement[1]))
        anchor_displacement = boxes[t, :2] - anchor
        still = float(np.hypot(anchor_displacement[0], anchor_displacement[1])) <= CENTER_STILL
        if not still:
            anchor = boxes[t, :2].copy()
        if still:
            if previous_size > 0 and abs(boxes[t, 2] - previous_size) > 1e-6:
                boxes[t, 2] = previous_size
        elif previous_size > 0:
            boxes[t, 2] = np.clip(
                boxes[t, 2], previous_size / MAX_SIZE_RATIO, previous_size * MAX_SIZE_RATIO
            )
        if distance > MAX_CENTER_SPEED:
            boxes[t, :2] = previous_center + displacement * (MAX_CENTER_SPEED / distance)
    return boxes.astype(np.float32)


def track_hand_detections(
    per_frame: list[list[HandDetection]],
    image_width: int,
    *,
    enforce_lr_order: bool = True,
    drop_gap_filled: bool = False,
) -> HandTrackResult:
    """Assign and stabilize detections on the complete decoded frame grid.

    Feed raw rectangular RT-DETR boxes with ``score_thr=0.15``, no padding,
    square conversion or top-one selection. Detection may run in batches; call
    this function once after collecting every frame's candidates. Constants are
    in pixels/frames, matching the original tracker without FPS rescaling.

    ``enforce_lr_order=False`` preserves continuity through later crossings;
    simultaneous two-hand bootstrap still uses x-order. A lone bootstrap uses
    the detector's left/right class. ``drop_gap_filled`` only changes emission,
    not association or stabilization. Candidates with ``side=None`` are excluded
    at this boundary; the GA tracker otherwise misassigns unused classes to the
    right hand. This function never changes the inputs.
    """
    records = []
    for detections in per_frame:
        record = {"L": [], "R": []}
        for detection in sorted(detections, key=lambda value: -value.score):
            if detection.side is None:
                continue
            x1, y1, x2, y2 = detection.bbox_xyxy
            candidate = [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
                detection.score,
                detection.label_id,
            ]
            record["L" if detection.label_id == 1 else "R"].append(candidate)
        records.append(record)
    tracks, stats = _assign(records)
    if enforce_lr_order:
        left, left_present = tracks["left"]
        right, right_present = tracks["right"]
        # Settle identities before rate limits. Never compare an absent side's
        # zero sentinel against an observed hand or reorder after stabilization.
        swap = left_present & right_present & (left[:, 0] > right[:, 0])
        left[swap], right[swap] = right[swap].copy(), left[swap].copy()
    boxes_xys, raw_present, emitted = {}, {}, {}
    for side, (boxes, present) in tracks.items():
        boxes = _stabilize(boxes, present, MIN_SIZE_FRAC * image_width)
        boxes_xys[side] = boxes
        raw_present[side] = present
        emitted[side] = (
            present.copy() if drop_gap_filled else np.full(len(present), present.any(), bool)
        )
        emitted[side] &= np.isfinite(boxes).all(axis=-1) & (boxes[:, 2] > 1.0)
    return HandTrackResult(boxes_xys, raw_present, emitted, stats)
