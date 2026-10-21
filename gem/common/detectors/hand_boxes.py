# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Offline smoothing of native square bbox tracks before Hand Pose cropping.

Association and gap filling stay in the native tracker. This filter rejects
isolated spikes, smooths centers and log sizes, and limits size changes over a
complete track segment. The size limiter uses future frames; callers must not
split a clip into independent batches. Missing boxes remain missing.
"""

from dataclasses import dataclass, fields, replace
from math import isfinite, log1p
from numbers import Real

import numpy as np

from .hand_tracking import PAD, HandTrackResult

REFERENCE_FPS = 20.0


@dataclass(frozen=True)
class BBoxSmoothingConfig:
    """EMA weights are defined at 20 fps; smaller weights smooth more strongly."""

    target_size_scale: float = 0.8
    size_alpha_grow: float = 0.12
    size_alpha_shrink: float = 0.06
    max_size_log_rate_per_second: float = REFERENCE_FPS * log1p(0.01)
    center_alpha_slow: float = 0.5
    center_alpha_fast: float = 0.9
    speed_alpha: float = 0.35
    slow_speed_sides_per_second: float = 0.15
    fast_speed_sides_per_second: float = 1.5
    max_center_lag_fraction: float = 0.06
    spike_center_fraction: float = 0.05
    spike_neighbor_fraction: float = 0.05
    spike_log_size: float = 0.08
    spike_neighbor_log_size: float = 0.04
    reset_gap_seconds: float = 0.25
    reset_jump_fraction: float = 0.75
    coverage_margin_fraction: float = 0.015

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        for name in (
            "target_size_scale",
            "size_alpha_grow",
            "size_alpha_shrink",
            "center_alpha_slow",
            "center_alpha_fast",
            "speed_alpha",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.center_alpha_slow > self.center_alpha_fast:
            raise ValueError("center_alpha_slow must not exceed center_alpha_fast")
        if self.slow_speed_sides_per_second >= self.fast_speed_sides_per_second:
            raise ValueError("fast_speed_sides_per_second must exceed slow_speed_sides_per_second")
        if self.reset_gap_seconds <= 0 or self.reset_jump_fraction <= 0:
            raise ValueError("reset_gap_seconds and reset_jump_fraction must be positive")


def validate_frame_times(frame_times: np.ndarray, count: int) -> np.ndarray:
    """Require finite, strictly increasing timestamps aligned with the boxes."""
    times = np.asarray(frame_times, dtype=np.float64)
    if times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("BBox smoothing requires one finite timestamp per frame")
    if np.any(np.diff(times) <= 0):
        raise ValueError("BBox smoothing frame timestamps must be strictly increasing")
    return times


def _alpha(weight: float, dt: float) -> float:
    return 1.0 - (1.0 - weight) ** (dt * REFERENCE_FPS)


def _reject_isolated_spikes(
    boxes: np.ndarray, observed: np.ndarray, config: BBoxSmoothingConfig
) -> np.ndarray:
    target = boxes.copy()
    for index in range(1, len(boxes) - 1):
        if not observed[index - 1 : index + 2].all():
            continue
        before, current, after = boxes[index - 1 : index + 2]
        scale = max(float(np.median([before[2], current[2], after[2]])), 1.0)
        midpoint = (before[:2] + after[:2]) / 2
        if (
            np.linalg.norm(after[:2] - before[:2]) / scale < config.spike_neighbor_fraction
            and np.linalg.norm(current[:2] - midpoint) / scale > config.spike_center_fraction
        ):
            target[index, :2] = midpoint
        log_sizes = np.log([before[2], current[2], after[2]])
        if (
            abs(log_sizes[2] - log_sizes[0]) < config.spike_neighbor_log_size
            and abs(log_sizes[1] - (log_sizes[0] + log_sizes[2]) / 2) > config.spike_log_size
        ):
            target[index, 2] = np.exp((log_sizes[0] + log_sizes[2]) / 2)
    return target


def _limit_size_changes(
    boxes: np.ndarray, times: np.ndarray, resets: list[int], log_rate: float
) -> None:
    """Raise sizes to the smallest envelope satisfying the rate bound.

    The forward pass slows contraction and the backward pass anticipates growth.
    Raising rather than clipping sizes preserves the earlier crop guard. Reset
    transitions are exempt, and segments never cross a missing box.
    """
    valid = np.isfinite(boxes).all(axis=1)
    for start, stop in zip(resets, resets[1:] + [len(boxes)]):
        indices = np.arange(start, stop)
        indices = indices[valid[indices]]
        logs = np.log(boxes[indices, 2])
        bounds = log_rate * np.diff(times[indices])
        for index in range(1, len(logs)):
            logs[index] = max(logs[index], logs[index - 1] - bounds[index - 1])
        for index in range(len(logs) - 2, -1, -1):
            logs[index] = max(logs[index], logs[index + 1] - bounds[index])
        boxes[indices, 2] = np.exp(logs)


def smooth_bbox_track(
    boxes: np.ndarray,
    frame_times: np.ndarray,
    observed: np.ndarray,
    config: BBoxSmoothingConfig,
) -> np.ndarray:
    """Filter a complete (N, 3) [cx, cy, side] track without modifying inputs.

    ``observed`` marks real native assignments before interpolation. NaN or
    nonpositive boxes are missing. State resets at missing boxes, a large center
    jump, or reacquisition after a long detection gap.

    The crop guard encloses the central 1/PAD of the incoming native square plus
    a margin. This is a heuristic, not a guarantee of finger coverage: native
    geometry already includes size freezing, rate limits and a minimum size.
    The guard deliberately uses the original box even if its size was rejected
    as a spike, prioritizing coverage over tightness in ambiguous frames.
    """
    raw = np.asarray(boxes, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Expected bbox array with shape (N, 3)")
    times = validate_frame_times(frame_times, len(raw))
    valid = np.isfinite(raw).all(axis=1) & (raw[:, 2] > 0)
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != (len(raw),):
        raise ValueError("observed must match the number of frames")
    observed = observed & valid
    target = _reject_isolated_spikes(raw, observed, config)
    result = np.full_like(raw, np.nan)
    state = previous_target = None
    filtered_velocity = np.zeros(2)
    absent_seconds = 0.0
    resets = []
    for index, box in enumerate(target):
        dt = times[index] - times[index - 1] if index else 1.0 / REFERENCE_FPS
        reacquired = observed[index] and absent_seconds >= config.reset_gap_seconds - 1e-9
        absent_seconds = 0.0 if observed[index] else absent_seconds + dt
        if not valid[index]:
            state = previous_target = None
            continue
        jump = previous_target is not None and (
            np.linalg.norm(box[:2] - previous_target[:2]) > config.reset_jump_fraction * box[2]
        )
        if state is None or reacquired or jump:
            state = box.copy()
            state[2] *= config.target_size_scale
            filtered_velocity[:] = 0
            resets.append(index)
        else:
            velocity = (box[:2] - previous_target[:2]) / (max(box[2], 1.0) * dt)
            filtered_velocity += _alpha(config.speed_alpha, dt) * (velocity - filtered_velocity)
            speed = np.linalg.norm(filtered_velocity)
            blend = np.clip(
                (speed - config.slow_speed_sides_per_second)
                / (config.fast_speed_sides_per_second - config.slow_speed_sides_per_second),
                0,
                1,
            )
            weight = config.center_alpha_slow + blend * (
                config.center_alpha_fast - config.center_alpha_slow
            )
            center = state[:2] + _alpha(weight, dt) * (box[:2] - state[:2])
            offset = center - raw[index, :2]
            limit = config.max_center_lag_fraction * raw[index, 2]
            distance = np.linalg.norm(offset)
            if distance > limit:
                center = raw[index, :2] + offset * (limit / distance)
            target_size = box[2] * config.target_size_scale
            weight = config.size_alpha_grow if target_size > state[2] else config.size_alpha_shrink
            alpha = _alpha(weight, dt)
            size = np.exp((1 - alpha) * np.log(state[2]) + alpha * np.log(target_size))
            state = np.array([center[0], center[1], size])
        required_size = (
            raw[index, 2] / PAD
            + 2 * np.max(np.abs(state[:2] - raw[index, :2]))
            + 2 * config.coverage_margin_fraction * raw[index, 2]
        )
        state[2] = max(state[2], required_size)
        result[index] = state
        previous_target = box
    _limit_size_changes(result, times, resets, config.max_size_log_rate_per_second)
    return result.astype(np.float32)


def smooth_hand_boxes(
    tracks: HandTrackResult,
    frame_times: np.ndarray,
    config: BBoxSmoothingConfig | None = None,
    *,
    enforce_lr_order: bool = True,
) -> HandTrackResult:
    """Filter both native tracks before selecting sides or sparse emission.

    Keep the association, original observations and emission policy. Only
    crossings introduced by independent smoothing are corrected, by moving
    each x-center toward its native center without changing hand identities.
    The input tracks are not modified.
    """
    config = BBoxSmoothingConfig() if config is None else config
    boxes = {
        side: (
            smooth_bbox_track(tracks.boxes_xys[side], frame_times, tracks.raw_present[side], config)
            if tracks.raw_present[side].any()
            else tracks.boxes_xys[side].copy()
        )
        for side in ("left", "right")
    }
    if enforce_lr_order and all(tracks.raw_present[side].any() for side in ("left", "right")):
        left, right = boxes["left"], boxes["right"]
        native_left = tracks.boxes_xys["left"][:, 0]
        native_right = tracks.boxes_xys["right"][:, 0]
        crossed = (native_left <= native_right) & (left[:, 0] > right[:, 0])
        boundary = np.clip(
            (left[crossed, 0] + right[crossed, 0]) / 2,
            native_left[crossed],
            native_right[crossed],
        )
        left[crossed, 0] = right[crossed, 0] = boundary
    emitted = {
        side: tracks.emitted[side]
        & np.isfinite(boxes[side]).all(axis=-1)
        & (boxes[side][:, 2] > 1.0)
        for side in ("left", "right")
    }
    return replace(tracks, boxes_xys=boxes, emitted=emitted)
