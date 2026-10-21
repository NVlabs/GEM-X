# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Offline, motion-adaptive smoothing of SOMA parameters.

Translations are in full-image camera coordinates (meters); ``poses`` are
the exported SOMA wrist/finger rotation matrices, after right-hand mirroring.
The result adapter decodes geometry again with the matching physical hand
layer and projects through the original camera. Vertices are never averaged.

Each contiguous run of finite predictions is independent. The filter needs
timestamps in seconds and both temporal directions. Defaults are experimental;
they are not tuned to a particular motion capture dataset.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Same positive-depth threshold as the inference projection contract.
MIN_CAMERA_DEPTH = 1e-4


@dataclass(frozen=True)
class AdaptiveSmoothingConfig:
    """Physical-time widths and noise scales, independent of video frame rate."""

    max_sigma_seconds: float = 2.0 / 30.0
    shape_sigma_seconds: float = 0.2
    translation_speed_mps: float = 0.25
    angular_speed_radps: float = np.pi
    translation_noise_floor_m: float = 0.0015
    angular_noise_floor_rad: float = np.deg2rad(0.5)
    noise_radius_seconds: float = 0.2
    fit_tolerance: float = 1.25
    # A larger interval splits a track, even if both endpoints have predictions.
    max_gap_seconds: float = 0.15

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")


def validate_frame_times(frame_times, count):
    times = np.asarray(frame_times, dtype=np.float64)
    if times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("frame_times must contain one finite timestamp per image")
    if count > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("frame_times must be strictly increasing (seconds)")
    return times


def _motion_sigma(edge_speed, times, reference_speed, max_sigma):
    """Use speed magnitudes on both sides, so a reversal cannot cancel speed.

    A short centered average anticipates nearby motion. Taking the maximum of
    that estimate and the two incident edges responds immediately to a burst
    or reversal. This is deliberately conservative: isolated prediction
    outliers can also open the gate; no confidence score is invented here.
    """
    speed = np.zeros((len(times), *edge_speed.shape[1:]), dtype=np.float64)
    speed[:-1] = np.maximum(speed[:-1], edge_speed)
    speed[1:] = np.maximum(speed[1:], edge_speed)
    # Look ahead/behind a short physical-time interval to protect motion onset
    # and the exact turning point; this is independent for each joint.
    guarded = speed.copy()
    for i, time in enumerate(times):
        lo = np.searchsorted(times, time - max_sigma / 2, side="left")
        hi = np.searchsorted(times, time + max_sigma / 2, side="right")
        guarded[i] = np.maximum(speed[i], speed[lo:hi].mean(axis=0))
    return max_sigma / (1.0 + (guarded / reference_speed) ** 2)


def _neighbors(times, index, sigma):
    # Widths below one sample naturally become an identity operation. Avoid
    # forcing a minimum number of neighboring frames during fast motion.
    radius = 3.0 * float(np.max(sigma))
    lo = np.searchsorted(times, times[index] - radius, side="left")
    hi = np.searchsorted(times, times[index] + radius, side="right")
    delta = times[lo:hi] - times[index]
    return lo, hi, delta


def _short_translation(values, times, config):
    """Centered Gaussian with a width driven by camera-space linear speed."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values.copy()
    speed = np.linalg.norm(np.diff(values, axis=0), axis=-1) / np.diff(times)
    sigma = _motion_sigma(speed, times, config.translation_speed_mps, config.max_sigma_seconds)
    out = values.copy()
    for i in range(len(times)):
        lo, hi, delta = _neighbors(times, i, sigma[i])
        weights = np.exp(-0.5 * (delta / max(sigma[i], 1e-12)) ** 2)
        out[i] = weights @ values[lo:hi] / weights.sum()
    return out


def _short_rotations(values, times, config):
    """Filter each joint in its current rotation's tangent space.

    R_out = R_t Exp(sum_i w_i Log(R_t.T R_i) / sum_i w_i).
    This is a local SO(3) mean, not an element-wise matrix/Euler average. The
    principal relative rotations handle quaternion sign and +/-180 degree
    wraparound. Fast motion narrows support before computing the mean.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values.copy()
    relative = np.swapaxes(values[:-1], -1, -2) @ values[1:]
    angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    speed = angles.reshape(len(times) - 1, -1) / np.diff(times)[:, None]
    sigma = _motion_sigma(speed, times, config.angular_speed_radps, config.max_sigma_seconds)
    out = values.copy()
    for i in range(len(times)):
        lo, hi, delta = _neighbors(times, i, sigma[i])
        weights = np.exp(-0.5 * (delta[:, None] / np.maximum(sigma[i], 1e-12)) ** 2)
        weights[np.abs(delta[:, None]) > 3 * sigma[i]] = 0
        relative = np.swapaxes(values[i], -1, -2)[None] @ values[lo:hi]
        vectors = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        vectors = vectors.reshape(hi - lo, values.shape[1], 3)
        mean = (weights[..., None] * vectors).sum(axis=0) / weights.sum(axis=0)[:, None]
        out[i] = values[i] @ Rotation.from_rotvec(mean).as_matrix()
    return out


def _local_noise(values, times, floor, radius, rotations):
    """Estimate vector RMS noise from normalized fifth divided differences.

    Six samples cancel polynomials through degree four, so a small, quick
    curved trajectory contributes much less than it would to a speed-based
    noise estimate. Normalizing the coefficients gives unit noise gain even
    with unequal frame intervals. Rotation differences use a local SO(3)
    tangent space; absolute axis-angle wraparound never enters the estimate.
    """
    indices = np.arange(len(times) - 5)[:, None] + np.arange(6)
    sample_times = times[indices]
    delta = sample_times[:, :, None] - sample_times[:, None, :]
    delta[:, np.arange(6), np.arange(6)] = 1
    coefficients = 1 / np.prod(delta, axis=-1)
    coefficients /= np.linalg.norm(coefficients, axis=-1, keepdims=True)
    blocks = values[indices]
    if rotations:
        relative = values[:-5, None].swapaxes(-1, -2) @ blocks
        blocks = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        blocks = blocks.reshape(len(indices), 6, values.shape[1], 3)
    residual = np.linalg.norm(np.einsum("nk,nkjd->njd", coefficients, blocks), axis=-1)
    centers = sample_times.mean(axis=-1)
    noise = np.empty((len(times), values.shape[1]))
    for i, time in enumerate(times):
        lo = np.searchsorted(centers, time - radius)
        hi = np.searchsorted(centers, time + radius, side="right")
        # Eight residuals are the minimum for this robust local estimate.
        if hi - lo < 8:
            lo = max(0, min(len(centers) - 8, (lo + hi) // 2 - 4))
            hi = lo + 8
        # Median/RMS ratio of an isotropic 3D Gaussian vector is about .888.
        noise[i] = np.maximum(floor, np.median(residual[lo:hi], axis=0) / 0.888)
    return noise


def _quadratic_fit(values, delta, sigma, noise, robust=False, valid=None):
    """Fit an independent vector trajectory per joint, evaluated at time zero."""
    delta = delta[:, None] if delta.ndim == 1 else delta
    u = delta / np.asarray(sigma)
    design = np.stack((np.ones_like(u), u, u * u), axis=-1)
    weights = np.exp(-0.5 * u * u)
    weights[np.abs(u) > 3.5] = 0
    if valid is not None:
        weights[~valid] = 0

    def fit(current_weights):
        weighted_design = (design * current_weights[..., None]).transpose(1, 2, 0)
        gram = weighted_design @ design.transpose(1, 0, 2)
        rhs = weighted_design @ values.transpose(1, 0, 2)
        # Sub-frame support can have almost no information about derivatives.
        # Regularize only those terms; do not bias the estimated position.
        gram[:, 1, 1] += 1e-12
        gram[:, 2, 2] += 1e-12
        return np.linalg.solve(gram, rhs)

    estimate = fit(weights)
    residual = np.linalg.norm(
        values - (design.transpose(1, 0, 2) @ estimate).transpose(1, 0, 2), axis=-1
    )
    score = np.sqrt(
        (weights * np.minimum(residual, 3 * noise) ** 2).sum(axis=0) / weights.sum(axis=0)
    )
    if robust:
        for _ in range(2):
            influence = np.minimum(1, 2.5 * noise / np.maximum(residual, 1e-12))
            estimate = fit(weights * influence)
            residual = np.linalg.norm(
                values - (design.transpose(1, 0, 2) @ estimate).transpose(1, 0, 2),
                axis=-1,
            )
    return estimate[:, 0], score


def _smooth_local_motion(values, times, config, noise_floor, rotations=False):
    """Fit the same independent local trajectories in bounded frame batches.

    Only execution is batched: every frame keeps its original timestamps,
    symmetric support, noise estimate, width selection and robust fit. Padded
    window entries have zero weight and never contribute to the fit.
    """
    noise = _local_noise(values, times, noise_floor, config.noise_radius_seconds, rotations)
    widths = config.max_sigma_seconds * np.array([1.0, 0.7, 0.45, 0.275])
    out = values.copy()
    centers = np.arange(1, len(times) - 1)
    max_sigma = np.minimum(
        config.max_sigma_seconds,
        np.minimum(times[centers] - times[0], times[-1] - times[centers]) / 3.5,
    )
    lo = np.searchsorted(times, times[centers] - 3.5 * max_sigma)
    hi = np.searchsorted(times, times[centers] + 3.5 * max_sigma, side="right")
    usable = hi - lo >= 3
    centers, max_sigma, lo, hi = (a[usable] for a in (centers, max_sigma, lo, hi))
    if not len(centers):
        return out
    joints = values.shape[1]
    # Bound the temporary rotation windows, including high-rate inputs. This
    # does not split temporal support: each batch still reads the full track.
    batch_size = max(1, min(256, 65536 // (int(np.max(hi - lo)) * joints)))
    for start in range(0, len(centers), batch_size):
        stop = start + batch_size
        indices = centers[start:stop]
        local_widths = np.minimum(widths[None], max_sigma[start:stop, None])
        counts = hi[start:stop] - lo[start:stop]
        offsets = np.arange(int(counts.max()))
        valid = offsets[None] < counts[:, None]
        window_ids = lo[start:stop, None] + offsets[None]
        # Substitute the center for padding so all intermediate values stay
        # finite, even when a window ends at the last frame of the track.
        window_ids = np.where(valid, window_ids, indices[:, None])
        delta = times[window_ids] - times[indices, None]
        window = values[window_ids]
        if rotations:
            relative = values[indices, None].swapaxes(-1, -2) @ window
            window = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
            window = window.reshape(len(indices), len(offsets), joints, 3)

        # Flatten frame/joint pairs into independent trajectories. The existing
        # solver then handles many frames in one NumPy call.
        window = window.transpose(1, 0, 2, 3).reshape(len(offsets), -1, 3)
        delta = np.repeat(delta.T, joints, axis=1)
        valid = np.repeat(valid.T, joints, axis=1)
        local_noise = noise[indices].reshape(-1)
        candidate_widths = np.repeat(local_widths, joints, axis=0)
        chosen = np.zeros((len(indices) * joints, 3))
        pending = np.arange(len(chosen))
        previous_score = None
        previous_width = None
        for candidate in range(len(widths)):
            width = candidate_widths[pending, candidate]
            sigma = width.copy()
            _, score = _quadratic_fit(
                window[:, pending],
                delta[:, pending],
                sigma,
                local_noise[pending],
                valid=valid[:, pending],
            )
            accepted = (score <= config.fit_tolerance * local_noise[pending]) | (
                width == candidate_widths[pending, -1]
            )
            if accepted.any():
                if previous_score is not None:
                    fraction = np.clip(
                        (config.fit_tolerance * local_noise[pending] - score)
                        / np.maximum(previous_score - score, 1e-12),
                        0,
                        1,
                    )
                    sigma *= (previous_width / width) ** fraction
                selected = pending[accepted]
                estimate, _ = _quadratic_fit(
                    window[:, selected],
                    delta[:, selected],
                    sigma[accepted],
                    local_noise[selected],
                    robust=True,
                    valid=valid[:, selected],
                )
                chosen[selected] = estimate
            pending = pending[~accepted]
            if not len(pending):
                break
            previous_score, previous_width = score[~accepted], width[~accepted]
        chosen = chosen.reshape(len(indices), joints, 3)
        out[indices] = (
            values[indices]
            @ Rotation.from_rotvec(chosen.reshape(-1, 3))
            .as_matrix()
            .reshape(len(indices), joints, 3, 3)
            if rotations
            else chosen
        )
    return out


def smooth_translation(values, times, config):
    """Robust local trajectory smoothing in camera coordinates (meters)."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    # Six-point noise differences need at least eight observations. Short
    # tracks retain the original speed-adaptive Gaussian as a fallback.
    if len(values) < 13:
        return _short_translation(values, times, config)
    return _smooth_local_motion(values[:, None], times, config, config.translation_noise_floor_m)[
        :, 0
    ]


def smooth_rotations(values, times, config):
    """Fit each joint's motion in SO(3), including small, rapid articulation."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if not len(values):
        return values.copy()
    # BF16 inference can leave matrices slightly non-orthogonal. Project to
    # SO(3) before taking relative logs and before reconstructing geometry.
    values = Rotation.from_matrix(values.reshape(-1, 3, 3)).as_matrix().reshape(values.shape)
    if len(values) < 13:
        return _short_rotations(values, times, config)
    return _smooth_local_motion(
        values, times, config, config.angular_noise_floor_rad, rotations=True
    )


def smooth_shape(values, times, sigma):
    values = np.asarray(values)
    out = values.astype(np.float64, copy=True)
    for i in range(len(times)):
        lo, hi, delta = _neighbors(times, i, sigma)
        weights = np.exp(-0.5 * (delta / sigma) ** 2)
        out[i] = np.tensordot(weights / weights.sum(), values[lo:hi], axes=(0, 0))
    return out
