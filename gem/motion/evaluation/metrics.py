# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion joint metrics with explicit joint sets and sequence weighting.

Inputs use meters. Distance errors use millimeters, RTE uses percent, and
jitter is mean joint jerk at 30 Hz divided by ten. Each returned array retains
its individual metric samples for aggregation across complete sequences.
"""

from dataclasses import dataclass

import numpy as np
import torch

from gem.common.evaluation import mean_joint_error, similarity_align


@dataclass(frozen=True)
class JointMetricProtocol:
    num_joints: int
    pelvis_indices: tuple[int, int]
    supports_global_metrics: bool


JOINT_METRIC_PROTOCOLS = {
    "smpl24": JointMetricProtocol(24, (1, 2), True),
    "3dpw_j14": JointMetricProtocol(14, (2, 3), False),
}


def _get_protocol(target_joint_set):
    if target_joint_set not in JOINT_METRIC_PROTOCOLS:
        raise ValueError(f"Unsupported target_joint_set: {target_joint_set!r}")
    return JOINT_METRIC_PROTOCOLS[target_joint_set]


def _prepare_joint_pair(batch, protocol, *, global_coordinates):
    suffix = "_glob" if global_coordinates else ""
    prediction = batch[f"pred_j3d{suffix}"].detach().float().cpu()
    target = batch[f"target_j3d{suffix}"].detach().float().cpu()
    if prediction.ndim != 3 or prediction.shape[1:] != (protocol.num_joints, 3):
        raise ValueError(f"Predicted joints must have shape (T,{protocol.num_joints},3)")
    if prediction.shape != target.shape:
        raise ValueError("Target joints must match predicted joints")
    return prediction, target


def _retain_frames(arrays, mask):
    if mask is None:
        return arrays
    if mask.dtype != torch.bool or mask.shape != (len(arrays[0]),):
        raise ValueError("Metric mask must be bool[T] on the original sequence timeline")
    return [array[mask.cpu()] for array in arrays]


def _center_joints(prediction, target, protocol):
    indices = list(protocol.pelvis_indices)
    return (
        prediction - prediction[:, indices].mean(1, keepdim=True),
        target - target[:, indices].mean(1, keepdim=True),
    )


def _camera_errors(prediction, target):
    return {
        "mpjpe": mean_joint_error(prediction, target).numpy() * 1000,
        "pa_mpjpe": mean_joint_error(similarity_align(prediction, target), target).numpy() * 1000,
    }


def _align_points(source, target, query, *, fixed_scale=False):
    """Fit a proper least-squares transform on paired points, then apply it."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    covariance = x.T @ y / len(source)
    left, singular, right = torch.linalg.svd(covariance)
    signs = torch.ones(3, dtype=source.dtype)
    signs[-1] = torch.linalg.det(left @ right).sign()
    rotation = (left * signs) @ right
    scale = 1.0 if fixed_scale else (singular * signs).sum() / x.square().sum(1).mean()
    return scale * ((query - source_mean) @ rotation) + target_mean


@torch.no_grad()
def compute_soma_camcoord_joint_metrics(batch, *, target_joint_set, mask=None):
    """Pelvis-relative MPJPE and similarity-aligned PA-MPJPE, one value per frame."""
    protocol = _get_protocol(target_joint_set)
    pair = _prepare_joint_pair(batch, protocol, global_coordinates=False)
    prediction, target = _retain_frames(pair, mask)
    prediction, target = _center_joints(prediction, target, protocol)
    return _camera_errors(prediction, target)


@torch.no_grad()
def compute_soma_global_joint_metrics(
    batch, *, target_joint_set, mask=None, fps=30, chunk_length=100
):
    """World joint metrics for SMPL24, after removing invalid frames.

    ``wa2_mpjpe`` (W-MPJPE100) fits a similarity transform on the first two
    frames of each window. ``waa_mpjpe`` (WA-MPJPE100) fits all its frames.
    The last partial window is retained. RTE uses joint zero and one rigid,
    fixed-scale trajectory alignment, normalized by the target path length.
    Jitter uses third differences on the retained timeline, including gaps.
    At least four valid frames and a nonzero target root path are required.
    """
    protocol = _get_protocol(target_joint_set)
    if not protocol.supports_global_metrics:
        raise ValueError(f"No world metric protocol is defined for {target_joint_set!r}")
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    pair = _prepare_joint_pair(batch, protocol, global_coordinates=True)
    prediction, target = _retain_frames(pair, mask)
    if len(prediction) < 4:
        raise ValueError("Global joint metrics require at least four valid frames")
    first_errors, window_errors = [], []
    for start in range(0, len(prediction), chunk_length):
        pred_window, target_window = (
            prediction[start : start + chunk_length],
            target[start : start + chunk_length],
        )
        first_aligned = _align_points(
            pred_window[:2].reshape(-1, 3), target_window[:2].reshape(-1, 3), pred_window
        )
        window_aligned = _align_points(
            pred_window.reshape(-1, 3), target_window.reshape(-1, 3), pred_window
        )
        first_errors.append(mean_joint_error(first_aligned, target_window).numpy() * 1000)
        window_errors.append(mean_joint_error(window_aligned, target_window).numpy() * 1000)
    root, target_root = prediction[:, 0], target[:, 0]
    aligned_root = _align_points(root, target_root, root, fixed_scale=True)
    path_length = (target_root[1:] - target_root[:-1]).norm(dim=-1).sum()
    rte = (aligned_root - target_root).norm(dim=-1) / path_length * 100
    jerk = prediction[3:] - 3 * prediction[2:-1] + 3 * prediction[1:-2] - prediction[:-3]
    return {
        "wa2_mpjpe": np.concatenate(first_errors),
        "waa_mpjpe": np.concatenate(window_errors),
        "rte": rte.numpy(),
        "jitter": (jerk * fps**3).norm(dim=-1).mean(-1).numpy() / 10,
    }
