# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose benchmark alignment math; coordinates and errors retain input units."""

import torch


def similarity_align(predicted, target):
    """Align ``[B,J,3]`` points by a proper rotation, uniform scale and translation.

    This is the released Pose Procrustes protocol. Degenerate predicted point
    clouds are not silently regularized into a different metric.
    """
    source, target = predicted.permute(0, 2, 1), target.permute(0, 2, 1)
    source_mean, target_mean = source.mean(2, keepdim=True), target.mean(2, keepdim=True)
    source_zero, target_zero = source - source_mean, target - target_mean
    variance = source_zero.square().sum((1, 2))
    covariance = source_zero @ target_zero.permute(0, 2, 1)
    u, _, v = torch.svd(covariance)
    correction = torch.eye(3, dtype=u.dtype, device=u.device)[None].repeat(len(source), 1, 1)
    correction[:, -1, -1] *= torch.sign(torch.linalg.det(u @ v.permute(0, 2, 1)))
    rotation = (v @ correction) @ u.permute(0, 2, 1)
    scale = (rotation @ covariance).diagonal(dim1=-1, dim2=-2).sum(-1) / variance
    scale = scale[:, None, None]
    translation = target_mean - scale * (rotation @ source_mean)
    return (scale * (rotation @ source) + translation).permute(0, 2, 1)


def scale_align(predicted, target, eps=1e-8):
    """Mean-center, fit a nonnegative uniform scale, and restore target mean."""
    source, target = predicted.permute(0, 2, 1), target.permute(0, 2, 1)
    source_mean, target_mean = source.mean(2, keepdim=True), target.mean(2, keepdim=True)
    source_zero, target_zero = source - source_mean, target - target_mean
    denominator = source_zero.square().sum((1, 2))
    numerator = (source_zero * target_zero).sum((1, 2))
    scale = torch.where(denominator > eps, numerator / denominator, torch.ones_like(denominator))
    scale = scale.clamp(min=0)[:, None, None]
    return (scale * source_zero + target_mean).permute(0, 2, 1)


def mean_joint_error(predicted, target):
    """Euclidean point error averaged over joints, returning one value per row."""
    return torch.sqrt(((predicted - target) ** 2).sum(-1)).mean(-1)
