# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable full-perspective translation helpers.

The lateral solver in this module assumes a calibrated pinhole camera and a
known root depth ``tz``.  It solves only ``tx`` and ``ty`` from corresponding
local 3D joints and full-image 2D pixels.  In particular, it does not make a
weak-perspective approximation: every joint uses its own absolute depth
``joint_z + tz``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch


class LateralTranslationSolveResult(NamedTuple):
    """Result of a batched fixed-depth lateral-translation solve.

    Attributes:
        translation_xy: Solved ``[..., 2]`` metric translation. Invalid
            samples are filled with zeros; the caller owns any fallback.
        valid: Boolean ``[...]`` mask for samples whose solve passed all
            geometric, point-count, conditioning, and finite-value checks.
        num_points: Integer ``[...]`` count of points used by each sample.
        condition_number: Detached ``[...]`` condition number of the
            undamped 2-by-2 normal matrix. Singular systems report ``inf``.
    """

    translation_xy: torch.Tensor
    valid: torch.Tensor
    num_points: torch.Tensor
    condition_number: torch.Tensor


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Use FP32 for low-precision inputs while preserving explicit FP64."""

    if any(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def _autocast_disabled(device: torch.device):
    """Disable autocast where PyTorch exposes a stable autocast backend."""

    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def crop_keypoints_to_full_image(
    keypoints_crop: torch.Tensor,
    bbx_xys: torch.Tensor,
) -> torch.Tensor:
    """Convert crop-centered keypoints to full-image pixels.

    Hand Pose crops use the convention

    ``keypoints_crop = (keypoints_full - box_center) / box_size``.

    Coordinates are intentionally not clipped, so predictions outside the
    crop remain representable.

    Args:
        keypoints_crop: ``[..., J, 2]`` crop-centered coordinates.
        bbx_xys: ``[..., 3]`` square boxes ``[center_x, center_y, size]``.

    Returns:
        ``[..., J, 2]`` full-image pixel coordinates.
    """

    if keypoints_crop.ndim < 2 or keypoints_crop.shape[-1] != 2:
        raise ValueError(
            "keypoints_crop must have shape [..., J, 2], got " f"{tuple(keypoints_crop.shape)}"
        )
    expected_box_shape = keypoints_crop.shape[:-2] + (3,)
    if bbx_xys.shape != expected_box_shape:
        raise ValueError(
            f"bbx_xys must have shape {expected_box_shape}, got " f"{tuple(bbx_xys.shape)}"
        )
    if keypoints_crop.device != bbx_xys.device:
        raise ValueError("keypoints_crop and bbx_xys must be on the same device")

    dtype = _compute_dtype(keypoints_crop, bbx_xys)
    with _autocast_disabled(keypoints_crop.device):
        crop = keypoints_crop.to(dtype=dtype)
        boxes = bbx_xys.to(dtype=dtype)
        center = boxes[..., :2]
        size = boxes[..., 2]
        return center[..., None, :] + size[..., None, None] * crop


def camera_scale_to_translation_z(
    camera_scale: torch.Tensor,
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    *,
    min_scale_box_product: float = 1e-8,
) -> torch.Tensor:
    """Convert the Hand Pose camera scale to metric root depth.

    The parameterization is ``tz = 2 * fx / (scale * box_size)``.  ``fx`` is
    read from the normalized intrinsic matrix, so an otherwise equivalent
    homogeneous rescaling of ``K_fullimg`` does not change the result.

    Camera heads are expected to enforce positive scale. Invalid inputs
    (including a non-positive scale-box product or invalid homogeneous scale
    in ``K``) produce ``NaN`` so the downstream solver marks that sample
    invalid instead of silently inventing a depth.

    Args:
        camera_scale: ``[...]`` positive camera scale.
        bbx_xys: ``[..., 3]`` square boxes.
        K_fullimg: ``[..., 3, 3]`` calibrated camera intrinsics.
        min_scale_box_product: Positive lower bound for ``scale * box_size``.

    Returns:
        ``[...]`` metric root depth.
    """

    if min_scale_box_product <= 0:
        raise ValueError("min_scale_box_product must be positive")
    batch_shape = camera_scale.shape
    if bbx_xys.shape != batch_shape + (3,):
        raise ValueError(
            f"bbx_xys must have shape {batch_shape + (3,)}, got " f"{tuple(bbx_xys.shape)}"
        )
    if K_fullimg.shape != batch_shape + (3, 3):
        raise ValueError(
            f"K_fullimg must have shape {batch_shape + (3, 3)}, got " f"{tuple(K_fullimg.shape)}"
        )
    if not (camera_scale.device == bbx_xys.device == K_fullimg.device):
        raise ValueError("camera_scale, bbx_xys, and K_fullimg must share a device")

    dtype = _compute_dtype(camera_scale, bbx_xys, K_fullimg)
    with _autocast_disabled(camera_scale.device):
        scale = camera_scale.to(dtype=dtype)
        boxes = bbx_xys.to(dtype=dtype)
        intrinsics = K_fullimg.to(dtype=dtype)
        k22 = intrinsics[..., 2, 2]
        k22_valid = torch.isfinite(k22) & (k22.abs() > torch.finfo(dtype).tiny)
        safe_k22 = torch.where(k22_valid, k22, torch.ones_like(k22))
        fx_input = intrinsics[..., 0, 0]
        fx_input_valid = torch.isfinite(fx_input)
        fx = torch.where(fx_input_valid, fx_input, torch.zeros_like(fx_input)) / safe_k22
        denominator = scale * boxes[..., 2]
        denominator_valid = torch.isfinite(denominator) & (denominator > min_scale_box_product)
        valid = k22_valid & fx_input_valid & denominator_valid
        safe_denominator = torch.where(denominator_valid, denominator, torch.ones_like(denominator))
        translation_z = 2.0 * fx / safe_denominator
        return torch.where(
            valid,
            translation_z,
            torch.full_like(translation_z, torch.nan),
        )


def solve_lateral_translation_full_perspective(
    joints_3d: torch.Tensor,
    keypoints_2d_full: torch.Tensor,
    translation_z: torch.Tensor,
    K_fullimg: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    point_mask: torch.Tensor | None = None,
    min_points: int = 4,
    min_depth: float = 1e-4,
    damping: float = 1e-6,
    max_condition_number: float = 1e6,
    intrinsic_row_tolerance: float = 1e-5,
) -> LateralTranslationSolveResult:
    """Solve metric ``tx, ty`` with fixed ``tz`` and complete intrinsics.

    For a local joint ``(X, Y, Z)``, fixed root depth ``tz``, full-image pixel
    ``p``, and normalized calibrated intrinsic matrix

    ``K = [[A, c], [0, 0, 1]]``,

    pinhole projection gives the linear two-variable system

    ``p - c - A @ [X, Y] / (Z + tz) = A @ [tx, ty] / (Z + tz)``.

    Stacking these systems yields a weighted 2-by-2 least-squares solve.  The
    full top-left block ``A`` is used, so anisotropic focal lengths, upper-row
    skew, and a nonzero lower-left element are all handled consistently.

    Invalid points are excluded before arithmetic.  A sample is valid only if
    it has at least ``min_points`` usable correspondences and its undamped
    normal matrix passes ``max_condition_number``.  Invalid output translations
    are zeros; callers should apply their own regressed-camera fallback.

    Floating-point work is performed in FP32 unless any input is explicitly
    FP64.  The solve remains differentiable with respect to valid joints,
    keypoints, depth, intrinsics, and positive weights.

    Args:
        joints_3d: ``[..., J, 3]`` local camera-frame 3D joints.
        keypoints_2d_full: ``[..., J, 2]`` full-image pixels.
        translation_z: ``[...]`` fixed metric root depth.
        K_fullimg: ``[..., 3, 3]`` calibrated pinhole intrinsics.
        weights: Optional non-negative ``[..., J]`` scalar joint weights.
            Non-finite or non-positive weights exclude a point.
        point_mask: Optional boolean-like ``[..., J]`` validity mask.
        min_points: Minimum usable correspondences per sample.
        min_depth: Minimum allowed absolute joint depth ``Z + tz``.
        damping: Non-negative relative ridge coefficient.  The ridge added to
            a valid normal matrix is ``damping * trace(H) / 2``.
        max_condition_number: Maximum condition number of the undamped normal
            matrix.
        intrinsic_row_tolerance: Absolute tolerance after homogeneous
            normalization for requiring the final row of ``K`` to be
            ``[0, 0, 1]``.

    Returns:
        A :class:`LateralTranslationSolveResult`.
    """

    if joints_3d.ndim < 2 or joints_3d.shape[-1] != 3:
        raise ValueError(f"joints_3d must have shape [..., J, 3], got {tuple(joints_3d.shape)}")
    expected_keypoint_shape = joints_3d.shape[:-1] + (2,)
    if keypoints_2d_full.shape != expected_keypoint_shape:
        raise ValueError(
            f"keypoints_2d_full must have shape {expected_keypoint_shape}, got "
            f"{tuple(keypoints_2d_full.shape)}"
        )
    batch_shape = joints_3d.shape[:-2]
    num_joints = joints_3d.shape[-2]
    if translation_z.shape != batch_shape:
        raise ValueError(
            f"translation_z must have shape {batch_shape}, got " f"{tuple(translation_z.shape)}"
        )
    if K_fullimg.shape != batch_shape + (3, 3):
        raise ValueError(
            f"K_fullimg must have shape {batch_shape + (3, 3)}, got " f"{tuple(K_fullimg.shape)}"
        )
    point_shape = batch_shape + (num_joints,)
    if weights is not None and weights.shape != point_shape:
        raise ValueError(f"weights must have shape {point_shape}, got {tuple(weights.shape)}")
    if point_mask is not None and point_mask.shape != point_shape:
        raise ValueError(
            f"point_mask must have shape {point_shape}, got " f"{tuple(point_mask.shape)}"
        )

    tensors = [joints_3d, keypoints_2d_full, translation_z, K_fullimg]
    if weights is not None:
        tensors.append(weights)
    if any(tensor.device != joints_3d.device for tensor in tensors):
        raise ValueError("all solver inputs must be on the same device")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if max_condition_number < 1:
        raise ValueError("max_condition_number must be at least 1")
    if intrinsic_row_tolerance < 0:
        raise ValueError("intrinsic_row_tolerance must be non-negative")

    dtype = _compute_dtype(*tensors)
    device = joints_3d.device

    with _autocast_disabled(device):
        joints = joints_3d.to(dtype=dtype)
        pixels = keypoints_2d_full.to(dtype=dtype)
        tz = translation_z.to(dtype=dtype)
        intrinsics_input = K_fullimg.to(dtype=dtype)

        # Sanitize intrinsics before normalizing their homogeneous scale.
        intrinsics_finite = torch.isfinite(intrinsics_input).all(dim=(-2, -1))
        intrinsics_clean = torch.where(
            torch.isfinite(intrinsics_input),
            intrinsics_input,
            torch.zeros_like(intrinsics_input),
        )
        k22 = intrinsics_clean[..., 2, 2]
        nonzero_k22 = k22.abs() > torch.finfo(dtype).tiny
        safe_k22 = torch.where(nonzero_k22, k22, torch.ones_like(k22))
        intrinsics = intrinsics_clean / safe_k22[..., None, None]
        pinhole_row_valid = (intrinsics[..., 2, 0].abs() <= intrinsic_row_tolerance) & (
            intrinsics[..., 2, 1].abs() <= intrinsic_row_tolerance
        )
        intrinsics_valid = intrinsics_finite & nonzero_k22 & pinhole_row_valid

        joint_finite = torch.isfinite(joints).all(dim=-1)
        pixel_finite = torch.isfinite(pixels).all(dim=-1)
        tz_finite = torch.isfinite(tz)

        joints_clean = torch.where(joint_finite[..., None], joints, torch.zeros_like(joints))
        pixels_clean = torch.where(pixel_finite[..., None], pixels, torch.zeros_like(pixels))
        tz_clean = torch.where(tz_finite, tz, torch.zeros_like(tz))
        absolute_depth = joints_clean[..., 2] + tz_clean[..., None]

        if weights is None:
            joint_weights = torch.ones(point_shape, dtype=dtype, device=device)
            weight_valid = torch.ones(point_shape, dtype=torch.bool, device=device)
        else:
            weight_input = weights.to(dtype=dtype)
            weight_valid = torch.isfinite(weight_input) & (weight_input > 0)
            joint_weights = torch.where(weight_valid, weight_input, torch.zeros_like(weight_input))

        if point_mask is None:
            requested_mask = torch.ones(point_shape, dtype=torch.bool, device=device)
        else:
            requested_mask = point_mask.to(device=device, dtype=torch.bool)

        sample_valid = intrinsics_valid & tz_finite
        usable = (
            requested_mask
            & joint_finite
            & pixel_finite
            & weight_valid
            & (absolute_depth > min_depth)
            & sample_valid[..., None]
        )
        num_points_used = usable.sum(dim=-1)
        effective_weights = torch.where(usable, joint_weights, torch.zeros_like(joint_weights))
        safe_depth = torch.where(usable, absolute_depth, torch.ones_like(absolute_depth))

        camera_xy = intrinsics[..., :2, :2]
        principal_point = intrinsics[..., :2, 2]
        local_xy_projected = torch.einsum("...ab,...jb->...ja", camera_xy, joints_clean[..., :2])
        residual = (
            pixels_clean
            - principal_point[..., None, :]
            - local_xy_projected / safe_depth[..., None]
        )
        design = camera_xy[..., None, :, :] / safe_depth[..., :, None, None]
        # Zero invalid rows explicitly, rather than relying on a later
        # zero-weight multiplication that could still encounter overflow.
        residual = torch.where(usable[..., None], residual, torch.zeros_like(residual))
        design = torch.where(usable[..., None, None], design, torch.zeros_like(design))

        normal_matrix = torch.einsum(
            "...j,...jri,...jrk->...ik",
            effective_weights,
            design,
            design,
        )
        normal_rhs = torch.einsum(
            "...j,...jri,...jr->...i",
            effective_weights,
            design,
            residual,
        )
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.transpose(-1, -2))

        normal_finite = torch.isfinite(normal_matrix).all(dim=(-2, -1))
        rhs_finite = torch.isfinite(normal_rhs).all(dim=-1)
        normal_matrix = torch.where(
            normal_finite[..., None, None],
            normal_matrix,
            torch.zeros_like(normal_matrix),
        )
        normal_rhs = torch.where(rhs_finite[..., None], normal_rhs, torch.zeros_like(normal_rhs))

        eigenvalues = torch.linalg.eigvalsh(normal_matrix)
        min_eigenvalue = eigenvalues[..., 0]
        max_eigenvalue = eigenvalues[..., 1]
        positive_definite = min_eigenvalue > 0
        condition_number = torch.where(
            positive_definite,
            max_eigenvalue / min_eigenvalue.clamp_min(torch.finfo(dtype).tiny),
            torch.full_like(min_eigenvalue, torch.inf),
        )
        solve_valid = (
            sample_valid
            & normal_finite
            & rhs_finite
            & (num_points_used >= min_points)
            & positive_definite
            & torch.isfinite(condition_number)
            & (condition_number <= max_condition_number)
        )

        normal_scale = 0.5 * normal_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        identity = torch.eye(2, dtype=dtype, device=device).expand(batch_shape + (2, 2))
        damped_normal = normal_matrix + (damping * normal_scale)[..., None, None] * identity
        matrix_to_solve = torch.where(solve_valid[..., None, None], damped_normal, identity)
        rhs_to_solve = torch.where(solve_valid[..., None], normal_rhs, torch.zeros_like(normal_rhs))
        translation_xy = torch.linalg.solve(matrix_to_solve, rhs_to_solve[..., None]).squeeze(-1)

        solution_finite = torch.isfinite(translation_xy).all(dim=-1)
        solve_valid = solve_valid & solution_finite
        translation_xy = torch.where(
            solve_valid[..., None],
            translation_xy,
            torch.zeros_like(translation_xy),
        )

    return LateralTranslationSolveResult(
        translation_xy=translation_xy,
        valid=solve_valid.detach(),
        num_points=num_points_used.detach(),
        condition_number=condition_number.detach(),
    )
