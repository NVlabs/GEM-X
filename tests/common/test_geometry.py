# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numeric conventions shared by pose heads and motion decoding."""

import math

import pytest
import torch

from gem.common.camera import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    estimate_K,
    get_a_pred_cam,
    perspective_projection,
    project_to_bi01,
)
from gem.common.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rot6d_to_rotmat,
    rotation_6d_to_matrix,
)
from gem.common.geometry.trajectory import get_local_transl_vel, rollout_local_transl_vel


def test_pose_and_motion_6d_have_distinct_axis_conventions():
    # First two rows of a positive 90-degree rotation about z.
    motion_axes = torch.tensor([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]])
    positive_z = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(rotation_6d_to_matrix(motion_axes), positive_z)
    torch.testing.assert_close(matrix_to_rotation_6d(positive_z), motion_axes)
    # The pose head interprets the same six numbers as columns, reversing yaw.
    torch.testing.assert_close(rot6d_to_rotmat(motion_axes), positive_z.mT)
    pose_axes = positive_z.mT[..., :2, :].reshape(1, 6)
    torch.testing.assert_close(rot6d_to_rotmat(pose_axes), positive_z)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.7, math.pi - 1e-6, math.pi])
def test_rotation_roundtrip_near_zero_and_pi(angle):
    axis = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    axis /= axis.norm()
    matrix = axis_angle_to_matrix(axis * angle)
    recovered = axis_angle_to_matrix(matrix_to_axis_angle(matrix))
    torch.testing.assert_close(recovered, matrix, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(matrix.mT @ matrix, torch.eye(3, dtype=matrix.dtype))
    torch.testing.assert_close(torch.linalg.det(matrix), torch.tensor(1.0, dtype=matrix.dtype))


def test_estimated_intrinsics_use_deployed_diagonal_focal():
    intrinsics = estimate_K(640, 480)
    expected = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(intrinsics, expected)


def test_camera_projection_and_cliff_convention():
    intrinsics = estimate_K(640, 480)
    crop = torch.tensor([400.0, 200.0, 200.0])
    camera = torch.tensor([2.0, 0.1, -0.2])
    translation = compute_transl_full_cam(camera, crop, intrinsics)
    torch.testing.assert_close(translation, torch.tensor([0.5, -0.4, 4.0]))
    torch.testing.assert_close(get_a_pred_cam(translation, crop, intrinsics), camera)
    torch.testing.assert_close(
        compute_bbox_info_bedlam(crop, intrinsics), torch.tensor([0.1, -0.05, 0.25])
    )
    projected = perspective_projection(translation[None], intrinsics)
    torch.testing.assert_close(projected, torch.tensor([[420.0, 160.0]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_motion_projection_keeps_large_pixels_and_signed_depth_in_float32(dtype):
    # The first pixel exceeds FP16 range; the second is behind the camera and
    # remains a finite signed projection until the caller's depth mask applies.
    points = torch.tensor([[[[100.0, 20.0, 1.0], [1.0, 2.0, -1.0]]]], requires_grad=True)
    intrinsics = torch.tensor([[[[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1.0]]]])
    bbox = torch.tensor([[[320.0, 240.0, 640.0]]])
    with torch.autocast("cpu", dtype=dtype):
        projection = project_to_bi01(points, bbox, intrinsics)
    assert projection.dtype == torch.float32
    torch.testing.assert_close(
        projection, torch.tensor([[[[156.75, 31.75], [-1.0625, -2.625]]]]), rtol=0, atol=0
    )
    projection.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_local_displacement_uses_previous_frame_rotation():
    orientations = torch.tensor([[[0.0, 0.0, math.pi / 2]]]).expand(1, 4, 3)
    translation = torch.tensor(
        [[[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0], [0.0, 8.0, 0.0]]]
    )
    local = get_local_transl_vel(translation, orientations)
    expected = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    torch.testing.assert_close(local, expected, atol=1e-6, rtol=1e-6)
    recovered = rollout_local_transl_vel(local, orientations, translation[:, :1])
    torch.testing.assert_close(recovered, translation)


def test_one_frame_has_no_outgoing_displacement():
    translation = torch.tensor([[[1.0, 2.0, 3.0]]])
    orientations = torch.tensor([[[0.2, -0.1, 0.3]]])
    velocity = get_local_transl_vel(translation, orientations)
    torch.testing.assert_close(velocity, torch.zeros_like(translation))
    torch.testing.assert_close(
        rollout_local_transl_vel(velocity, orientations), torch.zeros_like(translation)
    )
