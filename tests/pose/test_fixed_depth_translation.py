# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable fixed-depth camera geometry and gradient contracts."""

import pytest
import torch

from gem.pose.conditioning.fixed_depth_translation import (
    camera_scale_to_translation_z,
    crop_keypoints_to_full_image,
    solve_lateral_translation_full_perspective,
)


def _fixture(dtype=torch.float64, batch=1):
    joints = (
        torch.tensor(
            [
                [-0.08, -0.05, -0.12],
                [0.06, -0.04, -0.05],
                [0.12, 0.03, 0.02],
                [-0.03, 0.08, 0.09],
                [0.02, -0.10, 0.15],
                [-0.11, 0.06, 0.21],
            ],
            dtype=dtype,
        )
        .expand(batch, -1, -1)
        .clone()
    )
    translation = torch.tensor([0.23, -0.09, 1.7], dtype=dtype).expand(batch, -1).clone()
    camera = (
        torch.tensor(
            [[1320.0, 73.0, 811.0], [-19.0, 910.0, 437.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        .expand(batch, -1, -1)
        .clone()
    )
    points = joints + translation[:, None]
    homogeneous = torch.einsum("bij,bkj->bki", camera, points)
    pixels = homogeneous[..., :2] / homogeneous[..., 2:]
    return joints, pixels, translation, camera


def test_crop_coordinates_and_depth_use_camera_scale_without_clipping_or_epsilon():
    crop = torch.tensor([[[-0.5, 0.25], [0.0, 0.0], [0.75, -0.5]]], dtype=torch.float64)
    boxes = torch.tensor([[800.0, 450.0, 200.0]], dtype=torch.float64)
    expected = torch.tensor([[[700.0, 500.0], [800.0, 450.0], [950.0, 350.0]]], dtype=torch.float64)
    torch.testing.assert_close(crop_keypoints_to_full_image(crop, boxes), expected, rtol=0, atol=0)
    camera = torch.tensor(
        [[[2000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 2.0]]], dtype=torch.float64
    )
    scale = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
    depth = camera_scale_to_translation_z(scale, boxes, camera)
    torch.testing.assert_close(depth, torch.tensor([2.5], dtype=torch.float64), rtol=0, atol=0)
    depth.sum().backward()
    torch.testing.assert_close(
        scale.grad, torch.tensor([-0.625], dtype=torch.float64), rtol=0, atol=0
    )


def test_depth_invalid_scale_or_homogeneous_camera_is_explicitly_nan():
    scale = torch.tensor([2.0, 0.0, -1.0, float("nan"), 2.0, 1e-8])
    boxes = torch.tensor([320.0, 240.0, 1.0]).expand(6, -1)
    camera = torch.eye(3).expand(6, -1, -1).clone()
    camera[4, 2, 2] = 0
    depth = camera_scale_to_translation_z(scale, boxes, camera)
    assert depth[0] == 1
    assert torch.isnan(depth[1:]).all()


def test_full_perspective_recovery_preserves_skew_and_arbitrary_batch_shape():
    joints, pixels, translation, camera = _fixture(batch=6)
    # Homogeneous scaling must not alter the calibrated pinhole camera.
    result = solve_lateral_translation_full_perspective(
        joints.reshape(2, 3, 6, 3),
        pixels.reshape(2, 3, 6, 2),
        translation[:, 2].reshape(2, 3),
        (camera * 2).reshape(2, 3, 3, 3),
        damping=0,
    )
    assert result.valid.all()
    assert result.num_points.tolist() == [[6, 6, 6], [6, 6, 6]]
    torch.testing.assert_close(
        result.translation_xy, translation[:, :2].reshape(2, 3, 2), rtol=0, atol=1e-12
    )


def test_weighted_noisy_solve_matches_independent_augmented_least_squares():
    joints, pixels, translation, camera = _fixture()
    pixels = pixels + torch.tensor(
        [[[3.0, -5.0], [-2.0, 1.0], [1.0, 4.0], [0.0, 2.0], [-4.0, -2.0], [2.0, 1.0]]]
    )
    weights = torch.tensor([[0.5, 1.0, 0.0, 1.5, 2.0, 3.0]], dtype=torch.float64)
    damping = 0.003
    result = solve_lateral_translation_full_perspective(
        joints,
        pixels,
        translation[:, 2],
        camera,
        weights=weights,
        damping=damping,
    )
    depth = joints[0, :, 2] + translation[0, 2]
    design = camera[0, :2, :2][None] / depth[:, None, None]
    rhs = pixels[0] - camera[0, :2, 2] - (camera[0, :2, :2] @ joints[0, :, :2].T).T / depth[:, None]
    design = (weights[0].sqrt()[:, None, None] * design).reshape(-1, 2)
    rhs = (weights[0].sqrt()[:, None] * rhs).reshape(-1, 1)
    ridge = (damping * design.square().sum() / 2).sqrt() * torch.eye(2, dtype=torch.float64)
    expected = torch.linalg.lstsq(
        torch.cat((design, ridge)), torch.cat((rhs, torch.zeros(2, 1, dtype=torch.float64)))
    ).solution[:, 0]
    assert result.valid.item() and result.num_points.item() == 5
    torch.testing.assert_close(result.translation_xy[0], expected, rtol=0, atol=1e-12)


def test_point_exclusion_and_minimum_count_are_per_sample():
    joints, pixels, translation, camera = _fixture(batch=3)
    mask = torch.ones(3, 6, dtype=torch.bool)
    weights = torch.ones(3, 6, dtype=torch.float64)
    mask[0, 0] = False
    pixels[0, 0] = float("nan")
    weights[0, 1] = -1
    # Exactly min_depth is excluded, rather than accepted with >=.
    joints[1, 0, 2] = -translation[1, 2]
    pixels[1, 1] = float("inf")
    weights[1, 2] = float("nan")
    mask[2, :] = False
    result = solve_lateral_translation_full_perspective(
        joints,
        pixels,
        translation[:, 2],
        camera,
        weights=weights,
        point_mask=mask,
        min_depth=0,
        min_points=4,
        damping=0,
    )
    assert result.valid.tolist() == [True, False, False]
    assert result.num_points.tolist() == [4, 3, 0]
    torch.testing.assert_close(result.translation_xy[0], translation[0, :2], rtol=0, atol=1e-12)
    assert torch.equal(result.translation_xy[1:], torch.zeros(2, 2, dtype=torch.float64))
    assert torch.isinf(result.condition_number[2])


def test_bad_calibration_and_conditioning_are_rejected_before_damping():
    joints, pixels, translation, camera = _fixture(batch=4)
    camera[0, :2, :2] = 0
    camera[1, :2, :2] = torch.diag(torch.tensor([1000.0, 0.001], dtype=torch.float64))
    camera[2, 2, 0] = 0.01
    camera[3, 0, 0] = float("nan")
    result = solve_lateral_translation_full_perspective(
        joints,
        pixels,
        translation[:, 2],
        camera,
        damping=1.0,
        max_condition_number=1e6,
    )
    assert not result.valid.any()
    assert torch.equal(result.translation_xy, torch.zeros(4, 2, dtype=torch.float64))
    assert torch.isinf(result.condition_number[0])
    assert result.condition_number[1] > 1e6
    assert result.num_points.tolist() == [6, 6, 0, 0]


def test_solver_gradients_include_joints_pixels_depth_intrinsics_and_weights():
    joints, pixels, translation, camera = _fixture()
    pixels = pixels + 0.2 * torch.arange(12, dtype=torch.float64).reshape(1, 6, 2)
    values = tuple(
        t.clone().requires_grad_()
        for t in (
            joints,
            pixels,
            translation[:, 2],
            camera,
            torch.linspace(0.5, 1.5, 6, dtype=torch.float64)[None],
        )
    )

    def solve(joints, pixels, depth, intrinsics, weights):
        return solve_lateral_translation_full_perspective(
            joints,
            pixels,
            depth,
            intrinsics,
            weights=weights,
        ).translation_xy

    assert torch.autograd.gradcheck(solve, values, eps=1e-6, atol=2e-6, rtol=1e-4)
    result = solve_lateral_translation_full_perspective(*values[:4], weights=values[4])
    gradients = torch.autograd.grad(result.translation_xy.square().sum(), values)
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients)
    assert not any(
        t.requires_grad for t in (result.valid, result.num_points, result.condition_number)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_promote_to_fp32_even_inside_autocast(dtype):
    joints, pixels, translation, camera = _fixture(dtype=dtype)
    expected = solve_lateral_translation_full_perspective(
        joints.float(), pixels.float(), translation[:, 2].float(), camera.float()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = solve_lateral_translation_full_perspective(
            joints, pixels, translation[:, 2], camera
        )
    assert actual.translation_xy.dtype == torch.float32
    torch.testing.assert_close(actual.translation_xy, expected.translation_xy, rtol=0, atol=0)
    explicit_double = solve_lateral_translation_full_perspective(
        joints, pixels, translation[:, 2], camera, weights=torch.ones(1, 6, dtype=torch.float64)
    )
    assert explicit_double.translation_xy.dtype == torch.float64
