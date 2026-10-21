# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gem.pose.conditioning import CliffConditioning, compute_crop_ray_angles
from gem.pose.tasks.hand import mirror_hand_camera


def test_cliff_uses_real_principal_point_and_fx_for_both_axes():
    batch = {
        "bbx_xys": torch.tensor([[350.0, 200.0, 100.0]]),
        "K_fullimg": torch.tensor([[[500.0, 0.0, 300.0], [0.0, 750.0, 100.0], [0.0, 0.0, 1.0]]]),
        "img_size": torch.tensor([[1000.0, 600.0]]),
    }
    torch.testing.assert_close(CliffConditioning()(batch), torch.tensor([[0.1, 0.2, 0.2]]))
    torch.testing.assert_close(
        CliffConditioning("perspective", focal_length=500)(batch), torch.tensor([[-0.3, -0.2, 0.2]])
    )


def test_calibrated_hand_mirror_matches_pixel_projection_with_skew():
    bbox = torch.tensor([[350.0, 200.0, 100.0]])
    camera = torch.tensor([[[500.0, 10.0, 300.0], [0.0, 750.0, 100.0], [0.0, 0.0, 1.0]]])
    mirrored_bbox, mirrored_camera = mirror_hand_camera(bbox, camera, 1000)
    point = torch.tensor([[[0.2], [0.1], [2.0]]])
    pixels = camera @ point
    pixels = pixels[..., :2, 0] / pixels[..., 2:, 0]
    mirrored_point = point * point.new_tensor([-1.0, 1.0, 1.0])[None, :, None]
    mirrored_pixels = mirrored_camera @ mirrored_point
    mirrored_pixels = mirrored_pixels[..., :2, 0] / mirrored_pixels[..., 2:, 0]
    torch.testing.assert_close(mirrored_pixels[:, 0], 999 - pixels[:, 0])
    torch.testing.assert_close(mirrored_pixels[:, 1], pixels[:, 1])
    restored_bbox, restored_camera = mirror_hand_camera(mirrored_bbox, mirrored_camera, 1000)
    torch.testing.assert_close(restored_bbox, bbox)
    torch.testing.assert_close(restored_camera, camera)
    legacy_bbox, legacy_camera = mirror_hand_camera(bbox, camera, 1000, calibrated=False)
    assert legacy_bbox[0, 0] == mirrored_bbox[0, 0] + 1
    assert legacy_camera[0, 0, 1] == 10
    assert mirrored_camera[0, 0, 1] == -10


def test_ray_grid_uses_continuous_patch_centers():
    bbox = torch.tensor([[64.0, 64.0, 128.0]])
    camera = torch.tensor([[[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]])
    rays = compute_crop_ray_angles(bbox, camera, grid_hw=(2, 2), crop_hw=(256, 256))
    expected = torch.atan(
        torch.tensor([[[-0.32, -0.32], [0.32, -0.32], [-0.32, 0.32], [0.32, 0.32]]])
    )
    torch.testing.assert_close(rays, expected)
