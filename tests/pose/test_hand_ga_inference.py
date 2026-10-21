# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current Hand camera/2D-solver integration without external model or rig assets."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gem.common.camera import perspective_projection
from gem.pose.api import hand_config
from gem.pose.conditioning.camera_depth import (
    depth_parameter_to_legacy_camera,
    effective_camera_depth_parameter,
    legacy_camera_to_depth_parameter,
)
from gem.pose.heads.output import PoseHeadOutput
from gem.pose.predictor import PosePredictor
from gem.pose.tasks.hand import mirror_hand_camera


def local_joints():
    x = torch.linspace(-0.035, 0.035, 25)
    return torch.stack((x, x.sin() * 0.7, torch.linspace(0, 0.04, 25)), -1)


class KnownGeometry(nn.Module):
    def forward(self, parameters, *, right=None, transl=None):
        count = len(parameters["marker"])
        points = local_joints().expand(count, -1, -1).clone()
        translation = torch.zeros(count, 3) if transl is None else transl.clone()
        if right is not None:
            points[right, :, 0] *= -1
            translation[right, 0] *= -1
        return {
            "joints": points + translation[:, None],
            "vertices": points + translation[:, None],
            "transl": translation,
            "parameters": parameters,
            "faces": torch.tensor([[0, 1, 2]]).expand(count, -1, -1),
        }


class KnownDirectModel(nn.Module):
    def __init__(self, *, valid=True, mask_boundaries=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.task = SimpleNamespace(
            name="hand", head=SimpleNamespace(camera_depth_mode="log_lambda")
        )
        self.conditioning = None
        self.kp2d_head = nn.Identity()
        self.kp2d_translation = hand_config().kp2d_translation
        self.pose_release_id = "test-current-hand"
        self.valid = valid
        self.mask_boundaries = mask_boundaries

    def forward(self, batch):
        count = len(batch["img"])
        target = torch.tensor([0.07, -0.04, 0.75]).expand(count, -1)
        joints = local_joints().expand(count, -1, -1)
        pixels = perspective_projection(joints + target[:, None], batch["K_fullimg"])
        crop = (pixels - batch["bbx_xys"][:, None, :2]) / batch["bbx_xys"][:, None, 2:]
        presence = torch.ones(count, 25)
        if self.mask_boundaries:
            crop[:, 0, 0] = 0.5  # Half-open crop boundary is excluded.
            crop[:, 1, 0] = -0.1  # Inside this crop, outside the full image.
            presence[:, 2] = 0.49
        if not self.valid:
            presence[:, 3:] = 0
        scalar = (batch["bbx_xys"][:, 2] * target[:, 2] / batch["K_fullimg"][:, 0, 0]).log()
        # Deliberately wrong lateral values: a valid solve must replace them.
        camera = torch.stack(
            (scalar, scalar.new_full((count,), 0.5), scalar.new_full((count,), -0.4)), -1
        )
        return PoseHeadOutput(
            {"marker": self.anchor.expand(count, 1)},
            camera,
            {"kp2d": {"mu": crop, "presence": presence}},
            self.anchor.expand(count, 4),
        )


def camera_batch():
    # Large crop also lets the full-image and crop visibility gates differ.
    boxes = torch.tensor([[0.0, 0.0, 1600.0], [0.0, 0.0, 1600.0]])
    camera = (
        torch.tensor([[[500.0, 12.0, 300.0], [0.0, 720.0, 230.0], [0.0, 0.0, 1.0]]])
        .expand(2, -1, -1)
        .clone()
    )
    original_boxes, original_camera = boxes.clone(), camera.clone()
    original_boxes[1:], original_camera[1:] = mirror_hand_camera(boxes[1:], camera[1:], 640)
    return {
        "img": torch.zeros(2, 3, 256, 256),
        "bbx_xys": boxes,
        "K_fullimg": camera,
        "img_size": torch.tensor([[640.0, 480.0], [640.0, 480.0]]),
        "right_hand": torch.tensor([False, True]),
        "original_bbx_xys": original_boxes,
        "original_K_fullimg": original_camera,
    }


@pytest.mark.parametrize("mask_boundaries,points", [(False, 25), (True, 22)])
def test_direct_solver_replaces_xy_before_physical_right_mirroring(mask_boundaries, points):
    batch = camera_batch()
    predictor = PosePredictor(KnownDirectModel(mask_boundaries=mask_boundaries), KnownGeometry())
    result = predictor.predict_batch(batch)
    expected = torch.tensor([[0.07, -0.04, 0.75], [-0.07, -0.04, 0.75]])
    torch.testing.assert_close(result.camera_translation, expected, atol=1e-6, rtol=1e-6)
    solve = result.head.auxiliary["translation_solve"]
    assert solve["valid"].tolist() == [True, True]
    assert solve["num_points"].tolist() == [points, points]
    torch.testing.assert_close(result.intrinsics, batch["original_K_fullimg"])
    # Physical right-hand pixels reflect the canonical image, including skew.
    torch.testing.assert_close(
        result.keypoints_2d[1, :, 0], 639 - result.keypoints_2d[0, :, 0], atol=1e-4, rtol=1e-6
    )
    torch.testing.assert_close(
        result.keypoints_2d[1, :, 1], result.keypoints_2d[0, :, 1], atol=1e-4, rtol=1e-6
    )


def test_insufficient_direct_support_keeps_complete_crop_camera_fallback():
    batch = camera_batch()
    result = PosePredictor(KnownDirectModel(valid=False), KnownGeometry()).predict_batch(batch)
    # Crop camera uses fx for both legacy lateral offsets, as in the GA source.
    expected = torch.tensor([[0.5 - 300 * 0.75 / 500, -0.4 - 230 * 0.75 / 500, 0.75]])
    torch.testing.assert_close(result.camera_translation[:1], expected, atol=1e-6, rtol=1e-6)
    assert not result.head.auxiliary["translation_solve"]["valid"].any()
    assert result.head.auxiliary["translation_solve"]["num_points"].tolist() == [3, 3]


def test_log_lambda_camera_keeps_native_supervision_and_ste_bounded_decoding():
    bbox = torch.tensor([[100.0, 120.0, 200.0]]).expand(3, -1)
    camera = torch.tensor([[[500.0, 0.0, 300.0], [0.0, 700.0, 230.0], [0.0, 0.0, 1.0]]]).expand(
        3, -1, -1
    )
    raw = torch.tensor([[-5.0, 0.1, -0.2], [-1.0, 0.3, 0.4], [4.0, -0.1, 0.2]], requires_grad=True)
    effective = effective_camera_depth_parameter(raw, "log_lambda", log_value_bounds=(-3.5, 3))
    torch.testing.assert_close(effective[:, 0], torch.tensor([-3.5, -1.0, 3.0]))
    effective[:, 0].sum().backward()
    torch.testing.assert_close(raw.grad[:, 0], torch.ones(3))
    decoded = depth_parameter_to_legacy_camera(
        raw.detach(), bbox, camera, "log_lambda", log_value_bounds=(-3.5, 3)
    )
    torch.testing.assert_close(decoded[:, 0], 2 * torch.exp(torch.tensor([3.5, 1.0, -3.0])))
    ground_truth = torch.tensor([[2.0, 0.1, -0.2], [1.0, 0.3, 0.4], [4.0, -0.1, 0.2]])
    encoded = legacy_camera_to_depth_parameter(ground_truth, bbox, camera, "log_lambda")
    torch.testing.assert_close(encoded[:, 0], torch.tensor([0.0, math.log(2.0), -math.log(2.0)]))
    torch.testing.assert_close(encoded[:, 1:], ground_truth[:, 1:])
