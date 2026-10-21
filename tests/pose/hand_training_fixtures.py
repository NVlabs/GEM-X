# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small differentiable Hand fixtures, independent of SOMA assets and source checkouts."""

import torch
from torch import nn

from gem.common.geometry.rotations import axis_angle_to_matrix
from gem.pose.heads.output import PoseHeadOutput


class HandFixtureGeometry(nn.Module):
    """Small differentiable geometry with distinct wrist/finger/id/scale effects."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(718)
        template = torch.randn(25, 3, generator=generator) * 0.025
        template[0] = 0
        identity = torch.randn(5, 75, generator=generator) * 0.005
        scales = torch.randn(26, 75, generator=generator) * 0.003
        identity[:, :3] = 0
        scales[:, :3] = 0
        self.register_buffer("template", template, persistent=False)
        self.register_buffer("identity_basis", identity, persistent=False)
        self.register_buffer("scale_basis", scales, persistent=False)
        self.calls = []

    def forward(self, parameters):
        self.calls.append(torch.is_grad_enabled())
        scale = parameters["scale_params"].float()
        shape = self.template + (
            parameters["identity_coeffs"].float() @ self.identity_basis
            + scale[:, 1:] @ self.scale_basis
        ).reshape(-1, 25, 3)
        fingers = torch.einsum("bjik,bjk->bji", parameters["hand_pose"].float(), shape[:, 1:])
        posed = torch.cat((shape[:, :1], fingers), 1)
        joints = torch.einsum("bik,bjk->bji", parameters["wrist_orient"][:, 0].float(), posed)
        joints = joints * scale[:, :1, None]
        vertices = torch.cat((joints, joints[:, 1:6] + torch.tensor([0.002, 0.001, 0.003])), 1)
        return {"joints": joints, "vertices": vertices}


def hand_case():
    generator = torch.Generator().manual_seed(78)
    count = 3
    aa = torch.randn(count, 25, 3, generator=generator) * 0.12
    poses = axis_angle_to_matrix(aa)
    identity = torch.randn(count, 5, generator=generator) * 0.15
    local = torch.randn(count, 26, generator=generator) * 0.2
    global_scale = torch.tensor([[0.9], [1.05], [1.1]])
    predicted_pose = axis_angle_to_matrix(aa + 0.035)
    parameters = {
        "wrist_orient": predicted_pose[:, :1].detach().requires_grad_(),
        "hand_pose": predicted_pose[:, 1:].detach().requires_grad_(),
        "identity_coeffs": (identity + 0.05).detach().requires_grad_(),
        "scale_params": (torch.cat((global_scale, local), -1) + 0.04).detach().requires_grad_(),
    }
    camera = torch.tensor(
        [[18.0, 0.01, -0.02], [17.0, 0.02, 0.01], [19.0, -0.01, 0.005]], requires_grad=True
    )
    auxiliary = {"hand_pose": torch.zeros(count * 6, 24, 3, 3, requires_grad=True)}
    head = PoseHeadOutput(parameters, camera, auxiliary, torch.zeros(count, 8))
    kp2d = torch.randn(count, 25, 3, generator=generator)
    kp2d[..., 2] = 1
    kp2d[0, 4, 2] = 2
    kp2d[2, 9, 2] = 0
    kp3d = torch.randn(count, 25, 4, generator=generator)
    kp3d[..., 3] = 1
    kp3d[0, 19, 3] = 0
    batch = {
        "img": torch.zeros(count, 3, 16, 16),
        "bbx_xys": torch.tensor([[300.0, 210, 180], [275, 170, 160], [340, 250, 200]]),
        "K_fullimg": torch.tensor([[[480.0, 0, 315], [0, 520, 245], [0, 0, 1]]])
        .expand(count, -1, -1)
        .clone(),
        "img_size": torch.tensor([[640.0, 480]] * count),
        "aug_rot": torch.zeros(count),
        "hand_poses": poses,
        "hand_identity_coeffs": identity,
        "hand_global_scale": global_scale,
        "hand_scale_params": local,
        "has_hand_params": torch.tensor([1.0, 0.0, 1.0]),
        "has_cam_gt": torch.tensor([0.0, 1.0, 1.0]),
        "pred_cam_gt": camera.detach() + torch.tensor([0.2, 0.003, -0.004]),
        "keypoints_2d": kp2d,
        "keypoints_3d": kp3d,
    }
    return head, batch
