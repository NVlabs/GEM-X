# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared body/hand image-to-parameter model, independent of training and FK."""

import torch
from torch import Tensor, nn

from .conditioning import CliffConditioning, compute_crop_ray_angles, fourier_encode_angles
from .heads.output import ImageCondition, PoseHeadOutput
from .tasks.base import PoseTask


class PoseModel(nn.Module):
    """Run a shared backbone and a task decoder on normalized crops.

    ``forward`` is differentiable and returns native parameters, crop camera,
    auxiliary predictions and the final query. ``encode`` is inference-only
    and stops before SOMA FK, projection, or hand-side reconstruction.
    Training losses and image-to-mesh prediction are not part of this module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        task: PoseTask,
        *,
        conditioning: CliffConditioning | None = None,
        backbone_input_crop_w: tuple[int, int] | None = None,
        ray_pe_num_bands: int | None = None,
        kp2d_head: nn.Module | None = None,
        kp2d_translation: dict | None = None,
        pose_release_id: str | None = None,
    ):
        super().__init__()
        if ray_pe_num_bands is not None and backbone_input_crop_w is not None:
            raise ValueError("RayPE requires the full image crop")
        if task.head.cliff_cond_dim and conditioning is None:
            raise ValueError("This pose head requires CLIFF conditioning")
        expected_ray_dim = 0 if ray_pe_num_bands is None else 4 * ray_pe_num_bands
        if task.head.ray_cond_dim != expected_ray_dim:
            raise ValueError("RayPE band count does not match the head")
        self.backbone = backbone
        self.task = task
        self.conditioning = conditioning
        self.backbone_input_crop_w = backbone_input_crop_w
        self.ray_pe_num_bands = ray_pe_num_bands
        self.kp2d_head = kp2d_head
        self.kp2d_translation = dict(kp2d_translation or {})
        self.pose_release_id = pose_release_id

    def forward(self, batch: dict[str, Tensor]) -> PoseHeadOutput:
        image = batch["img"]
        if self.backbone_input_crop_w is not None:
            start, stop = self.backbone_input_crop_w
            image = image[..., start:stop]
        features = self.backbone(image)
        cliff = self.conditioning(batch) if self.conditioning is not None else None
        ray_feats = None
        if self.ray_pe_num_bands is not None:
            bbox = batch.get("bbx_xys")
            if bbox is None:
                bbox = torch.cat((batch["box_center"], batch["box_size"][:, None]), dim=-1)
            angles = compute_crop_ray_angles(
                bbox, batch["K_fullimg"], grid_hw=features.shape[-2:], crop_hw=image.shape[-2:]
            )
            ray_feats = fourier_encode_angles(angles, num_bands=self.ray_pe_num_bands)
        output = self.task(features, cliff_cond=cliff, ray_feats=ray_feats)
        if self.kp2d_head is not None:
            output.auxiliary["kp2d"] = self.kp2d_head(features)
        return output

    @torch.inference_mode()
    def encode(self, batch: dict[str, Tensor]) -> ImageCondition:
        if self.training:
            raise RuntimeError("encode requires model.eval()")
        output = self(batch)
        return ImageCondition(
            output.token, output.pred_cam, output.parameters, self.task.name, self.pose_release_id
        )
