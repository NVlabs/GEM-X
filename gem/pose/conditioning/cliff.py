# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEP body/hand CLIFF conditioning in full-image pixel coordinates."""

import torch
from torch import Tensor, nn


def cliff_features(bbx_xys: Tensor, focal: Tensor, principal: Tensor) -> Tensor:
    """Normalize both center offsets and square size by fx, as trained.

    Using fy for the vertical channel would change existing checkpoint inputs.
    Geometry stays fp32 even when image/backbone features use mixed precision.
    """
    bbox, focal, principal = bbx_xys.float(), focal.float(), principal.float()
    return torch.cat((bbox[..., :2] - principal, bbox[..., 2:3]), dim=-1) / focal[..., None]


class CliffConditioning(nn.Module):
    def __init__(
        self,
        camera_mode: str = "sam3db_perspective",
        focal_length: float = 5000.0,
        focal_length_mode: str = "fixed",
    ):
        super().__init__()
        if camera_mode not in {"weak_perspective", "perspective", "sam3db_perspective"}:
            raise ValueError(f"Unsupported camera mode {camera_mode!r}")
        if focal_length_mode not in {"fixed", "diagonal"}:
            raise ValueError(f"Unsupported focal length mode {focal_length_mode!r}")
        self.camera_mode = camera_mode
        self.focal_length = focal_length
        self.focal_length_mode = focal_length_mode

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        if "bbx_xys" in batch:
            bbox = batch["bbx_xys"]
        else:
            bbox = torch.cat((batch["box_center"], batch["box_size"][:, None]), dim=-1)
        if self.camera_mode == "sam3db_perspective":
            intrinsics = batch["K_fullimg"]
            focal = intrinsics[..., 0, 0]
            principal = intrinsics[..., :2, 2]
        else:
            if "img_size" in batch:
                image_wh = batch["img_size"].float()
            else:
                image_wh = batch["K_fullimg"][..., :2, 2].float() * 2
            principal = image_wh / 2
            focal = (
                (image_wh.square().sum(dim=-1)).sqrt()
                if self.focal_length_mode == "diagonal"
                else image_wh.new_full(image_wh.shape[:-1], self.focal_length)
            )
        return cliff_features(bbox, focal, principal)
