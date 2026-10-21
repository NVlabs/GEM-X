# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical left-hand decoder and right-hand camera canonicalization."""

import torch
from torch import Tensor

from ..heads.soma_hand_head import SOMAHandHeadV2
from .base import PoseTask


def mirror_hand_camera(
    bbx_xys: Tensor, intrinsics: Tensor, image_width: Tensor | int, *, calibrated: bool = True
) -> tuple[Tensor, Tensor]:
    """Mirror full-image camera metadata for a right hand.

    Calibrated crops use pixel centers (W-1-x) and K'=F K M. Legacy
    crop-then-flip inputs use W-x and only mirror the principal point.
    The caller must also apply the corresponding image/crop transform.
    """
    bbox, camera = bbx_xys.float().clone(), intrinsics.float().clone()
    limit = torch.as_tensor(image_width, device=bbox.device, dtype=bbox.dtype) - int(calibrated)
    bbox[..., 0] = limit - bbox[..., 0]
    if calibrated:
        pixel_flip = (
            torch.eye(3, device=camera.device, dtype=camera.dtype).expand(camera.shape).clone()
        )
        pixel_flip[..., 0, 0] = -1
        pixel_flip[..., 0, 2] = limit
        camera_mirror = camera.new_tensor([-1, 1, 1])
        camera = (pixel_flip @ camera) * camera_mirror
    else:
        camera[..., 0, 2] = limit - camera[..., 0, 2]
    return bbox, camera


class HandTask(PoseTask):
    """MHR identity5 and GS+local26 parameter head for canonical left hands.

    Inputs are already canonical left-hand crops. Mirroring poses back to the
    right hand requires the SOMA rigs and belongs to the FK predictor.
    """

    def __init__(self, head: SOMAHandHeadV2):
        super().__init__("hand", head)
