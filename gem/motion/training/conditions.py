# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image-only condition policies; supervision never participates in dropout."""

from __future__ import annotations

import torch


def image_t1_conditions(batch, cliff_cam, arm, *, generator=None):
    """Build the final conditions once, with a normalized-null angular input.

    ``obs`` uses the selected joint schema in the trained [-1, 1] crop convention.
    Every returned tensor is independent of the cached labels and conditions.
    C uses the same condition policy as B2.
    """
    if arm not in ("B1", "B2", "B3", "C"):
        raise ValueError(f"Unknown T1 condition arm: {arm}")
    shape = batch["f_imgseq"].shape[:2]
    if shape[1] not in (1, 3) or (arm == "C" and shape[1] != 1):
        raise ValueError("Image conditions require T=1 or repeated-image T=3; C remains T=1")
    device = batch["f_imgseq"].device

    def draw():
        return torch.rand(shape, device=device, generator=generator)

    # Draw the same three fields in every arm so changing the condition policy
    # does not shift subsequent network/video dropout RNG consumption.
    image_draw, obs_draw, bbox_draw = draw(), draw(), draw()
    if arm == "B1":
        keep_image = torch.ones(shape, device=device, dtype=torch.bool)
        keep_obs = torch.zeros_like(keep_image)
    elif arm in ("B2", "C"):
        keep_image, keep_obs = image_draw >= 0.1, obs_draw >= 0.1
    else:
        keep_image = image_draw < 0.5
        keep_obs = ~keep_image
    keep_bbox = bbox_draw >= 0.1
    return {
        "obs": batch["obs"].masked_fill(~keep_obs[..., None, None], 0),
        "f_imgseq": batch["f_imgseq"].masked_fill(~keep_image[..., None], 0),
        "f_cliffcam": cliff_cam.masked_fill(~keep_bbox[..., None], 0),
        # No angular interval exists for an isolated image. This is already
        # normalized null; do not send a raw all-zero 6D rotation to the codec.
        "f_cam_angvel": cliff_cam.new_zeros(*shape, 6),
    }
