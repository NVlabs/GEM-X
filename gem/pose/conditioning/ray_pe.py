# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera-ray positional features for crop-aligned patch tokens."""

from __future__ import annotations

import torch


def crop_points_to_ray_angles(
    crop_xy: torch.Tensor,
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    *,
    crop_hw: tuple[int, int],
) -> torch.Tensor:
    """Map continuous crop coordinates to camera-ray azimuth/elevation.

    Args:
        crop_xy: ``(B, N, 2)`` coordinates in the model crop, ordered ``(u, v)``.
        bbx_xys: ``(B, 3)`` augmented full-image ``(cx, cy, size)`` crop boxes.
        K_fullimg: ``(B, 3, 3)`` intrinsics in the same augmented image frame.
        crop_hw: Model crop height and width.

    Returns:
        ``(B, N, 2)`` containing azimuth and elevation in radians, in fp32.
    """

    crop_height, crop_width = crop_hw
    solve_dtype = torch.float32
    crop_scale = torch.tensor(
        [crop_width, crop_height],
        device=crop_xy.device,
        dtype=solve_dtype,
    )
    crop_xy_fp32 = crop_xy.to(dtype=solve_dtype)
    bbx_xys_fp32 = bbx_xys.to(dtype=solve_dtype)
    K_fullimg_fp32 = K_fullimg.to(dtype=solve_dtype)

    crop_normalized = crop_xy_fp32 / crop_scale - 0.5
    full_xy = bbx_xys_fp32[:, None, :2] + crop_normalized * bbx_xys_fp32[:, None, 2:3]
    full_pixels = torch.cat((full_xy, torch.ones_like(full_xy[..., :1])), dim=-1)
    with torch.autocast(device_type=K_fullimg.device.type, enabled=False):
        rays = torch.linalg.solve(
            K_fullimg_fp32,
            full_pixels.transpose(-1, -2),
        ).transpose(-1, -2)

    azimuth = torch.atan2(rays[..., 0], rays[..., 2])
    elevation = torch.atan2(rays[..., 1], rays[..., 2])
    return torch.stack((azimuth, elevation), dim=-1)


def compute_crop_ray_angles(
    bbx_xys: torch.Tensor,
    K_fullimg: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
    crop_hw: tuple[int, int],
) -> torch.Tensor:
    """Return one camera-ray angle pair per row-major patch token.

    Patch locations use continuous token-cell centers, matching DINO RoPE and
    ViDiHand: ``(index + 0.5) * crop_size / grid_size``.
    """

    grid_height, grid_width = grid_hw
    crop_height, crop_width = crop_hw
    device = bbx_xys.device
    rows = (torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5) * (
        crop_height / grid_height
    )
    cols = (torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5) * (
        crop_width / grid_width
    )
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    crop_xy = torch.stack((u, v), dim=-1).reshape(1, -1, 2)
    crop_xy = crop_xy.expand(bbx_xys.shape[0], -1, -1)
    return crop_points_to_ray_angles(
        crop_xy,
        bbx_xys,
        K_fullimg,
        crop_hw=crop_hw,
    )


def fourier_encode_angles(angles: torch.Tensor, *, num_bands: int = 8) -> torch.Tensor:
    """Fourier-encode azimuth/elevation with frequencies ``2**[0, bands)``."""

    frequencies = 2.0 ** torch.arange(
        num_bands,
        device=angles.device,
        dtype=torch.float32,
    )
    angles_fp32 = angles.to(dtype=torch.float32)
    azimuth = angles_fp32[..., 0, None] * frequencies
    elevation = angles_fp32[..., 1, None] * frequencies
    return torch.cat(
        (
            azimuth.sin(),
            azimuth.cos(),
            elevation.sin(),
            elevation.cos(),
        ),
        dim=-1,
    )
