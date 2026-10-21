# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode crop cameras once and project the resulting full-camera geometry."""

import torch

from gem.common.camera import compute_transl_full_cam, perspective_projection


def decode_camera(pred_cam, bbox, intrinsics, *, representation="legacy"):
    """Decode a body/hand camera vector into full-image translation in meters.

    Legacy [s,tx,ty] retains fx for both center offsets. The body crop_2d
    representation instead predicts the origin's crop-normalized coordinates.
    Input is a raw head camera, never an already solved camera translation.
    """
    camera, bbox, intrinsics = pred_cam.float(), bbox.float(), intrinsics.float()
    if representation == "legacy":
        return compute_transl_full_cam(camera, bbox, intrinsics)
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    center = intrinsics[:, :2, 2]
    if representation == "log_lambda":
        depth = camera[:, 0].exp() * fx / bbox[:, 2]
        xy = camera[:, 1:] + depth[:, None] / fx[:, None] * (bbox[:, :2] - center)
    elif representation == "crop_2d":
        depth = 2 * fx / (camera[:, 0] * bbox[:, 2] + 1e-9)
        pixel = camera[:, 1:] * bbox[:, 2:] + bbox[:, :2]
        xy = depth[:, None] * (pixel - center) / torch.stack((fx, fy), dim=-1)
    else:
        raise ValueError(f"Unknown pose camera representation {representation!r}")
    return torch.cat((xy, depth[:, None]), dim=-1)


def project_pose(joints_camera, intrinsics, bbox, *, minimum_depth=1e-4):
    """Return full-image pixels, crop-centered coordinates and visibility mask.

    Nonfinite or behind-camera joints have NaN coordinates and an explicit
    false validity mask, matching the hand inference output contract.
    """
    points, intrinsics, bbox = joints_camera.float(), intrinsics.float(), bbox.float()
    valid = torch.isfinite(points).all(dim=-1) & (points[..., 2] > minimum_depth)
    safe = torch.where(valid[..., None], points, points.new_tensor([0, 0, 1]))
    pixels = perspective_projection(safe, intrinsics)
    valid = valid & torch.isfinite(pixels).all(dim=-1)
    pixels = pixels.masked_fill(~valid[..., None], float("nan"))
    normalized = (pixels - bbox[:, None, :2]) / bbox[:, None, 2:]
    return pixels, normalized, valid
