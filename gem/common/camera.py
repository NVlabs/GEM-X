# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera geometry shared by pose and motion.

Weak-perspective conversion is explicit. Full-perspective pose heads already
predict translation and must not be passed through compute_transl_full_cam.
"""
import math

import torch


def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    p2d: (..., (N), 2)
    bbx_lurb: (..., 4)
    """
    if len(p2d.shape) == len(bbx_lurb.shape) + 1:
        bbx_lurb = bbx_lurb[..., None, :]

    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]
    bi01_p2d = (p2d - bbx_lurb[..., :2]) / bbx_wh
    return bi01_p2d


def convert_bbx_xys_to_lurb(bbx_xys):
    """
    Args: bbx_xys (..., 3) -> bbx_lurb (..., 4)
    """
    size = bbx_xys[..., 2:]
    center = bbx_xys[..., :2]
    lurb = torch.cat([center - size / 2, center + size / 2], dim=-1)
    return lurb


def perspective_projection(points, K):
    """Standard pinhole camera projection.

    Args:
        points: (*, J, 3)
        K: (*, 3, 3)
    Returns:
        p2d: (*, J, 2)
    """
    projected = points / points[..., -1:].clamp(min=1e-6)
    projected = torch.einsum("...ij,...kj->...ki", K, projected.float())
    return projected[..., :2]


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    """Convert weak-perspective camera parameters to full-perspective translation.

    Args:
        pred_cam: (*, 3)  [s, tx, ty]
        bbx_xys: (*, 3)  [cx, cy, size]
        K_fullimg: (*, 3, 3)
    Returns:
        cam_t: (*, 3)
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse of compute_transl_full_cam: full translation → weak-perspective params.

    Args:
        transl: (*, 3)
        bbx_xys: (*, 3)
        K_fullimg: (*, 3, 3)  one extra leading dim compared to transl/bbx_xys
    Returns:
        pred_cam: (*, 3)  [s, tx, ty]
    """
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    return torch.stack([gt_s, gt_x, gt_y], dim=-1)


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """Compute BEDLAM-style bounding box info (normalised by focal length).

    Args:
        bbx_xys: (*, 3)  [cx, cy, size]
        K_fullimg: (*, 3, 3)
    Returns:
        bbox_info: (*, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    return bbox_info / fl


def estimate_K(width, height):
    """Estimate default pinhole intrinsics from image dimensions.

    Uses the image diagonal as focal length, matching the motion model's
    training and deployed preprocessing convention.

    Args:
        width, height: image dimensions (int or float)
    Returns:
        K: (3, 3) camera intrinsics tensor
    """
    focal = math.hypot(width, height)
    K = torch.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(width) / 2.0
    K[1, 2] = float(height) / 2.0
    return K


def project_to_bi01(points, bbx_xys, K_fullimg):
    """Project 3D points to bounding-box-normalised [0,1] 2D coordinates.

    Motion's full-frame loss projects signed depths before masking invalid
    points. Pixel coordinates can exceed FP16 range even when the geometry is
    finite, so projection and crop normalization must run in float32.

    Args:
        points: (*, J, 3)
        bbx_xys: (*, 3)
        K_fullimg: (*, 3, 3)
    Returns:
        p2d_bi01: (*, J, 2)
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        projected = points / points[..., -1:]
        pixels = torch.einsum("...ij,...kj->...ki", K_fullimg.float(), projected)[..., :2]
        bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys.float())
        return cvt_to_bi01_p2d(pixels, bbx_lurb)
