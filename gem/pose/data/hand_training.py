# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared canonical-left hand samples for the shared Pose training loader."""

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

_SHAPES = {
    "img": (3, 256, 256),
    "bbx_xys": (3,),
    "K_fullimg": (3, 3),
    "img_size": (2,),
    "hand_poses": (25, 3, 3),
    "hand_identity_coeffs": (5,),
    "hand_global_scale": (1,),
    "hand_scale_params": (26,),
    "pred_cam_gt": (3,),
    "keypoints_2d": (25, 3),
    "keypoints_3d": (25, 4),
    "has_hand_params": (),
    "has_cam_gt": (),
}
_FINITE_TARGETS = (
    "bbx_xys",
    "K_fullimg",
    "img_size",
    "hand_poses",
    "hand_identity_coeffs",
    "hand_global_scale",
    "hand_scale_params",
    "pred_cam_gt",
    "has_hand_params",
    "has_cam_gt",
)


def _prepared_hand_sample(sample):
    if not isinstance(sample, Mapping):
        raise TypeError("A prepared hand sample must be a mapping")
    for name, shape in _SHAPES.items():
        value = sample[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Prepared hand {name} must be a tensor shaped {shape}")
    if not sample["img"].is_floating_point():
        raise ValueError("Prepared hand img must contain normalized floating-point RGB")
    for name in _FINITE_TARGETS:
        if not bool(torch.isfinite(sample[name]).all()):
            raise ValueError(f"Prepared hand {name} must be finite, including masked rows")
    for name in ("keypoints_2d", "keypoints_3d"):
        if not bool(torch.isfinite(sample[name][..., -1]).all()):
            raise ValueError(f"Prepared hand {name} confidence must be finite")
    if float(sample["bbx_xys"][2]) <= 0:
        raise ValueError("Prepared hand bbx_xys must have positive square size")
    if not bool((sample["img_size"] > 0).all()):
        raise ValueError("Prepared hand img_size[W,H] must be positive")
    if float(sample["hand_global_scale"][0]) <= 0:
        raise ValueError("Prepared hand global scale must be positive")
    # Local MHR controls have a zero neutral value and may be negative.
    # Full-image camera roll is already part of every prepared camera/label.
    rotation = torch.as_tensor(sample.get("aug_rot", 0.0), dtype=torch.float32)
    if rotation.ndim != 0 or float(rotation) != 0.0:
        raise ValueError("Prepared hand aug_rot must be zero; bake full-image camera roll upstream")
    if "data_cursor" in sample:
        raise ValueError("data_cursor is reserved for the Pose training loader")
    result = copy.deepcopy(dict(sample))
    result["aug_rot"] = rotation.clone()
    return result


class PreparedHandPoseDataset(Dataset):
    """Read finite prepared MHR-hand rows without applying augmentation or FK.

    ``path`` contains ``{'samples': [sample, ...]}``. Each image is normalized
    RGB256 in the canonical-left camera frame. ``hand_poses`` contains wrist
    then 24 local finger rotations as matrices, with five identity coefficients,
    one positive global scale and 26 signed local MHR controls (neutral zero).

    ``bbx_xys`` and ``K_fullimg`` describe that same prepared camera frame;
    ``pred_cam_gt`` is legacy ``[s, tx, ty]``. Both keypoint arrays carry the
    native 25-joint order; the objective rebuilds their coordinates from GT FK
    and reads only their confidence channels. Values above one are retained.
    Parameters and camera targets remain finite even when their flags are zero.

    Right-hand mirroring and full-image roll belong to the external preparation
    boundary. They must already be applied consistently to RGB, camera and GT.
    ``aug_rot`` is absent or zero, so the trainer never repeats these transforms.
    Optional metadata such as original side is preserved, not interpreted.
    Use this dataset with ``PoseTrainingDataModule`` from ``.training`` for the
    same consumption-based cursor and resume behavior as other Pose tasks.
    """

    def __init__(self, samples: Sequence[Mapping] | None = None, *, path=None):
        if (samples is None) == (path is None):
            raise ValueError("Provide exactly one of samples or path")
        self.path = None if path is None else str(Path(path).resolve())
        if path is not None:
            samples = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["samples"]
        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("Prepared hand samples must be a nonempty finite sequence")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return _prepared_hand_sample(self.samples[index])
