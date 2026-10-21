# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SOMA25 online hand metrics and corresponding-joint error helpers.

Every retained sample has equal weight; all joints, including fingertips and
the wrist, enter each joint mean.
"""

import torch
from torch import Tensor

from gem.common.evaluation import mean_joint_error, similarity_align

SOMA25_ONLINE = "soma25_online"


def hand_joint_errors(predicted: Tensor, target: Tensor, *, valid: Tensor | None = None):
    """Wrist-centered MPJPE/PA-MPJPE in mm for corresponding prepared joints.

    Inputs are B×J×3 in meters, joint0 is the wrist, and both arrays describe
    the same physical hand side in the same camera axes. The function performs
    no joint conversion, mirroring, camera rotation or per-joint visibility
    weighting.
    ``valid[B]`` selects complete labeled samples before alignment; excluded
    rows return NaN so a caller can count and aggregate retained samples.
    """
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("Hand joints must have matching B×J×3 shapes")
    count, device = predicted.shape[0], predicted.device
    if valid is None:
        valid = torch.ones(count, dtype=torch.bool, device=device)
    elif valid.shape != (count,) or valid.dtype != torch.bool:
        raise ValueError("Hand sample validity must be bool[B]")
    else:
        valid = valid.to(device)
    target = target.to(device=device)
    result = {
        name: torch.full((count,), float("nan"), dtype=torch.float32, device=device)
        for name in ("mpjpe_mm", "pa_mpjpe_mm")
    }
    if not bool(valid.any()):
        return result
    with torch.autocast(device_type=device.type, enabled=False):
        prediction, truth = predicted[valid].float(), target[valid].float()
        if not bool(torch.isfinite(prediction).all() & torch.isfinite(truth).all()):
            raise ValueError("Selected hand metric samples must contain finite joints")
        prediction = prediction - prediction[:, :1]
        truth = truth - truth[:, :1]
        result["mpjpe_mm"][valid] = mean_joint_error(prediction, truth) * 1000
        result["pa_mpjpe_mm"][valid] = (
            mean_joint_error(similarity_align(prediction, truth), truth) * 1000
        )
    return result


class HandEvaluator:
    """Evaluate actual-side ``PosePrediction`` against prepared camera GT.

    ``soma25_online`` reads native prediction joints and camera translation,
    comparing with ``batch['gt_joints_camera']`` (B×25×3 meters) and
    ``batch['gt_camera_translation']`` (B×3 meters). RTE is translation error,
    not an aligned trajectory metric. Optional ``batch['valid']`` is bool[B].

    Reader visibility/side filtering happens before this boundary.
    """

    def __init__(self, *, protocol=SOMA25_ONLINE):
        if protocol != SOMA25_ONLINE:
            raise ValueError(f"Unsupported hand protocol {protocol!r}; no mapping is registered")
        self.protocol = protocol

    def evaluate(self, prediction, batch):
        if prediction.task != "hand":
            raise ValueError("HandEvaluator requires a hand prediction")
        joints = prediction.joints_camera
        if joints.ndim != 3 or joints.shape[1:] != (25, 3):
            raise ValueError(f"{self.protocol} requires B×25×3 prediction joints")
        valid = batch.get("valid")
        result = hand_joint_errors(joints, batch["gt_joints_camera"], valid=valid)
        target = batch["gt_camera_translation"].to(prediction.camera_translation)
        if target.shape != (len(joints), 3):
            raise ValueError("Hand GT camera translation must have shape B×3")
        with torch.autocast(device_type=joints.device.type, enabled=False):
            error = (
                torch.linalg.vector_norm(
                    prediction.camera_translation.float() - target.float(), dim=-1
                )
                * 1000
            )
        if valid is not None:
            error = error.masked_fill(~valid.to(error.device), float("nan"))
        result["rte_mm"] = error
        return result
