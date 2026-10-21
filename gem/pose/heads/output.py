# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named decoder outputs shared by body and hand tasks."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class PoseHeadOutput:
    parameters: dict[str, Tensor]
    pred_cam: Tensor
    auxiliary: dict[str, Any]
    token: Tensor
    joint_tokens: Tensor | None = None


@dataclass
class ImageCondition:
    """The final decoder query, crop camera and native model parameters.

    These are predictions in the supplied crop/canonical camera frame. Hand
    right-side reconstruction and body-model FK belong to the task predictor.
    """

    pose_tokens: Tensor
    pred_cam: Tensor
    parameters: dict[str, Tensor]
    task: str
    pose_release_id: str | None

    @property
    def feature_version(self) -> str | None:
        """Identify final-query features in the existing Motion cache format."""
        if self.pose_release_id is None:
            return None
        return f"{self.pose_release_id}:pose_tokens"
