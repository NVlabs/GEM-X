# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native body pose decoder (identity128, local56, fixed global scale)."""

from ..heads.soma_body_head import SOMABodyHeadV2
from .base import PoseTask


class BodyTask(PoseTask):
    def __init__(self, head: SOMABodyHeadV2 | None = None):
        if head is None:
            head = SOMABodyHeadV2(
                num_identity_coeffs=128,
                num_scale_params=56,
                scale_comps_dim=0,
                scale_param_mode="direct",
                scale_parameterization="local56",
                fixed_global_scale=1.0,
                transformer_mlp_dim=1024,
                cliff_cond_dim=3,
                camera_scale_mode="ste_clamp",
            )
        super().__init__("body", head)
