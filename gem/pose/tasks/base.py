# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-specific decoder ownership; training/FK adapters are separate."""

from torch import Tensor, nn

from ..heads.output import PoseHeadOutput


class PoseTask(nn.Module):
    def __init__(self, name: str, head: nn.Module):
        super().__init__()
        self.name = name
        self.head = head

    def forward(
        self, features: Tensor, *, cliff_cond: Tensor | None = None, ray_feats: Tensor | None = None
    ) -> PoseHeadOutput:
        return self.head(features, cliff_cond=cliff_cond, ray_feats=ray_feats)
