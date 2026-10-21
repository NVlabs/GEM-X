# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEM-Motion neural network, with no training or body-model imports."""

from .r3_matched import R3MatchedGEMNetwork

__all__ = ["R3MatchedGEMNetwork"]
