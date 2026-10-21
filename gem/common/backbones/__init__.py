# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbones shared by image pose and 2D observation models."""

from .dinov3 import DinoV3Backbone, convert_dinov3_native_state_dict, dinov3_vith16plus_config

__all__ = ["DinoV3Backbone", "convert_dinov3_native_state_dict", "dinov3_vith16plus_config"]
