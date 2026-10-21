# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared image-to-2D-observation frontends."""

from .native77 import Native77Pose2D, load_native77_detector

__all__ = ["Native77Pose2D", "load_native77_detector"]
