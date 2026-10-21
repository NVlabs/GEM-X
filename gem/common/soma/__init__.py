# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-GS1 SOMA geometry; the official runtime is imported on first use."""

from .body import SomaFixedGs1BodyAdapter, SomaFixedGs1ShapeSpec, SomaRuntimeConfig
from .procedural import configure_procedural_reference

__all__ = [
    "SomaFixedGs1BodyAdapter",
    "SomaFixedGs1ShapeSpec",
    "SomaRuntimeConfig",
    "configure_procedural_reference",
]
