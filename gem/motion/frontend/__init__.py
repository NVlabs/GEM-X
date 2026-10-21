# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation schemas and conditioning inputs for GEM-Motion."""


def encode_observations(observations, coordinate_projection, missing_embeddings):
    """Retain the frontend entry point without importing Torch for metadata."""
    from .observation import encode_observations as encode

    return encode(observations, coordinate_projection, missing_embeddings)


__all__ = ["encode_observations"]
