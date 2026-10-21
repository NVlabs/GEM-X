# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confidence-aware 2D observation encoding, independent of the joint source."""

import torch

from .joint_sets import (
    SOMA29_NOFINGER,
    SOMA_OBSERVATION_SCHEMAS,
    get_observation_schema,
    select_observation_joints,
)


def normalize_keypoints(keypoints, bbx_xys):
    """Convert full-image XY/score to the trained [-1, 1] crop convention.

    Both crop boundaries are inclusive. Outside points retain normalized XY
    but receive zero confidence; the observation encoder owns missing tokens.
    ``bbx_xys`` contains the positive crop size from preprocessing.
    """
    xy, confidence = keypoints[..., :2], keypoints[..., 2]
    center, size = bbx_xys[..., :2], bbx_xys[..., 2:]
    upper, lower = center + size / 2, center - size / 2
    outside = (
        (xy[..., 0] < lower[..., None, 0])
        | (xy[..., 0] > upper[..., None, 0])
        | (xy[..., 1] < lower[..., None, 1])
        | (xy[..., 1] > upper[..., None, 1])
    )
    normalized = 2 * (xy - center.unsqueeze(-2)) / size.unsqueeze(-2)
    return torch.cat((normalized, (confidence * ~outside)[..., None]), dim=-1)


def prepare_native_observations(public77, bbx_xys, schema=SOMA29_NOFINGER):
    """Select native joints from a full77 detection cache, then normalize.

    The full77 tensor is left intact for camera masking and other consumers.
    This is inference preparation; no training augmentation is applied.
    """
    schema = get_observation_schema(schema)
    if schema.name not in SOMA_OBSERVATION_SCHEMAS:
        raise ValueError("Full77 detections require a native SOMA29 or SOMA77 schema")
    selected = select_observation_joints(public77, schema)
    return normalize_keypoints(selected, bbx_xys)


def encode_observations(observations, coordinate_projection, missing_embeddings):
    """Encode crop XY; confidence <= 0.5 uses a learned per-joint token.

    Joint ordering is part of the model's input contract. This operation does
    not convert between detector skeletons, even when joint counts match.
    """
    visible = observations[..., [2]] > 0.5
    coordinates = observations[..., :2].masked_fill(~visible, 0)
    encoded = coordinate_projection(coordinates)
    return torch.where(visible, encoded, missing_embeddings).flatten(-2)
