# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, input-only per-frame observation augmentation for image supervision."""

from __future__ import annotations

import torch

from gem.motion.frontend.joint_sets import SAPIENS70_NOFINGER30_V1, get_observation_schema

from .observation_noise import get_observation_visible_mask
from .observation_noise import (
    make_observation_noise_generator as make_observation_noise_generator,
)

OBSERVATION_AUGMENTATION_MODES = ("none", "common2d_v1")
AUGMENTATION_SEED_OFFSET = 10_000_019
_SCHEMA = "sapiens70_nofinger30_v1"
_XY_STD = 0.02
_JOINT_MISSING_PROBABILITY = 0.03
_LOWER_BODY_HIDE_PROBABILITY = 0.015


def observation_augmentation_profile(mode: str, *, schema=None) -> dict:
    """Return JSON-friendly policy metadata without importing the reference."""
    if mode not in OBSERVATION_AUGMENTATION_MODES:
        raise ValueError(f"Unknown image observation augmentation mode: {mode!r}")
    return {
        "mode": mode,
        "enabled": mode != "none",
        "coord_space": "crop_normalized_minus_one_to_one",
        "observation_schema": _SCHEMA if schema is None else get_observation_schema(schema).name,
        "seed_offset": AUGMENTATION_SEED_OFFSET,
        "source_independent": True,
        "xy_gaussian_std": _XY_STD if mode != "none" else 0.0,
        "joint_missing_probability": _JOINT_MISSING_PROBABILITY if mode != "none" else 0.0,
        "lower_body_hide_probability": _LOWER_BODY_HIDE_PROBABILITY if mode != "none" else 0.0,
        "original_visibility": "confidence_strictly_greater_than_0.5",
        "joint_missing_policy": "reference_semantic_tree_on_new_random_missingness",
        "lower_body_policy": "single_frame_bernoulli_over_reference_lower_body_indices",
        "draw_order": [
            "full_shape_xy_gaussian",
            "full_shape_joint_visibility",
            "per_frame_lower_body",
        ],
        "crop_boundary_policy": "no_clipping_or_additional_mask",
        "surviving_confidence": "unchanged",
        "hidden_observation": "all_three_values_zero",
        "mutates_supervision": False,
    }


def augment_image_observations(
    obs: torch.Tensor, mode: str, *, generator: torch.Generator | None = None, schema=None
) -> torch.Tensor:
    """Perturb ``obs[B,T,J,3]`` (T=1 or3), drawing at the schema's joint count.

    All random tensors are drawn at full shape before consulting visibility.
    Original invisible points stay absent, but do not cause extra propagation
    into originally visible children. New missingness follows the exact
    reference tree. Gaussian XY may leave the crop; no clipping or additional
    crop gate is applied. ``none`` is a strict no-op, including object identity.
    """
    if mode == "none":
        return obs
    if mode != "common2d_v1":
        raise ValueError(f"Unknown image observation augmentation mode: {mode!r}")
    if generator is None:
        raise ValueError("Enabled image observation augmentation requires an explicit generator")
    schema = get_observation_schema(SAPIENS70_NOFINGER30_V1 if schema is None else schema)
    if obs.ndim != 4 or obs.shape[1] not in (1, 3) or obs.shape[2:] != (schema.num_joints, 3):
        raise ValueError(
            f"Image observation augmentation requires shape (B,T,{schema.num_joints},3), T=1 or3"
        )

    noise = torch.randn(*obs.shape[:-1], 2, device=obs.device, dtype=obs.dtype, generator=generator)
    visible = get_observation_visible_mask(
        obs.shape[:2],
        schema,
        mask_probability=_JOINT_MISSING_PROBABILITY,
        device=obs.device,
        generator=generator,
    )
    lower_hidden = (
        torch.rand(*obs.shape[:2], 1, device=obs.device, generator=generator)
        < _LOWER_BODY_HIDE_PROBABILITY
    )
    visible[..., list(schema.lower_body_indices)] &= ~lower_hidden
    visible &= obs[..., 2] > 0.5

    changed = obs.clone()
    changed[..., :2] += noise * _XY_STD
    return changed.masked_fill(~visible[..., None], 0)


# Compatibility names for the historical compact30 image policy.
VISIBILITY_PARENTS = SAPIENS70_NOFINGER30_V1.visibility_parents
LOWER_BODY_INDICES = SAPIENS70_NOFINGER30_V1.lower_body_indices


def compact30_visible_mask(shape, *, mask_probability=0.03, device=None, generator=None):
    """Draw full-shape missingness, then propagate in the deployed tree order."""
    return get_observation_visible_mask(
        shape,
        SAPIENS70_NOFINGER30_V1,
        mask_probability=mask_probability,
        device=device,
        generator=generator,
    )
