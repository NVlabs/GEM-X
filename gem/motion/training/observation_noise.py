# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native observation noise with the established bias/peak/jitter draw order.

COCO17 supplies augmentation strengths, not coordinate correspondences or a
measured detector-error distribution. Each schema draws directly at its own
joint count. The small fixed tables are embedded here to avoid runtime assets.
"""

from __future__ import annotations

import torch

from gem.motion.frontend.joint_sets import ObservationSchema, get_observation_schema

# Float32 values of the original four COCO17 augmentation profiles.
_COCO17_AUG = {
    "jittering": torch.tensor(
        (0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.4, 0.4, 0.2, 0.2, 0.1, 0.1),
        dtype=torch.float32,
        device="cpu",
    ),
    "bias": torch.tensor(
        (
            0.4,
            0.1,
            0.1,
            0.1,
            0.1,
            0.15,
            0.15,
            0.25,
            0.25,
            0.35,
            0.35,
            0.1,
            0.1,
            0.15,
            0.15,
            0.25,
            0.25,
        ),
        dtype=torch.float32,
        device="cpu",
    ),
    "peak": torch.tensor(
        (
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.15,
            0.15,
            0.25,
            0.25,
            0.4,
            0.4,
            0.08,
            0.08,
            0.3,
            0.3,
            0.4,
            0.4,
        ),
        dtype=torch.float32,
        device="cpu",
    ),
    "pmask": torch.tensor(
        (0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3),
        dtype=torch.float32,
        device="cpu",
    ),
}

_MAPPED_AUG_PROFILES: dict[ObservationSchema, dict[str, torch.Tensor]] = {}
_DEVICE_AUG_PROFILES: dict[
    tuple[ObservationSchema, torch.device, torch.dtype], dict[str, torch.Tensor]
] = {}


def _resolve_aug_profile_device(device: torch.device | str | None) -> torch.device:
    # The source tables are CPU tensors, so their existing .to(device=None)
    # stays on CPU. Resolve an unindexed CUDA device on every lookup: each
    # training rank may have a different current device.
    resolved = torch.device("cpu") if device is None else torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if resolved.type == "cpu":
        return torch.device("cpu")
    return resolved


def _cached_aug_profiles(
    schema: ObservationSchema, device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    key = (schema, _resolve_aug_profile_device(device), dtype)
    if key not in _DEVICE_AUG_PROFILES:
        if schema not in _MAPPED_AUG_PROFILES:
            _MAPPED_AUG_PROFILES[schema] = {
                name: torch.stack(
                    [source[list(indices)].mean() for indices in schema.coco17_aug_sources]
                )
                for name, source in _COCO17_AUG.items()
            }
        _DEVICE_AUG_PROFILES[key] = {
            name: values.to(device=key[1], dtype=dtype)
            for name, values in _MAPPED_AUG_PROFILES[schema].items()
        }
    return _DEVICE_AUG_PROFILES[key]


def get_coco17_aug_profile(
    name: str,
    schema: str | ObservationSchema,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    cache_profiles: bool = False,
) -> torch.Tensor:
    """Map fixed tables into a schema's joint order.

    The default retains the original per-call mapping/conversion. Opting in
    reuses read-only tables per schema, resolved device and dtype in this
    process. Callers must not modify a cached table in place. No random draws,
    checkpoint state or DDP buffers are added by the cache.
    """

    schema = get_observation_schema(schema)
    try:
        source = _COCO17_AUG[name]
    except KeyError as error:
        raise ValueError(f"Unknown coco_aug_dict profile {name!r}") from error
    if cache_profiles:
        return _cached_aug_profiles(schema, device, dtype)[name]
    values = torch.stack([source[list(indices)].mean() for indices in schema.coco17_aug_sources])
    return values.to(device=device, dtype=dtype)


def get_observation_bias(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    scale: float = 1e-1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    cache_profiles: bool = False,
) -> torch.Tensor:
    """Sequence-static 3D keypoint bias noise."""

    schema = get_observation_schema(schema)
    batch_size, sequence_length = shape
    profile = get_coco17_aug_profile(
        "bias", schema, device=device, dtype=dtype, cache_profiles=cache_profiles
    ).reshape(1, -1, 1)
    bias = torch.randn(
        batch_size,
        schema.num_joints,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return (bias * profile * scale)[:, None].expand(-1, sequence_length, -1, -1)


def get_observation_peak(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    scale: float = 3e-1,
    mask_scale: float = 5e-3,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    cache_profiles: bool = False,
) -> torch.Tensor:
    """Sparse high-peak 3D keypoint noise."""

    schema = get_observation_schema(schema)
    batch_size, sequence_length = shape
    pmask = get_coco17_aug_profile(
        "pmask", schema, device=device, dtype=dtype, cache_profiles=cache_profiles
    )
    peak = get_coco17_aug_profile(
        "peak", schema, device=device, dtype=dtype, cache_profiles=cache_profiles
    ).reshape(1, 1, -1, 1)
    peak_mask = (
        torch.rand(
            batch_size,
            sequence_length,
            schema.num_joints,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * pmask
        < mask_scale
    )
    direction = torch.randn(3, device=device, dtype=dtype, generator=generator)
    return peak_mask[..., None] * direction * peak * scale


def get_observation_jitter(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    scale: float = 5e-2,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    cache_profiles: bool = False,
) -> torch.Tensor:
    """Per-frame 3D keypoint jitter noise."""

    schema = get_observation_schema(schema)
    batch_size, sequence_length = shape
    profile = get_coco17_aug_profile(
        "jittering", schema, device=device, dtype=dtype, cache_profiles=cache_profiles
    ).reshape(1, 1, -1, 1)
    jitter = torch.randn(
        batch_size,
        sequence_length,
        schema.num_joints,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return jitter * profile * scale


def get_observation_aug_kp3d(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    bias_scale: float = 1e-1,
    peak_scale: float = 3e-1,
    peak_mask_scale: float = 5e-3,
    jitter_scale: float = 5e-2,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    cache_profiles: bool = False,
) -> torch.Tensor:
    """Compose bias, peak, and jitter in the legacy random-draw order."""

    bias = get_observation_bias(
        shape,
        schema,
        scale=bias_scale,
        device=device,
        dtype=dtype,
        generator=generator,
        cache_profiles=cache_profiles,
    )
    peak = get_observation_peak(
        shape,
        schema,
        scale=peak_scale,
        mask_scale=peak_mask_scale,
        device=device,
        dtype=dtype,
        generator=generator,
        cache_profiles=cache_profiles,
    )
    jitter = get_observation_jitter(
        shape,
        schema,
        scale=jitter_scale,
        device=device,
        dtype=dtype,
        generator=generator,
        cache_profiles=cache_profiles,
    )
    return bias + peak + jitter


def randomly_modify_observation_hands_legs(
    joints: torch.Tensor,
    schema: str | ObservationSchema,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw six random masks and return the input joints unchanged.

    These draws preserve the observation RNG stream used by later augmentation.
    This helper does not swap or copy hand/leg coordinates.
    """

    schema = get_observation_schema(schema)
    if joints.ndim != 4 or joints.shape[-2:] != (schema.num_joints, 3):
        raise ValueError(
            f"{schema.name} endpoint no-op expects (B, L, {schema.num_joints}, 3), "
            f"got {tuple(joints.shape)}"
        )

    batch_size, sequence_length = joints.shape[:2]
    for _ in range(6):
        torch.rand(batch_size, sequence_length, device=joints.device, generator=generator)
    return joints


def get_observation_visible_mask(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    mask_probability: float = 0.03,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-joint missingness followed by semantic-tree propagation."""

    schema = get_observation_schema(schema)
    batch_size, sequence_length = shape
    visible = ~(
        torch.rand(
            batch_size,
            sequence_length,
            schema.num_joints,
            device=device,
            generator=generator,
        )
        < mask_probability
    )
    flat = visible.reshape(-1, schema.num_joints)
    for child, parents in enumerate(schema.visibility_parents):
        for parent in parents:
            flat[:, child] *= flat[:, parent]
    return flat.reshape(batch_size, sequence_length, schema.num_joints).clone()


def get_observation_lower_body_invisible_mask(
    shape: tuple[int, int],
    schema: str | ObservationSchema,
    *,
    sequence_probability: float = 0.03,
    min_frames: int = 30,
    max_frames: int = 90,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One-interval lower-body occlusion with schema-specific joints."""

    schema = get_observation_schema(schema)
    batch_size, sequence_length = shape
    if min_frames <= 0 or max_frames <= min_frames:
        raise ValueError("lower-body bounds require 0 < min_frames < max_frames")
    if sequence_length <= max_frames:
        raise ValueError("lower-body occlusion requires sequence_length > max_frames")
    starts = torch.randint(
        0,
        sequence_length - max_frames,
        (batch_size,),
        device=device,
        generator=generator,
    )
    durations = torch.randint(
        min_frames,
        max_frames,
        (batch_size,),
        device=device,
        generator=generator,
    )
    timeline = torch.arange(sequence_length, device=device)[None]
    active = (timeline >= starts[:, None]) & (timeline < (starts + durations)[:, None])
    mask = torch.zeros(
        batch_size,
        sequence_length,
        schema.num_joints,
        dtype=torch.bool,
        device=device,
    )
    lower_body = torch.tensor(schema.lower_body_indices, dtype=torch.long, device=device)
    mask.index_fill_(-1, lower_body, True)
    mask &= active[..., None]
    selected = (
        torch.rand(batch_size, 1, 1, device=device, generator=generator) < sequence_probability
    )
    return mask & selected


def make_observation_noise_generator(device, *, seed, attempt_step, global_rank, world_size):
    """Reconstruct independent augmentation RNG from restored training counters."""
    local_seed = int(seed) + int(attempt_step) * int(world_size) + int(global_rank)
    return torch.Generator(device=device).manual_seed(local_seed)
