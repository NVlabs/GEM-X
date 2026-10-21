# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit procedural-twist conventions for historical Pose checkpoints."""

import torch


def configure_procedural_reference(runtime, reference="runtime"):
    """Select the twist binding used by this already constructed body layer.

    ``runtime`` keeps the official current behavior (including GEM-Motion).
    ``legacy_pose021`` preserves the frozen GEP body wrapper's SOMA0.2.1
    convention: procedural twist extraction was bound to the T-pose, while
    SOMA0.3.3 binds it to the bind pose. Public reference rotations alone do
    not select that convention. Only the two twist-binding buffers change;
    public pose, shape, skinning, correctives and assets remain untouched.

    This compatibility profile is validated for official SOMA0.3.3 with the
    pinned release assets. Call once before the layer's first FK evaluation.
    """
    if reference == "runtime":
        return runtime
    if reference != "legacy_pose021":
        raise ValueError(f"Unknown SOMA procedural reference {reference!r}")

    import soma
    from soma.procedural_transforms import (
        _bind_alignment_quaternions,
        matrix_to_quaternion_xyzw,
    )

    if soma.__version__ != "0.3.3":
        raise ValueError("legacy_pose021 compatibility requires the validated SOMA0.3.3 runtime")
    procedural = runtime.procedural_transforms
    if procedural is None:
        raise ValueError("legacy_pose021 requires procedural transforms")
    target_indices = {
        str(name): index for index, name in enumerate(runtime.rig_data["joint_names"])
    }
    public_indices = [target_indices[str(name)] for name in runtime.public_joint_names]
    # The official constructor computes these fixed buffers on CPU before
    # moving the layer; retain that arithmetic for compatibility initialization.
    public_t_pose = torch.as_tensor(
        runtime.rig_data["t_pose_world"], dtype=torch.float32, device="cpu"
    )[public_indices]
    binding = matrix_to_quaternion_xyzw(public_t_pose[..., :3, :3])
    alignment = _bind_alignment_quaternions(
        public_t_pose,
        procedural.segment_start_source_ids.cpu(),
        procedural.segment_end_source_ids.cpu(),
    )
    with torch.no_grad():
        procedural.source_bind_quaternions.copy_(binding.to(procedural.source_bind_quaternions))
        procedural.segment_bind_align_quaternions.copy_(
            alignment.to(procedural.segment_bind_align_quaternions)
        )
    return runtime
