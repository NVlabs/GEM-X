# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SOMA77 contact translation and two-pass CCD corrections.

The caller supplies a complete 30 Hz sequence and the frozen, FP32 Motion body
facade: 77 public-joint parents, 78 transforms including virtual Root, and the runtime
as ``body.soma``. These functions return replacement tensors without modifying
input parameter dictionaries. Result assembly belongs to the inference API.
"""

from __future__ import annotations

import torch

from gem.common.geometry.rotations import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
)

from . import _kinematics as matrix
from ._kinematics import qbetween
from .smoothing import gaussian_smooth

SOMA_CONTACT_JOINT_IDS = (69, 70, 74, 75, 14, 42)
SOMA_LEFT_LEG_CHAIN = (0, 67, 68, 69, 70)
SOMA_RIGHT_LEG_CHAIN = (0, 72, 73, 74, 75)
SOMA_LEFT_ARM_CHAIN = (3, 11, 12, 13, 14)
SOMA_RIGHT_ARM_CHAIN = (3, 39, 40, 41, 42)


class _CCDIK:
    """Position-only two-pass CCD solver."""

    def __init__(
        self,
        local_mat: torch.Tensor,
        parents: list[int],
        target_indices: tuple[int, ...],
        target_positions: torch.Tensor,
        chain: tuple[int, ...],
        *,
        max_iter: int = 2,
    ):
        global_mat = matrix.forward_kinematics(local_mat, parents)
        chain_local = local_mat[..., chain, :, :].clone()
        chain_local[..., 0, :, :] = global_mat[..., chain[0], :, :]

        self.local_mat = chain_local
        self.parents = [-1, *range(len(chain) - 1)]
        self.global_mat = matrix.forward_kinematics(self.local_mat, self.parents)
        self.target_indices = target_indices
        self.target_positions = target_positions
        self.max_iter = max_iter

    def solve(self) -> torch.Tensor:
        for _ in range(self.max_iter):
            self._optimize(1)
        return self.local_mat

    def _optimize(self, joint_index: int) -> None:
        if joint_index == self.local_mat.shape[-3] - 1:
            return

        position = matrix.get_position(self.global_mat)[..., joint_index, :]
        rotation = matrix.get_rotation(self.global_mat)[..., joint_index, :, :]
        quaternion = matrix_to_quaternion(rotation)

        x_axis = torch.zeros_like(position)
        x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(position)
        y_axis[..., 1] = 1.0
        x_sum = torch.zeros_like(position)
        y_sum = torch.zeros_like(position)
        target_count = 0

        for target_offset, target_index in enumerate(self.target_indices):
            if joint_index >= target_index:
                continue
            end_position = matrix.get_position(self.global_mat)[..., target_index, :]
            target_position = self.target_positions[..., target_offset, :]
            target_rotation = quaternion_multiply(
                qbetween(end_position - position, target_position - position),
                quaternion,
            )
            weight = torch.as_tensor(
                (joint_index + 1) / self.local_mat.shape[-3],
                dtype=quaternion.dtype,
                device=quaternion.device,
            )
            solved = matrix.slerp(quaternion, target_rotation, weight)
            x_sum += quaternion_apply(solved, x_axis)
            y_sum += quaternion_apply(solved, y_axis)
            target_count += 1

        if target_count:
            x_axis = torch.nn.functional.normalize(x_sum / target_count, dim=-1)
            y_axis = torch.nn.functional.normalize(y_sum / target_count, dim=-1)
            z_axis = torch.cross(x_axis, y_axis, dim=-1)
            solved_world_rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
            parent_rotation = matrix.get_rotation(self.global_mat)[
                ..., self.parents[joint_index], :, :
            ]
            self.local_mat[..., joint_index, :3, :3] = matrix.get_mat_BtoA(
                parent_rotation,
                solved_world_rotation,
            )
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parents)

        self._optimize(joint_index + 1)


def _soma_forward(soma_model, params):
    with torch.no_grad(), torch.autocast(params["body_pose"].device.type, enabled=False):
        return soma_model(
            **{
                name: params[name].float()
                for name in (
                    "body_pose",
                    "global_orient",
                    "identity_coeffs",
                    "scale_params",
                    "transl",
                )
            },
            return_joints_only=True,
        )


def correct_static_translation(
    params_global: dict[str, torch.Tensor],
    static_conf_logits: torch.Tensor,
    soma_model,
) -> torch.Tensor:
    """Apply contact-weighted root-displacement correction."""

    forward = _soma_forward(soma_model, params_global)
    joints = forward["joints"]
    if static_conf_logits.shape[:2] != joints.shape[:2] or static_conf_logits.shape[-1] != 6:
        raise ValueError(
            "static confidence must have shape (B,L,6), got "
            f"{tuple(static_conf_logits.shape)} for joints {tuple(joints.shape)}"
        )

    contact_joints = joints[..., SOMA_CONTACT_JOINT_IDS, :]
    joint_displacement = contact_joints[:, 1:] - contact_joints[:, :-1]
    logits = static_conf_logits[:, :-1].float().clone()
    positive = logits > 0
    logits = logits.masked_fill(~positive, -1.0e6)
    has_contact = positive.any(dim=-1)
    weights = logits.softmax(dim=-1) * has_contact[..., None]
    contact_displacement = (joint_displacement * weights[..., None]).sum(dim=-2)

    translation = params_global["transl"].float()
    root_displacement = translation[:, 1:] - translation[:, :-1]
    corrected = torch.cumsum(
        torch.cat((translation[:, :1], root_displacement - contact_displacement), dim=1),
        dim=1,
    )
    corrected[..., 0] = gaussian_smooth(corrected[..., 0], dim=-1)
    corrected[..., 2] = gaussian_smooth(corrected[..., 2], dim=-1)

    corrected_joints = joints - translation[..., None, :] + corrected[..., None, :]
    ground_y = corrected_joints[..., 1].flatten(-2).min(dim=-1).values
    corrected[..., 1] -= ground_y[:, None]
    return corrected.to(params_global["transl"])


def _world_to_local(world_mat: torch.Tensor, parents: list[int]) -> torch.Tensor:
    local_mat = world_mat.clone()
    for joint_index, parent_index in enumerate(parents):
        if parent_index == -1:
            local_mat[..., joint_index, :, :] = world_mat[..., joint_index, :, :]
        else:
            local_mat[..., joint_index, :, :] = matrix.get_mat_BtoA(
                world_mat[..., parent_index, :, :],
                world_mat[..., joint_index, :, :],
            )
    return local_mat


def _solve_chain(
    local_mat: torch.Tensor,
    parents: list[int],
    target_positions: torch.Tensor,
    target_indices: tuple[int, ...],
    chain: tuple[int, ...],
) -> torch.Tensor:
    solved = _CCDIK(
        local_mat,
        parents,
        target_indices,
        target_positions,
        chain,
        max_iter=2,
    ).solve()
    output = local_mat.clone()
    output[..., chain[1:], :3, :3] = solved[..., 1:, :3, :3]
    return output


def correct_static_pose(
    params_global: dict[str, torch.Tensor],
    static_conf_logits: torch.Tensor,
    soma_model,
) -> torch.Tensor:
    """Apply contact-target blending and two-pass CCD to SOMA77."""

    forward = _soma_forward(soma_model, params_global)
    joints = forward["joints"]
    world_mat = forward["transforms"][..., 1:, :, :]
    parents = list(soma_model.parents)
    if len(parents) != 77 or world_mat.shape[-3] != 77:
        raise ValueError(
            f"SOMA post-processing requires the public 77-joint tree, got {len(parents)}"
        )

    static_conf = static_conf_logits.sigmoid().float()
    target_joints = joints.clone()
    for frame_index in range(1, joints.shape[1]):
        previous = target_joints[:, frame_index - 1, SOMA_CONTACT_JOINT_IDS]
        current = joints[:, frame_index, SOMA_CONTACT_JOINT_IDS]
        confidence = static_conf[:, frame_index - 1, :, None]
        target_joints[:, frame_index, SOMA_CONTACT_JOINT_IDS] = previous * confidence + current * (
            1.0 - confidence
        )

    local_mat = _world_to_local(world_mat, parents)
    local_mat = _solve_chain(
        local_mat,
        parents,
        target_joints[..., (69, 70), :],
        (3, 4),
        SOMA_LEFT_LEG_CHAIN,
    )
    local_mat = _solve_chain(
        local_mat,
        parents,
        target_joints[..., (74, 75), :],
        (3, 4),
        SOMA_RIGHT_LEG_CHAIN,
    )
    local_mat = _solve_chain(
        local_mat,
        parents,
        target_joints[..., (14,), :],
        (4,),
        SOMA_LEFT_ARM_CHAIN,
    )
    local_mat = _solve_chain(
        local_mat,
        parents,
        target_joints[..., (42,), :],
        (4,),
        SOMA_RIGHT_ARM_CHAIN,
    )

    procedural = soma_model.soma.procedural_transforms
    if procedural is None:
        raise RuntimeError("SOMA post-processing requires the public procedural rig contract")
    from soma.geometry.rig_utils import remove_joint_orient_local

    absolute_local_rotation = matrix.get_rotation(local_mat).reshape(-1, 77, 3, 3)
    relative_rotation = remove_joint_orient_local(
        absolute_local_rotation,
        procedural.source_joint_orient[1:].to(absolute_local_rotation),
        procedural.source_joint_orient_parent_t[1:].to(absolute_local_rotation),
    )
    relative_rotation = relative_rotation.reshape(*local_mat.shape[:2], 77, 3, 3)
    relative_axis_angle = matrix_to_axis_angle(relative_rotation)
    return relative_axis_angle[..., 1:, :].flatten(-2).to(params_global["body_pose"])
