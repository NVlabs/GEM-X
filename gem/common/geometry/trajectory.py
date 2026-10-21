# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local displacements and rollout in metres per sampled interval."""
import torch

from gem.common.geometry.rotations import axis_angle_to_matrix


def get_local_transl_vel(transl, global_orient):
    """
    Translation displacement is expressed in the local root coordinate frame.
    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3)
    Returns:
        transl_vel: (*, L, 3)
    """
    if transl.shape[-2] == 1:
        return torch.zeros_like(transl)
    assert len(transl.shape) == len(global_orient.shape)
    global_orient_R = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]  # (B, L-1, 3)
    transl_vel = torch.cat(
        [transl_vel, transl_vel[..., [-1], :]], dim=-2
    )  # (B, L, 3)  last-padding

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    """
    Translation displacement is expressed in the local root coordinate frame.
    Args:
        local_transl_vel: (*, L, 3)
        global_orient: (*, L, 3)
        transl_0: (*, 1, 3), if not provided, the start point is 0
    Returns:
        transl: (*, L, 3)
    """
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)

    # set start point
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)

    # rollout from start point
    transl = torch.cumsum(transl_, dim=-2)
    return transl
