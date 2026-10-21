# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-body/dense-root supervision and per-source static-contact clocks."""

import torch

from gem.common.geometry.rotations import axis_angle_to_matrix
from gem.motion.codec.soma_native_fixed_gs1 import root_motion_rows


def root_motion_simple_mask(inputs, mask_simple, layout):
    """Dense root fields have their own masks; body and shape remain sparse."""
    rows = root_motion_rows(inputs)
    if rows is None:
        return mask_simple
    masks = inputs["mask"]
    result = mask_simple.clone()
    for field, valid in (
        ("global_orient_c_r6d", masks["root_orientation"]),
        ("global_orient_gv_r6d", masks["root_orientation"]),
        ("local_transl_vel", masks["root_interval"]),
    ):
        selected = layout.field_slice(field)
        value = (valid & masks["valid"])[..., None].expand_as(result[..., selected])
        result[..., selected] = torch.where(rows[:, None, None], value, result[..., selected])
    return result


def incam_geometry_root_conditioning(inputs, predicted_params):
    """Use original GT root orientation for dense-root rows with sparse body labels only.

    Body pose, shape and camera translation retain their predicted values.
    New camera/gravity root outputs are supervised by the coherent SDK root
    targets instead. This does not change prediction/inference dictionaries.
    """
    rows = root_motion_rows(inputs)
    if rows is None:
        return predicted_params
    masks = inputs["mask"]
    selected = rows[:, None] & masks["valid"] & masks["label_valid"]
    rotation = torch.where(
        selected[..., None],
        inputs["soma_params_c"]["global_orient"].detach(),
        predicted_params["global_orient"],
    )
    return {**predicted_params, "global_orient": rotation}


def segmented_root_rollout(
    local_displacement, global_orient, target_translation, position_valid, interval_valid
):
    """Anchor each continuous valid interval chain at its own GT start.

    There is no supervision through a missing label interval. Subsequent valid
    segments receive a new GT origin, so a masked intermediate prediction cannot
    affect later world-position loss. The final outgoing placeholder is ignored.
    """
    rotation = axis_angle_to_matrix(global_orient)
    world_displacement = torch.einsum("blij,blj->bli", rotation, local_displacement)
    world_displacement = world_displacement.masked_fill(~interval_valid[..., None], 0)
    prefix = torch.cat(
        (torch.zeros_like(world_displacement[:, :1]), world_displacement[:, :-1].cumsum(dim=1)),
        dim=1,
    )
    previous_connected = torch.cat(
        (torch.zeros_like(interval_valid[:, :1]), interval_valid[:, :-1]), dim=1
    )
    starts = position_valid & ~previous_connected
    indices = torch.arange(position_valid.shape[1], device=position_valid.device)[None]
    anchors = torch.where(starts, indices, torch.zeros_like(indices)).cummax(dim=1).values
    gather = anchors[..., None].expand(-1, -1, 3)
    return prefix - prefix.gather(1, gather) + target_translation.gather(1, gather)


def get_static_joint_mask(w_j3d, vel_thr=0.25, repeat_last=False, interval_seconds=None):
    """
    w_j3d: (*, L, J, 3)
    vel_thr: Threshold is in metres per second; the formal recipe uses 0.15m/s
    """
    displacement = (w_j3d[..., 1:, :, :] - w_j3d[..., :-1, :, :]).pow(2).sum(-1).sqrt()
    if interval_seconds is None:
        joint_v_ = displacement / 0.033  # (*, L-1, J), legacy unchanged
    elif w_j3d.shape[-3] == 1:
        return (
            torch.zeros(w_j3d.shape[:-1], device=w_j3d.device, dtype=torch.bool)
            if repeat_last
            else displacement.bool()
        )
    else:
        joint_v_ = displacement / interval_seconds[..., None]

    static_joint_mask = joint_v_ < vel_thr  # 1 as stable, 0 as moving

    if repeat_last:  # repeat the last frame, this makes the shape same as w_j3d
        static_joint_mask = torch.cat([static_joint_mask, static_joint_mask[..., [-1], :]], dim=-2)

    return static_joint_mask


def override_source_static_time(static_gt, world_joints, inputs, vel_thr, fps_by_data_name=None):
    """Replace matched rows only; callers retain their existing masks/reduction.

    Configured sources supply contiguous original-frame windows. Their padding
    and final-frame convention use the existing static-target implementation;
    the caller's valid-frame mask still excludes padding from the loss.
    """
    if not fps_by_data_name:
        return static_gt
    rows = [
        index for index, meta in enumerate(inputs["meta"]) if meta["data_name"] in fps_by_data_name
    ]
    if not rows:
        return static_gt
    indices = torch.tensor(rows, device=world_joints.device)
    fps = world_joints.new_tensor(
        [fps_by_data_name[inputs["meta"][index]["data_name"]] for index in rows]
    )
    intervals = fps.reciprocal()[:, None].expand(-1, world_joints.shape[1] - 1)
    result = static_gt.clone()
    result[indices] = get_static_joint_mask(
        world_joints[indices],
        vel_thr=vel_thr,
        repeat_last=True,
        interval_seconds=intervals,
    )
    return result
