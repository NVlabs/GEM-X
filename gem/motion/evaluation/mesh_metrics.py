# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional corresponding-mesh errors and foot-contact samples for Motion."""

import numpy as np
import torch

from gem.common.evaluation import mean_joint_error

from .metrics import (
    _camera_errors,
    _get_protocol,
    _prepare_joint_pair,
    _retain_frames,
    compute_soma_global_joint_metrics,
)

SMPL_FOOT_VERTEX_INDICES = (3216, 3387, 6617, 6787)


def _prepare_vertex_pair(batch, *, frame_count, global_coordinates):
    suffix = "_glob" if global_coordinates else ""
    pair = [
        batch[f"{name}{suffix}"].detach().float().cpu() for name in ("pred_verts", "target_verts")
    ]
    if any(vertices.shape != (frame_count, 6890, 3) for vertices in pair):
        raise ValueError("Corresponding mesh vertices must have shape (T,6890,3)")
    return pair


@torch.no_grad()
def compute_soma_camcoord_metrics(batch, *, target_joint_set, mask=None):
    """Camera joint errors/PVE in mm and acceleration error in m/s² at 30 Hz.

    Each mesh is centered on the pelvis regressed from its corresponding
    joints. PVE performs no scale or rotation fitting. Acceleration is measured
    on pelvis-relative joints after compressing invalid frames; fewer than
    three retained frames produce no acceleration samples.
    """
    protocol = _get_protocol(target_joint_set)
    joints = _prepare_joint_pair(batch, protocol, global_coordinates=False)
    vertices = _prepare_vertex_pair(batch, frame_count=len(joints[0]), global_coordinates=False)
    pred_joints, gt_joints, pred_vertices, gt_vertices = _retain_frames([*joints, *vertices], mask)
    pelvis = list(protocol.pelvis_indices)
    pred_root = pred_joints[:, pelvis].mean(1, keepdim=True)
    gt_root = gt_joints[:, pelvis].mean(1, keepdim=True)
    pred_joints, gt_joints = pred_joints - pred_root, gt_joints - gt_root
    result = _camera_errors(pred_joints, gt_joints)
    result["pve"] = (
        mean_joint_error(pred_vertices - pred_root, gt_vertices - gt_root).numpy() * 1000
    )
    pred_accel = pred_joints[:-2] - 2 * pred_joints[1:-1] + pred_joints[2:]
    gt_accel = gt_joints[:-2] - 2 * gt_joints[1:-1] + gt_joints[2:]
    result["accel"] = np.linalg.norm((pred_accel - gt_accel).numpy(), axis=-1).mean(-1) * 30**2
    return result


@torch.no_grad()
def compute_soma_global_metrics(batch, *, target_joint_set, mask=None):
    """World joint errors plus foot displacement per reference contact, in mm.

    A contact is a target foot-vertex displacement strictly below 1 cm between
    consecutive retained frames. Each of the four foot vertices contributes
    independently; absent contacts produce an empty ``fs`` array, not zero.
    """
    result = compute_soma_global_joint_metrics(batch, target_joint_set=target_joint_set, mask=mask)
    pair = _prepare_vertex_pair(
        batch, frame_count=len(batch["pred_j3d_glob"]), global_coordinates=True
    )
    prediction, target = _retain_frames(pair, mask)
    prediction, target = (
        vertices[:, SMPL_FOOT_VERTEX_INDICES] for vertices in (prediction, target)
    )
    contact = (target[1:] - target[:-1]).norm(dim=-1) < 0.01
    displacement = (prediction[1:] - prediction[:-1]).norm(dim=-1)
    result["fs"] = displacement[contact].numpy() * 1000
    return result
