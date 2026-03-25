# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cyclic Coordinate Descent (CCD) Inverse Kinematics solver."""

import torch

import gem.utils.matrix as matrix
from gem.utils.geo_transform import qbetween
from gem.utils.quaternion import qinv, qmul, qrot, qslerp
from gem.utils.rotation_conversions import matrix_to_quaternion


class CCD_IK:
    def __init__(
        self,
        local_mat,
        parent,
        target_ind,
        target_pos=None,
        target_rot=None,
        kinematic_chain=None,
        max_iter=2,
        threshold=0.001,
        pos_weight=1.0,
        rot_weight=0.0,
    ):
        if kinematic_chain is None:
            kinematic_chain = range(local_mat.shape[-3])
        global_mat = matrix.forward_kinematics(local_mat, parent)

        # Extract kinematic chain and assign root as global mat (do not modify root during IK)
        local_mat = local_mat.clone()
        local_mat = local_mat[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        parent = [i - 1 for i in range(len(kinematic_chain))]
        self.local_mat = local_mat
        self.global_mat = matrix.forward_kinematics(local_mat, parent)  # (*, J, 4, 4)
        self.parent = parent

        self.target_ind = target_ind
        self.target_pos = target_pos  # (*, O, 3) or None
        if target_rot is not None:
            self.target_q = matrix_to_quaternion(target_rot)  # (*, O, 4)
        else:
            self.target_q = None

        self.threshold = threshold
        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_ind)
        self.max_iter = max_iter
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def solve(self):
        for _ in range(self.max_iter):
            self._optimize(1)
        return self.local_mat

    def _optimize(self, i):
        if i == self.J_N - 1:
            return
        pos = matrix.get_position(self.global_mat)[..., i, :]  # (*, 3)
        rot = matrix.get_rotation(self.global_mat)[..., i, :, :]  # (*, 3, 3)
        quat = matrix_to_quaternion(rot)  # (*, 4)
        x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        x_vec[..., 0] = 1.0
        x_vec_sum = torch.zeros_like(x_vec)
        y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        y_vec[..., 1] = 1.0
        y_vec_sum = torch.zeros_like(y_vec)

        count = 0

        for target_i, j in enumerate(self.target_ind):
            if i >= j:
                continue
            end_pos = matrix.get_position(self.global_mat)[..., j, :]  # (*, 3)
            end_rot = matrix.get_rotation(self.global_mat)[..., j, :, :]  # (*, 3, 3)
            end_quat = matrix_to_quaternion(end_rot)  # (*, 4)

            if self.target_pos is not None:
                target_pos = self.target_pos[..., target_i, :]  # (*, 3)
                solved_pos_target_quat = qslerp(
                    quat,
                    qmul(qbetween(end_pos - pos, target_pos - pos), quat),
                    self._get_weight(i),
                )

                x_vec_sum += qrot(solved_pos_target_quat, x_vec)
                y_vec_sum += qrot(solved_pos_target_quat, y_vec)
                if self.pos_weight > 0:
                    count += 1

            if self.target_q is not None:
                if target_i < self.target_N - 1:
                    continue
                target_q = self.target_q[..., target_i, :]  # (*, 4)
                solved_q_target_quat = qslerp(
                    quat,
                    qmul(qmul(target_q, qinv(end_quat)), quat),
                    self._get_weight(i),
                )
                x_vec_sum += qrot(solved_q_target_quat, x_vec) * self.rot_weight
                y_vec_sum += qrot(solved_q_target_quat, y_vec) * self.rot_weight
                if self.rot_weight > 0:
                    count += 1

        if count > 0:
            x_vec_avg = matrix.normalize(x_vec_sum / count)
            y_vec_avg = matrix.normalize(y_vec_sum / count)
            z_vec_avg = torch.cross(x_vec_avg, y_vec_avg, dim=-1)
            solved_rot = torch.stack([x_vec_avg, y_vec_avg, z_vec_avg], dim=-1)

            parent_rot = matrix.get_rotation(self.global_mat)[..., self.parent[i], :, :]
            solved_local_rot = matrix.get_mat_BtoA(parent_rot, solved_rot)
            self.local_mat[..., i, :-1, :-1] = solved_local_rot
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent)
        self._optimize(i + 1)

    def _get_weight(self, i):
        return (i + 1) / self.J_N
