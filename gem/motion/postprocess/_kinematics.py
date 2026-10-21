# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only kinematic primitives for CCD inverse kinematics."""

import torch


def normalized_matrix(matrix):
    rotation = matrix[..., :3, :3] if matrix.shape[-1] == 4 else matrix
    rotation = rotation / (rotation.norm(2, dim=-2, keepdim=True) + 1e-9)
    if matrix.shape[-1] == 3:
        return rotation
    result = torch.zeros_like(matrix)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = matrix[..., :3, 3]
    result[..., 3, 3] = 1.0
    return result


def get_mat_BtoA(matrix_a, matrix_b):
    inverse = normalized_matrix(torch.inverse(matrix_a))
    return normalized_matrix(inverse @ matrix_b)


def get_rotation(matrix):
    return matrix[..., :3, :3]


def get_position(matrix):
    return matrix[..., :3, 3]


def forward_kinematics(local, parents):
    global_matrices = torch.eye(local.shape[-1], device=local.device)
    global_matrices = global_matrices.repeat(local.shape[:-2] + (1, 1))
    for index, parent in enumerate(parents):
        if parent == -1:
            global_matrices = torch.cat(
                (local[..., : index + 1, :, :], global_matrices[..., index + 1 :, :, :]), dim=-3
            )
        else:
            matrix = normalized_matrix(global_matrices[..., parent, :, :] @ local[..., index, :, :])
            global_matrices = torch.cat(
                (
                    global_matrices[..., :index, :, :],
                    matrix[..., None, :, :],
                    global_matrices[..., index + 1 :, :, :],
                ),
                dim=-3,
            )
    return global_matrices


def slerp(q0, q1, weight):
    cosine = torch.sum(q0 * q1, dim=-1)
    negative = cosine < 0
    q1 = q1.clone()
    q1[negative] = -q1[negative]
    cosine = cosine.abs().unsqueeze(-1)
    angle = torch.acos(cosine)
    sine = torch.sqrt(1 - cosine * cosine)
    first = torch.sin((1 - weight) * angle) / sine
    second = torch.sin(weight * angle) / sine
    result = first * q0 + second * q1
    result = torch.where(sine.abs() < 0.001, 0.5 * q0 + 0.5 * q1, result)
    return torch.where(cosine.abs() >= 1, q0, result)


def qbetween(first, second):
    vector = torch.cross(first, second, dim=-1)
    scalar = torch.sqrt((first**2).sum(-1, keepdim=True) * (second**2).sum(-1, keepdim=True))
    scalar = scalar + (first * second).sum(-1, keepdim=True)
    # Retain the original y-axis choice for anti-parallel/zero-length inputs.
    y_axis = torch.zeros_like(vector)
    y_axis[..., 1] = 1
    antiparallel = (vector.norm(dim=-1) == 0) & (scalar.sum(-1).abs() <= 1e-4)
    vector = vector.clone()
    vector[antiparallel] = y_axis[antiparallel]
    quaternion = torch.cat((scalar, vector), dim=-1)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp(min=1e-8)
