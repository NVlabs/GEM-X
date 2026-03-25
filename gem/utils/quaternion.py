# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quaternion utilities (w, x, y, z convention) for CCD IK solver."""

import torch


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def qmul(q, r):
    """Multiply quaternion(s) q with quaternion(s) r.  Shape: (*, 4)."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """Rotate vector(s) v by quaternion(s) q.  q: (*, 4), v: (*, 3) -> (*, 3)."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qpow(q0, t, dtype=torch.float):
    """Raise quaternion q0 to the power t."""
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., :1])

    # if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    mask = mask.float()
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0)

    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape, device=q0.device)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:
        q = torch.zeros(q0.shape, device=q0.device)
        theta = t * theta0

    q[..., :1] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta)

    return q.to(dtype)


def qslerp(q0, q1, t):
    """Spherical linear interpolation between q0 and q1 at parameter t."""
    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)
    return qmul(q_, q0)
