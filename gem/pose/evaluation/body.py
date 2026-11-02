# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-frame Body joint metrics for EMDB and 3DPW, in millimeters."""

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import nn

from gem.common.evaluation import mean_joint_error, scale_align, similarity_align

_PROTOCOLS = {
    "emdb": (24, tuple(range(24)), (1, 2)),
    "3dpw": (44, (*range(25, 38), 43), (39,)),
}


@contextmanager
def _full_precision_matmul(device):
    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class BodyJointProjector(nn.Module):
    """Project local mesh vertices into the asset's complete benchmark joint order.

    Standalone assets contain ``matrix[J,V]``, ``source_num_vertices`` and
    optionally ``face_vertex_indices[F,3]``. Benchmark bundles supply a named
    regressor and shared faces. With faces, the matrix has width ``V+F``;
    each extra point is ``v0 + cross(v1-v0, v2-v0)`` for that triangle.
    Projection uses FP32 without autocast or TF32. Asset buffers are not saved
    as model weights.
    """

    def __init__(self, matrix, source_num_vertices, face_vertex_indices=None):
        super().__init__()
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
        faces = (
            torch.empty(0, 3, dtype=torch.long)
            if face_vertex_indices is None
            else torch.as_tensor(face_vertex_indices, dtype=torch.long)
        )
        self.source_num_vertices = int(source_num_vertices)
        if matrix.ndim != 2 or matrix.shape[0] not in (24, 44):
            raise ValueError("Body projection must produce complete SMPL24 or SMPL44 joints")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("face_vertex_indices must have shape [F,3]")
        if matrix.shape[1] != self.source_num_vertices + len(faces):
            raise ValueError("Joint matrix width must match source vertices plus face points")
        if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= self.source_num_vertices):
            raise ValueError("Face indices must refer to source mesh vertices")
        self.register_buffer("matrix", matrix.contiguous(), persistent=False)
        self.register_buffer("face_vertex_indices", faces.contiguous(), persistent=False)

    @classmethod
    def from_asset(cls, path):
        """Load a benchmark bundle or a legacy standalone Body joint asset."""
        asset = torch.load(path, map_location="cpu", weights_only=True)
        if asset.get("schema") == "gem_benchmark_projectors_v1":
            joint_set = {"emdb": "smpl24", "3dpw": "smpl44"}[asset["benchmark"]]
            return cls(
                asset["joint_regressors"][joint_set],
                asset["source_num_vertices"],
                asset["face_vertex_indices"],
            )
        return cls(**asset)

    def forward(self, vertices):
        if vertices.shape[-2:] != (self.source_num_vertices, 3):
            raise ValueError("Body projector source vertex count differs from its asset")
        with torch.autocast(device_type=vertices.device.type, enabled=False):
            vertices = vertices.float()
            if len(self.face_vertex_indices):
                v0 = vertices[..., self.face_vertex_indices[:, 0], :]
                v1 = vertices[..., self.face_vertex_indices[:, 1], :]
                v2 = vertices[..., self.face_vertex_indices[:, 2], :]
                fourth = v0 + torch.linalg.cross(v1 - v0, v2 - v0, dim=-1)
                vertices = torch.cat((vertices, fourth), dim=-2)
            with _full_precision_matmul(vertices.device):
                return torch.einsum("jv,...vc->...jc", self.matrix, vertices)


class BodyPoseEvaluator(nn.Module):
    """Evaluate a Body ``PosePrediction`` or prepared ``{'joints': B×J×3}``.

    ``batch['keypoints_3d']`` supplies complete matching joints in meters with
    shape B×24×(3|4) for EMDB or B×44×(3|4) for 3DPW. A fourth confidence
    channel does not weight these 3D metrics. EMDB centers all 24 joints on
    the mean of hips 1/2; 3DPW centers on joint 39 and scores 25..37 and 43.

    Metrics are one scalar per sample: ``mpjpe``, ``pa_mpjpe`` and
    ``scale_mpjpe``, all in millimeters. PA fits a proper rotation, scale and
    translation; scale-only alignment fits a nonnegative scale and translation.
    Samples with nonfinite selected GT joints return NaN and are excluded by
    the shared evaluation runner. Joint conversion removes camera translation
    before projecting the mesh.
    """

    def __init__(self, protocol, joint_projector=None):
        super().__init__()
        if protocol not in _PROTOCOLS:
            raise ValueError(f"Unsupported Body protocol: {protocol!r}")
        self.protocol = protocol
        self.joint_projector = joint_projector
        self.joint_count, self.indices, self.pelvis = _PROTOCOLS[protocol]

    @torch.no_grad()
    def evaluate(self, prediction, batch):
        if isinstance(prediction, Mapping):
            predicted = prediction["joints"].detach().float()
        else:
            if prediction.task != "body":
                raise ValueError("BodyPoseEvaluator requires a body prediction")
            if self.joint_projector is None:
                raise ValueError("Body Pose evaluation requires a joint projector")
            vertices = (
                prediction.vertices_camera.float() - prediction.camera_translation.float()[:, None]
            )
            self.joint_projector.to(vertices.device)
            predicted = self.joint_projector(vertices)
        if predicted.ndim != 3 or predicted.shape[1:] != (self.joint_count, 3):
            raise ValueError(
                f"{self.protocol} prediction requires full {self.joint_count}-joint order"
            )
        target = batch["keypoints_3d"].to(predicted).detach()
        if (
            target.ndim != 3
            or target.shape[:2] != predicted.shape[:2]
            or target.shape[-1] not in (3, 4)
        ):
            raise ValueError(
                "Body GT must match prediction joint order, with XYZ or XYZ-confidence"
            )
        with (
            torch.autocast(device_type=predicted.device.type, enabled=False),
            _full_precision_matmul(predicted.device),
        ):
            indices, pelvis = list(self.indices), list(self.pelvis)
            pred = predicted[:, indices] - predicted[:, pelvis].mean(1, keepdim=True)
            truth = target[..., :3]
            truth = truth[:, indices] - truth[:, pelvis].mean(1, keepdim=True)
            valid = torch.isfinite(truth).all((1, 2))
            if bool((valid & ~torch.isfinite(pred).all((1, 2))).any()):
                raise ValueError("Nonfinite Body prediction on a finite benchmark GT row")
            result = {
                name: predicted.new_full((len(predicted),), float("nan"))
                for name in ("mpjpe", "pa_mpjpe", "scale_mpjpe")
            }
            if bool(valid.any()):
                p, g = pred[valid], truth[valid]
                result["mpjpe"][valid] = mean_joint_error(p, g) * 1000
                result["pa_mpjpe"][valid] = mean_joint_error(similarity_align(p, g), g) * 1000
                result["scale_mpjpe"][valid] = mean_joint_error(scale_align(p, g), g) * 1000
        return result
