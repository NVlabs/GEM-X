# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SOMA conversion to benchmark joints and meshes without an SMPL runtime."""

from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch import nn

SMPL_NUM_VERTICES = 6890

JOINT_COUNTS = {"smpl24": 24, "3dpw_j14": 14}


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


class SomaJointRegressor(nn.Module):
    """Use a named joint regressor from a benchmark bundle or standalone asset.

    The loaded asset owns source topology and joint ordering. ``tet_bary``
    evaluates its extra face points as ``p3=f0+cross(f1-f0,f2-f0)`` before the
    precomposed matrix multiplication. Neither SMPL code nor SMPL parameters
    are needed at this prediction boundary.
    """

    def __init__(
        self,
        asset_path,
        expected_target_joint_set,
        *,
        expected_source_topology=None,
        expected_asset_version=None,
    ):
        super().__init__()
        asset = torch.load(asset_path, map_location="cpu", weights_only=True)
        self.target_joint_set = expected_target_joint_set
        self.target_num_joints = JOINT_COUNTS[expected_target_joint_set]
        if asset["schema"] == "gem_benchmark_projectors_v1":
            if expected_target_joint_set not in asset["joint_regressors"]:
                raise ValueError("Joint asset does not match the requested benchmark joint set")
            matrix = asset["joint_regressors"][expected_target_joint_set]
            self.mode = "tet_bary"
        elif asset["schema"] == "gvhmr_soma_to_target_joints_asset_v1":
            if asset["target_joint_set"] != expected_target_joint_set:
                raise ValueError("Joint asset does not match the requested benchmark joint set")
            if asset["target_num_joints"] != self.target_num_joints:
                raise ValueError("Joint regressor dimensions do not match its named topology")
            matrix = asset["joint_regressor"]
            self.mode = asset["mode"]
        else:
            raise ValueError("Unknown SOMA benchmark joint asset schema")
        if (
            expected_source_topology is not None
            and asset["source_topology"] != expected_source_topology
        ):
            raise ValueError("Joint asset does not match the expected SOMA topology")
        if expected_asset_version is not None and asset["asset_version"] != expected_asset_version:
            raise ValueError("Joint asset version does not match the configured version")
        self.source_num_vertices = asset["source_num_vertices"]
        self.source_topology = asset["source_topology"]
        self.asset_version = asset["asset_version"]
        if self.mode not in {"direct", "tet_bary"}:
            raise ValueError(f"Unknown SOMA benchmark regressor mode {self.mode!r}")
        faces = (
            asset["face_vertex_indices"]
            if self.mode == "tet_bary"
            else torch.empty(0, 3, dtype=torch.long)
        )
        width = self.source_num_vertices + (len(faces) if self.mode == "tet_bary" else 0)
        if matrix.shape != (self.target_num_joints, width):
            raise ValueError("Joint regressor dimensions do not match its named topology")
        self.register_buffer("joint_regressor", matrix.float().contiguous(), persistent=False)
        self.register_buffer("face_vertex_indices", faces.long().contiguous(), persistent=False)

    def forward(self, soma_vertices):
        if soma_vertices.shape[-2:] != (self.source_num_vertices, 3):
            raise ValueError(f"Expected SOMA vertices ending in ({self.source_num_vertices},3)")
        prefix = soma_vertices.shape[:-2]
        with torch.autocast(device_type=soma_vertices.device.type, enabled=False):
            vertices = soma_vertices.float().reshape(-1, self.source_num_vertices, 3)
            if self.mode == "tet_bary":
                f0 = vertices[:, self.face_vertex_indices[:, 0]]
                f1 = vertices[:, self.face_vertex_indices[:, 1]]
                f2 = vertices[:, self.face_vertex_indices[:, 2]]
                fourth = f0 + torch.linalg.cross(f1 - f0, f2 - f0, dim=-1)
                vertices = torch.cat((vertices, fourth), dim=1)
            with _full_precision_matmul(soma_vertices.device):
                joints = torch.einsum("jv,nvc->njc", self.joint_regressor, vertices)
        return joints.reshape(*prefix, self.target_num_joints, 3)


class SomaMeshProjector(nn.Module):
    """Reconstruct SMPL-topology vertices without fitting or nearest-neighbour search.

    Load an EMDB benchmark bundle or a directory with four mapping files.
    The fourth tetrahedron point is ``p0 + cross(p1-p0, p2-p0)``;
    the original barycentric weights, including negative weights, are applied
    unchanged. Inputs and outputs use the same metre coordinate convention as
    the current SOMA eval joint adapter. Interpolation is accumulated in FP64
    to avoid cancellation from the existing large signed weights, then returned
    in FP32. Internal frame chunks bound temporary memory; they do not change
    the output sequence or metric aggregation.
    """

    def __init__(self, assets_dir: str | Path, *, source_num_vertices: int, chunk_size: int = 120):
        super().__init__()
        if source_num_vertices <= 0 or chunk_size <= 0:
            raise ValueError("source_num_vertices and chunk_size must be positive")
        assets_dir = Path(assets_dir)
        if assets_dir.is_dir():
            tensors = {
                name: torch.load(assets_dir / f"{name}.pt", map_location="cpu", weights_only=True)
                for name in (
                    "face_vertex_indices",
                    "tet_vertex_indices",
                    "smpl_vertex_face_ids",
                    "smpl_vertex_barycentric",
                )
            }
        else:
            asset = torch.load(assets_dir, map_location="cpu", weights_only=True)
            if asset["schema"] != "gem_benchmark_projectors_v1":
                raise ValueError("Unknown SOMA benchmark mesh asset schema")
            if "mesh" not in asset:
                raise ValueError("Benchmark projector bundle does not include mesh projection")
            if asset["source_num_vertices"] != source_num_vertices:
                raise ValueError("Mesh asset does not match the declared SOMA topology")
            tensors = {"face_vertex_indices": asset["face_vertex_indices"], **asset["mesh"]}
        faces, tets = tensors["face_vertex_indices"], tensors["tet_vertex_indices"]
        face_ids, weights = tensors["smpl_vertex_face_ids"], tensors["smpl_vertex_barycentric"]
        if faces.dtype != torch.int64 or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("face_vertex_indices must be a nonempty int64 (F,3) table")
        if int(faces.min()) < 0 or int(faces.max()) >= source_num_vertices:
            raise ValueError("face_vertex_indices exceed the declared SOMA topology")
        if tets.dtype != torch.int64 or tets.shape != (len(faces), 4):
            raise ValueError("tet_vertex_indices must be int64 (F,4)")
        if not torch.equal(tets[:, :3], faces) or not torch.equal(
            tets[:, 3], source_num_vertices + torch.arange(len(faces))
        ):
            raise ValueError(
                "mesh assets must use the original [face vertices, V+face_id] tetrahedra"
            )
        if face_ids.dtype != torch.int64 or face_ids.shape != (SMPL_NUM_VERTICES,):
            raise ValueError("smpl_vertex_face_ids must be int64 (6890,)")
        if int(face_ids.min()) < 0 or int(face_ids.max()) >= len(faces):
            raise ValueError("smpl_vertex_face_ids exceed the face table")
        if weights.dtype != torch.float32 or weights.shape != (SMPL_NUM_VERTICES, 4):
            raise ValueError("smpl_vertex_barycentric must be float32 (6890,4)")
        if not torch.isfinite(weights).all():
            raise ValueError("smpl_vertex_barycentric contains non-finite weights")
        self.source_num_vertices = source_num_vertices
        self.chunk_size = chunk_size
        self.register_buffer("source_faces", tets[face_ids, :3].contiguous(), persistent=False)
        self.register_buffer("barycentric", weights.contiguous(), persistent=False)

    def forward(self, soma_vertices: torch.Tensor) -> torch.Tensor:
        if (
            soma_vertices.shape[-2:] != (self.source_num_vertices, 3)
            or not soma_vertices.is_floating_point()
        ):
            raise ValueError(f"soma_vertices must be floating (...,{self.source_num_vertices},3)")
        if soma_vertices.device != self.barycentric.device:
            raise ValueError("SOMA vertices and mesh assets must be on the same device")
        prefix = soma_vertices.shape[:-2]
        context = (
            torch.autocast(device_type=soma_vertices.device.type, enabled=False)
            if soma_vertices.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            vertices = soma_vertices.reshape(-1, self.source_num_vertices, 3)
            output = torch.empty(
                (len(vertices), SMPL_NUM_VERTICES, 3),
                dtype=torch.float32,
                device=soma_vertices.device,
            )
            weights = self.barycentric.double()[None]
            for start in range(0, len(vertices), self.chunk_size):
                chunk = vertices[start : start + self.chunk_size].double()
                p0 = chunk[:, self.source_faces[:, 0]]
                p1 = chunk[:, self.source_faces[:, 1]]
                p2 = chunk[:, self.source_faces[:, 2]]
                p3 = p0 + torch.linalg.cross(p1 - p0, p2 - p0, dim=-1)
                output[start : start + self.chunk_size] = (
                    p0 * weights[..., 0:1]
                    + p1 * weights[..., 1:2]
                    + p2 * weights[..., 2:3]
                    + p3 * weights[..., 3:4]
                )
        return output.reshape(*prefix, SMPL_NUM_VERTICES, 3)
