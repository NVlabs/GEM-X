# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Motion projection fixtures without model or conversion assets."""

import numpy as np
import pytest
import torch

from gem.motion.evaluation.geometry import SomaJointRegressor, SomaMeshProjector


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def joint_asset(tmp_path, target="smpl24", mode="tet_bary"):
    count = 24 if target == "smpl24" else 14
    matrix = torch.zeros(count, 5 if mode == "tet_bary" else 3)
    matrix[:, 1] = 0.25
    matrix[:, 2] = 0.75
    faces = torch.tensor([[0, 1, 2], [2, 1, 0]])
    if mode == "tet_bary":
        matrix[:, 0] = -0.5
        matrix[:, 3:] = torch.tensor([0.75, -0.25])
    path = tmp_path / f"{target}.pt"
    torch.save(
        {
            "schema": "gvhmr_soma_to_target_joints_asset_v1",
            "asset_version": "fixture-v1",
            "source_topology": "fixture3",
            "source_num_vertices": 3,
            "target_joint_set": target,
            "target_num_joints": count,
            "mode": mode,
            "joint_regressor": matrix,
            "face_vertex_indices": faces,
        },
        path,
    )
    return path


@pytest.mark.parametrize("joint_set", ["smpl24", "3dpw_j14"])
@pytest.mark.parametrize("mode", ["direct", "tet_bary"])
def test_joint_projection_topology_signed_weights_and_area_scaling(tmp_path, joint_set, mode):
    projector = SomaJointRegressor(joint_asset(tmp_path, joint_set, mode), joint_set)
    vertices = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0, 3.0, 0]])
    points = torch.stack((vertices, 2 * vertices))[None]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = projector(points)
    expected = torch.tensor([[0.5, 2.25, 0], [1.0, 4.5, 0]])
    if mode == "tet_bary":
        expected = torch.tensor([[0.5, 1.5, 6], [1.0, 3.0, 24]])
    torch.testing.assert_close(output, expected[None, :, None].expand_as(output))
    assert output.dtype == torch.float32
    assert projector.state_dict() == {}


def test_joint_projection_rejects_wrong_target_topology_and_matrix_shape(tmp_path):
    path = joint_asset(tmp_path)
    with pytest.raises(ValueError, match="benchmark joint set"):
        SomaJointRegressor(path, "3dpw_j14")
    with pytest.raises(ValueError, match="SOMA topology"):
        SomaJointRegressor(path, "smpl24", expected_source_topology="another_mesh")
    projector = SomaJointRegressor(path, "smpl24")
    with pytest.raises(ValueError, match="Expected SOMA vertices"):
        projector(torch.zeros(2, 4, 3))
    asset = torch.load(path, weights_only=True)
    asset["joint_regressor"] = asset["joint_regressor"][:, :-1]
    torch.save(asset, path)
    with pytest.raises(ValueError, match="dimensions"):
        SomaJointRegressor(path, "smpl24")


def mesh_assets(tmp_path):
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    face_ids = torch.arange(6890) % 2
    weights = torch.tensor([[20000.0, -19999.25, 0.5, -0.25], [-0.5, 0.25, 0.5, 0.75]])
    for name, value in {
        "face_vertex_indices": faces,
        "tet_vertex_indices": torch.cat((faces, torch.tensor([[6], [7]])), dim=1),
        "smpl_vertex_face_ids": face_ids,
        "smpl_vertex_barycentric": weights[face_ids],
    }.items():
        torch.save(value, tmp_path / f"{name}.pt")
    return faces[face_ids], weights[face_ids]


def test_mesh_projection_preserves_signed_fp64_interpolation_and_chunks(tmp_path):
    faces, weights = mesh_assets(tmp_path)
    vertices = torch.randn(2, 3, 6, 3, generator=torch.Generator().manual_seed(79)) + 1000
    projector = SomaMeshProjector(tmp_path, source_num_vertices=6, chunk_size=2)
    output = projector(vertices)
    points = vertices.numpy().astype(np.float64)
    a, b, c = (points[..., faces[:, i], :] for i in range(3))
    d = a + np.cross(b - a, c - a)
    w = weights.numpy().astype(np.float64)
    expected = a * w[:, 0:1] + b * w[:, 1:2] + c * w[:, 2:3] + d * w[:, 3:4]
    np.testing.assert_array_equal(output.numpy(), expected.astype(np.float32))
    assert output.shape == (2, 3, 6890, 3)
    assert projector.state_dict() == {}
    torch.testing.assert_close(
        output, SomaMeshProjector(tmp_path, source_num_vertices=6)(vertices), atol=0, rtol=0
    )
    a, b, c = (vertices[..., faces[:, i], :] for i in range(3))
    d = a + torch.linalg.cross(b - a, c - a, dim=-1)
    low_precision = sum(point * weights[:, i : i + 1] for i, point in enumerate((a, b, c, d)))
    assert (output - low_precision).abs().max() > 0.1


def test_mesh_projection_preserves_translation_once_and_rejects_bad_assets(tmp_path):
    mesh_assets(tmp_path)
    projector = SomaMeshProjector(tmp_path, source_num_vertices=6, chunk_size=1)
    vertices = torch.arange(36).reshape(2, 6, 3).float() / 8
    translation = torch.tensor([16.0, -8.0, 4.0])
    torch.testing.assert_close(
        projector(vertices + translation), projector(vertices) + translation, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="floating"):
        projector(vertices.long())
    path = tmp_path / "tet_vertex_indices.pt"
    tets = torch.load(path, weights_only=True)
    tets[0, 3] = 0
    torch.save(tets, path)
    with pytest.raises(ValueError, match="tetrahedra"):
        SomaMeshProjector(tmp_path, source_num_vertices=6)
