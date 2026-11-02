# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable geometry and protocol checks without model or benchmark assets."""

from dataclasses import replace

import pytest
import torch

from gem.pose.evaluation.body import BodyJointProjector, BodyPoseEvaluator
from gem.pose.heads.output import PoseHeadOutput
from gem.pose.predictor import PosePrediction


def _joints(count, batch_size=2):
    return torch.randn(batch_size, count, 3, generator=torch.Generator().manual_seed(19))


def _prediction(vertices, translation):
    batch_size = len(vertices)
    return PosePrediction(
        task="body",
        parameters={},
        camera_translation=translation,
        joints_camera=torch.zeros(batch_size, 77, 3),
        vertices_camera=vertices + translation[:, None],
        keypoints_2d=torch.zeros(batch_size, 77, 2),
        keypoints_crop=torch.zeros(batch_size, 77, 2),
        projection_valid=torch.ones(batch_size, 77, dtype=torch.bool),
        faces=torch.empty(batch_size, 0, 3, dtype=torch.long),
        right_hand=None,
        head=PoseHeadOutput({}, torch.zeros(batch_size, 3), {}, torch.zeros(batch_size, 4)),
        pose_release_id="test-body",
        intrinsics=torch.eye(3).expand(batch_size, -1, -1),
    )


def test_tetra_projection_keeps_signed_weights_and_deformed_face_points(tmp_path):
    matrix = torch.tensor([[-0.5, 0.25, 1.0, 0.25]]).expand(24, -1).clone()
    asset = tmp_path / "emdb_projector.pt"
    torch.save(
        {
            "matrix": matrix,
            "source_num_vertices": 3,
            "face_vertex_indices": torch.tensor([[0, 1, 2]]),
        },
        asset,
    )
    projector = BodyJointProjector.from_asset(asset)
    triangle = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    vertices = torch.stack((triangle, 2 * triangle))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        projected = projector(vertices)
    expected = torch.tensor([[0.5, 3.0, 1.5], [1.0, 6.0, 6.0]])
    assert projected.dtype == torch.float32
    torch.testing.assert_close(projected, expected[:, None].expand(-1, 24, -1))
    shift = torch.tensor([2.0, -1.0, 4.0])
    torch.testing.assert_close(projector(vertices + shift), projected + shift)
    assert not projector.state_dict()


@pytest.mark.parametrize(
    "benchmark,joint_set,count", [("emdb", "smpl24", 24), ("3dpw", "smpl44", 44)]
)
def test_benchmark_bundle_selects_complete_pose_joints(tmp_path, benchmark, joint_set, count):
    matrix = torch.randn(count, 4, generator=torch.Generator().manual_seed(23))
    faces = torch.tensor([[0, 1, 2]])
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {"matrix": matrix, "source_num_vertices": 3, "face_vertex_indices": faces},
        legacy_path,
    )
    bundle_path = tmp_path / f"{benchmark}_projectors.pt"
    regressors = {joint_set: matrix}
    if benchmark == "3dpw":
        regressors["3dpw_j14"] = torch.zeros(14, 4)
    torch.save(
        {
            "schema": "gem_benchmark_projectors_v1",
            "benchmark": benchmark,
            "source_num_vertices": 3,
            "source_topology": "fixture3",
            "asset_version": "gem-eval-projectors-v1",
            "face_vertex_indices": faces,
            "joint_regressors": regressors,
        },
        bundle_path,
    )
    vertices = torch.randn(2, 3, 3, 3, generator=torch.Generator().manual_seed(27))
    bundle_projector = BodyJointProjector.from_asset(bundle_path)
    legacy_projector = BodyJointProjector.from_asset(legacy_path)
    output = bundle_projector(vertices)
    assert output.shape == (2, 3, count, 3)
    torch.testing.assert_close(output, legacy_projector(vertices), atol=0, rtol=0)
    assert not bundle_projector.state_dict()


def test_direct_projector_preserves_leading_dimensions():
    matrix = torch.zeros(44, 4)
    matrix[:, 1] = 0.25
    matrix[:, 3] = 0.75
    vertices = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3)
    projected = BodyJointProjector(matrix, 4)(vertices)
    expected = (vertices[..., 1, :] + 3 * vertices[..., 3, :]) / 4
    torch.testing.assert_close(projected, expected[..., None, :].expand(2, 3, 44, 3))


@pytest.mark.parametrize(
    "matrix,vertices,faces,message",
    [
        (torch.zeros(14, 3), 3, None, "complete SMPL24 or SMPL44"),
        (torch.zeros(24, 3), 4, None, "width"),
        (torch.zeros(24, 4), 3, [[0, 1]], "shape"),
        (torch.zeros(24, 4), 3, [[0, 1, 3]], "Face indices"),
        (torch.zeros(24, 4), 3, [[-1, 1, 2]], "Face indices"),
    ],
)
def test_projector_rejects_mismatched_asset_geometry(matrix, vertices, faces, message):
    with pytest.raises(ValueError, match=message):
        BodyJointProjector(matrix, vertices, faces)


def test_projector_rejects_wrong_input_topology():
    projector = BodyJointProjector(torch.zeros(24, 4), 4)
    with pytest.raises(ValueError, match="source vertex count"):
        projector(torch.zeros(2, 3, 3))


@pytest.mark.parametrize("protocol,count,scored_joint", [("emdb", 24, 0), ("3dpw", 44, 25)])
def test_joint_selection_units_translation_and_confidence(protocol, count, scored_joint):
    truth = _joints(count)
    prediction = truth.clone()
    prediction[:, scored_joint, 0] += 0.024 if protocol == "emdb" else 0.014
    if protocol == "3dpw":
        prediction[:, 0] += 100  # This joint is outside the scored subset and pelvis.
    prediction += torch.tensor([3.0, -2.0, 5.0])
    target = torch.cat((truth, torch.zeros(len(truth), count, 1)), dim=-1)
    metrics = BodyPoseEvaluator(protocol).evaluate({"joints": prediction}, {"keypoints_3d": target})
    torch.testing.assert_close(metrics["mpjpe"], torch.ones(2), atol=0.001, rtol=0)
    assert set(metrics) == {"mpjpe", "pa_mpjpe", "scale_mpjpe"}
    assert all(value.shape == (2,) and value.dtype == torch.float32 for value in metrics.values())


@pytest.mark.parametrize(
    "protocol,count,pelvis_joint,offset", [("emdb", 24, 1, 0.02), ("3dpw", 44, 39, 0.01)]
)
def test_protocol_uses_its_exact_pelvis(protocol, count, pelvis_joint, offset):
    truth = _joints(count)
    prediction = truth.clone()
    prediction[:, pelvis_joint, 0] += offset
    metrics = BodyPoseEvaluator(protocol).evaluate({"joints": prediction}, {"keypoints_3d": truth})
    torch.testing.assert_close(metrics["mpjpe"], torch.full((2,), 10.0), atol=0.001, rtol=0)


@pytest.mark.parametrize("protocol,count", [("emdb", 24), ("3dpw", 44)])
def test_scale_and_rotation_have_distinct_metrics(protocol, count):
    truth = _joints(count)
    evaluator = BodyPoseEvaluator(protocol)
    scaled = evaluator.evaluate({"joints": truth * 1.4 + 3}, {"keypoints_3d": truth})
    assert bool((scaled["mpjpe"] > 100).all())
    torch.testing.assert_close(scaled["scale_mpjpe"], torch.zeros(2), atol=0.002, rtol=0)
    torch.testing.assert_close(scaled["pa_mpjpe"], torch.zeros(2), atol=0.002, rtol=0)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = evaluator.evaluate({"joints": 1.4 * (truth @ rotation) + 3}, {"keypoints_3d": truth})
    torch.testing.assert_close(rotated["pa_mpjpe"], torch.zeros(2), atol=0.002, rtol=0)
    assert bool((rotated["scale_mpjpe"] > 100).all())


def test_alignment_does_not_allow_reflection_or_negative_scale():
    truth = _joints(24)
    evaluator = BodyPoseEvaluator("emdb")
    reflected = evaluator.evaluate(
        {"joints": truth * torch.tensor([-1.0, 1.0, 1.0])}, {"keypoints_3d": truth}
    )
    assert bool((reflected["pa_mpjpe"] > 100).all())
    inverted = evaluator.evaluate({"joints": -truth}, {"keypoints_3d": truth})
    expected = (truth - truth.mean(1, keepdim=True)).norm(dim=-1).mean(-1) * 1000
    torch.testing.assert_close(inverted["scale_mpjpe"], expected)


@pytest.mark.parametrize("protocol,count", [("emdb", 24), ("3dpw", 44)])
def test_pose_prediction_removes_camera_translation_before_projection(protocol, count):
    vertices = _joints(count)
    # Nonuniform row sums make a forgotten translation visible after root centering.
    weights = torch.linspace(0.5, 1.5, count)
    projector = BodyJointProjector(torch.diag(weights), count)
    target = vertices * weights[None, :, None]
    prediction = _prediction(vertices, torch.tensor([[1.0, -2.0, 8.0], [-4.0, 3.0, 10.0]]))
    metrics = BodyPoseEvaluator(protocol, projector).evaluate(prediction, {"keypoints_3d": target})
    for error in metrics.values():
        torch.testing.assert_close(error, torch.zeros(2), atol=0.002, rtol=0)


def test_missing_gt_is_excluded_but_nonfinite_predictions_are_errors():
    truth = _joints(24)
    predicted = truth.clone()
    truth[1, 5, 0] = float("nan")
    metrics = BodyPoseEvaluator("emdb").evaluate({"joints": predicted}, {"keypoints_3d": truth})
    assert all(torch.isfinite(value[0]) and torch.isnan(value[1]) for value in metrics.values())
    predicted[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite Body prediction"):
        BodyPoseEvaluator("emdb").evaluate({"joints": predicted}, {"keypoints_3d": truth})


def test_evaluator_rejects_incompatible_protocol_shapes_and_prediction_types():
    truth = _joints(24)
    with pytest.raises(ValueError, match="Unsupported Body protocol"):
        BodyPoseEvaluator("unknown")
    with pytest.raises(ValueError, match="full 44-joint order"):
        BodyPoseEvaluator("3dpw").evaluate({"joints": truth}, {"keypoints_3d": truth})
    with pytest.raises(ValueError, match="GT must match"):
        BodyPoseEvaluator("emdb").evaluate({"joints": truth}, {"keypoints_3d": truth[:, :14]})
    prediction = _prediction(truth, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="requires a joint projector"):
        BodyPoseEvaluator("emdb").evaluate(prediction, {"keypoints_3d": truth})
    with pytest.raises(ValueError, match="requires a body prediction"):
        BodyPoseEvaluator("emdb").evaluate(
            replace(prediction, task="hand"), {"keypoints_3d": truth}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical policy")
def test_cuda_projection_and_metrics_restore_tf32_policy():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        truth = _joints(24).cuda()
        projector = BodyJointProjector(torch.eye(24), 24).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            projected = projector(truth)
            metrics = BodyPoseEvaluator("emdb").evaluate(
                {"joints": projected}, {"keypoints_3d": truth}
            )
        torch.testing.assert_close(projected, truth, rtol=0, atol=0)
        assert torch.backends.cuda.matmul.allow_tf32
        assert all(value.dtype == torch.float32 for value in metrics.values())
        torch.testing.assert_close(
            metrics["pa_mpjpe"], torch.zeros(2, device="cuda"), atol=0.002, rtol=0
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
