# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analytic Motion metric fixtures, full-sequence contracts and pooling."""

import numpy as np
import pytest
import torch
from torch import nn

from gem.motion.evaluation.callback import aggregate_sequence_records
from gem.motion.evaluation.evaluator import MotionEvaluator
from gem.motion.evaluation.mesh_metrics import (
    SMPL_FOOT_VERTEX_INDICES,
    compute_soma_camcoord_metrics,
    compute_soma_global_metrics,
)
from gem.motion.evaluation.metrics import (
    compute_soma_camcoord_joint_metrics,
    compute_soma_global_joint_metrics,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def moving_joints(frames=207, joints=24):
    skeleton = torch.randn(joints, 3, generator=torch.Generator().manual_seed(39)) * 0.2
    time = torch.arange(frames).float()
    root = torch.stack((time * 0.02, time.sin() * 0.01, time * 0.005), dim=-1)
    return skeleton[None] + root[:, None]


@pytest.mark.parametrize("joint_set,count,pelvis", [("smpl24", 24, 1), ("3dpw_j14", 14, 2)])
@pytest.mark.parametrize("frames", [1, 2, 3, 8])
def test_camera_short_sequences_similarity_and_exact_pelvis(joint_set, count, pelvis, frames):
    target = moving_joints(frames, count)
    rotation = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]])
    prediction = 1.7 * (target @ rotation) + torch.tensor([2.0, -1, 3])
    result = compute_soma_camcoord_joint_metrics(
        {"pred_j3d": prediction, "target_j3d": target}, target_joint_set=joint_set
    )
    assert result["pa_mpjpe"].shape == (frames,)
    assert result["pa_mpjpe"].max() < 0.002
    assert result["mpjpe"].min() > 100
    prediction = target.clone()
    prediction[:, pelvis, 0] += 0.02
    result = compute_soma_camcoord_joint_metrics(
        {"pred_j3d": prediction, "target_j3d": target}, target_joint_set=joint_set
    )
    np.testing.assert_allclose(result["mpjpe"], 10, atol=0.001)


def test_world_similarity_alignment_and_rte_fixed_scale():
    target = moving_joints()
    rotation = torch.tensor([[0.0, -1, 0], [1.0, 0, 0], [0.0, 0, 1]])
    prediction = (target @ rotation) * 2 + 3
    metric = compute_soma_global_joint_metrics(
        {"pred_j3d_glob": prediction, "target_j3d_glob": target}, target_joint_set="smpl24"
    )
    assert metric["wa2_mpjpe"].shape == metric["waa_mpjpe"].shape == metric["rte"].shape == (207,)
    assert metric["jitter"].shape == (204,)
    assert metric["wa2_mpjpe"].max() < 0.02
    assert metric["waa_mpjpe"].max() < 0.005
    assert metric["rte"].mean() > 10  # RTE cannot fit away the factor-of-two trajectory.


def test_jitter_cubic_trajectory_has_exact_30hz_reporting_scale():
    skeleton = torch.arange(72).reshape(24, 3).float()
    trajectory = torch.arange(7).float() ** 3
    joints = skeleton[None].repeat(7, 1, 1)
    joints[:, :, 0] += trajectory[:, None]
    metrics = compute_soma_global_joint_metrics(
        {"pred_j3d_glob": joints, "target_j3d_glob": joints}, target_joint_set="smpl24"
    )
    np.testing.assert_array_equal(metrics["jitter"], np.full(4, 6 * 30**3 / 10))


def test_world_window_alignment_uses_first_two_or_all_frames_and_retains_tail():
    target = moving_joints(201)
    prediction = target.clone()
    prediction[2:100, :, 0] += torch.arange(98)[:, None] * 0.002
    prediction[100:] += 4  # A new window fits this separate translation.
    result = compute_soma_global_joint_metrics(
        {"pred_j3d_glob": prediction, "target_j3d_glob": target}, target_joint_set="smpl24"
    )
    np.testing.assert_allclose(result["wa2_mpjpe"][2:100], np.arange(98) * 2, atol=0.005)
    assert result["wa2_mpjpe"][100:].max() < 0.01
    assert len(result["waa_mpjpe"]) == 201
    assert result["waa_mpjpe"][:100].mean() < result["wa2_mpjpe"][:100].mean()


def mesh_batch(frames=7, global_coordinates=False):
    target = moving_joints(frames)
    vertices = target[:, :1].expand(-1, 6890, -1).clone()
    suffix = "_glob" if global_coordinates else ""
    return {
        f"pred_j3d{suffix}": target.clone(),
        f"target_j3d{suffix}": target,
        f"pred_verts{suffix}": vertices.clone(),
        f"target_verts{suffix}": vertices,
    }


def test_camera_pve_acceleration_and_short_sequence_sample_counts():
    batch = mesh_batch()
    batch["target_j3d"] = batch["target_j3d"].round()
    batch["pred_j3d"] = batch["target_j3d"].clone()
    batch["pred_j3d"][:, 4, 0] += torch.arange(7).float() ** 2
    batch["pred_verts"][:, :, 0] += 1
    result = compute_soma_camcoord_metrics(batch, target_joint_set="smpl24")
    np.testing.assert_allclose(result["pve"], 1000, atol=0.001)
    np.testing.assert_array_equal(result["accel"], np.full(5, 75.0))
    for count in (1, 2):
        short = compute_soma_camcoord_metrics(
            {key: value[:count] for key, value in batch.items()}, target_joint_set="smpl24"
        )
        assert short["pve"].shape == (count,) and short["accel"].size == 0


def test_foot_contact_units_individual_pooling_and_no_contact():
    batch = mesh_batch(frames=6, global_coordinates=True)
    for foot, index in enumerate(SMPL_FOOT_VERTEX_INDICES):
        batch["target_verts_glob"][:, index] = 0
        batch["pred_verts_glob"][:, index] = 0
        batch["target_verts_glob"][:, index, 0] = torch.arange(6) * (0.02 if foot % 2 else 0)
        batch["pred_verts_glob"][:, index, 0] = torch.arange(6) * ((foot + 1) * 0.001)
    result = compute_soma_global_metrics(batch, target_joint_set="smpl24")
    np.testing.assert_allclose(result["fs"], np.tile([1, 3], 5), rtol=1e-6)
    summary, _ = aggregate_sequence_records(
        {"EMDB_2": {"short": {"fs": result["fs"]}, "long": {"fs": np.array([20.0])}}}
    )
    assert summary["val_metric_EMDB_2/fs"] == pytest.approx(40 / 11)
    for index in SMPL_FOOT_VERTEX_INDICES:
        batch["target_verts_glob"][:, index, 0] = torch.arange(6) * 0.02
    no_contact = compute_soma_global_metrics(batch, target_joint_set="smpl24")
    assert no_contact["fs"].size == 0


@pytest.mark.parametrize("global_coordinates", [False, True])
def test_mask_compresses_all_arrays_before_temporal_differences(global_coordinates):
    batch = mesh_batch(frames=9, global_coordinates=global_coordinates)
    suffix = "_glob" if global_coordinates else ""
    batch[f"pred_j3d{suffix}"][:, 4, 0] += torch.arange(9).float() ** 2 * 0.001
    mask = torch.tensor([True, False, True, True, False, True, True, True, False])
    compute = compute_soma_global_metrics if global_coordinates else compute_soma_camcoord_metrics
    masked = compute(batch, target_joint_set="smpl24", mask=mask)
    compressed = compute(
        {key: value[mask] for key, value in batch.items()}, target_joint_set="smpl24"
    )
    for key in masked:
        np.testing.assert_array_equal(masked[key], compressed[key])


def test_protocol_and_timeline_boundaries():
    joints = moving_joints(3)
    with pytest.raises(ValueError, match="four valid frames"):
        compute_soma_global_joint_metrics(
            {"pred_j3d_glob": joints, "target_j3d_glob": joints}, target_joint_set="smpl24"
        )
    with pytest.raises(ValueError, match="No world metric protocol"):
        compute_soma_global_joint_metrics({}, target_joint_set="3dpw_j14")
    batch = {"pred_j3d": joints, "target_j3d": joints}
    with pytest.raises(ValueError, match="bool"):
        compute_soma_camcoord_joint_metrics(batch, target_joint_set="smpl24", mask=torch.ones(3))
    with pytest.raises(ValueError, match="shape"):
        compute_soma_camcoord_joint_metrics(batch, target_joint_set="3dpw_j14")
    full = mesh_batch()
    full["pred_verts"] = full["pred_verts"][:, :10]
    with pytest.raises(ValueError, match="6890"):
        compute_soma_camcoord_metrics(full, target_joint_set="smpl24")


class SequenceBody(nn.Module):
    runtime_is_initialized = True

    def __init__(self):
        super().__init__()
        self.sequence_shapes = []

    def forward(self, *, global_orient, vertices, identity_coeffs, return_joints_only):
        self.sequence_shapes.append(tuple(identity_coeffs.shape))
        assert not return_joints_only
        return {"vertices": vertices + identity_coeffs.mean(1, keepdim=True)[:, :, None]}


@pytest.mark.parametrize("dataset,joints", [("EMDB_1", 24), ("EMDB_2", 24), ("3DPW", 14)])
def test_evaluator_preserves_full_sequence_shape_before_masking(dataset, joints):
    body = SequenceBody()
    target = moving_joints(frames=7, joints=joints)
    identity = torch.arange(7).float()[:, None].expand(-1, 3)
    params = {
        "global_orient": torch.zeros(7, 3),
        "vertices": target - 3,
        "identity_coeffs": identity,
    }
    evaluator = MotionEvaluator(
        body, {"smpl24" if joints == 24 else "3dpw_j14": nn.Identity()}, evaluate_camera_root=True
    )
    batch = {
        "B": 1,
        "meta": [{"dataset_id": dataset, "vid": "clip"}],
        "mask": torch.tensor([[True, False, True, True, False, True, True]]),
        "eval_targets": {"incam": target[None], "global": target[None]},
    }
    outputs = {"pred_soma_params_incam": params, "pred_soma_params_global": params}
    result = evaluator.evaluate(outputs, batch)
    assert body.sequence_shapes == [(1, 7, 3)] * (2 if dataset == "EMDB_2" else 1)
    assert result["dataset_id"] == dataset and result["vid"] == "clip"
    assert result["metrics"]["mpjpe"].shape == (5,)
    assert result["metrics"]["mpjpe"].max() < 0.001
    if dataset == "EMDB_2":
        assert result["metrics"]["jitter"].shape == (2,)
    if dataset == "EMDB_1":
        assert result["metrics"]["camera_root_translation_error_mm"].max() < 0.001
    with pytest.raises(ValueError, match="one complete sequence"):
        evaluator.evaluate(outputs, {**batch, "B": 2})
    with pytest.raises(ValueError, match="mask must select"):
        evaluator.evaluate(outputs, {**batch, "mask": torch.zeros(1, 7, dtype=torch.bool)})
