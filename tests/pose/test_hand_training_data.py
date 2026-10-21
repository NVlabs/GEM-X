# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared hand semantics and integration with the shared consumed-data cursor."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from gem.pose.data.hand_training import PreparedHandPoseDataset
from gem.pose.data.training import PoseTrainingDataModule


@pytest.fixture
def samples():
    path = Path(__file__).resolve().parents[2] / "scripts/make_hand_pose_example.py"
    spec = importlib.util.spec_from_file_location("make_hand_pose_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_samples(6)


def test_prepared_hand_file_preserves_canonical_labels_and_signed_controls(tmp_path, samples):
    row = samples[0]
    row["hand_scale_params"] = torch.linspace(-0.3, 0.3, 26)
    row["hand_global_scale"] = torch.tensor([1.2])
    row["is_flipped"] = True  # Already canonicalized source right hand.
    row["full_image_roll_deg"] = torch.tensor(30.0)
    row["keypoints_2d"][0, 2] = 1.7
    row["keypoints_3d"][1, 3] = 0
    row["K_fullimg"][0, 2] = 90
    row["bbx_xys"][0] = 110
    path = tmp_path / "hands.pt"
    torch.save({"samples": samples, "synthetic": True}, path)
    before_rng = torch.get_rng_state()
    dataset = PreparedHandPoseDataset(path=path)
    prepared = dataset[0]
    assert torch.equal(torch.get_rng_state(), before_rng)
    for key, value in row.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(prepared[key], value, rtol=0, atol=0)
        else:
            assert prepared[key] == value
    prepared["hand_poses"].zero_()
    prepared["img"].zero_()
    assert torch.equal(dataset[0]["hand_poses"], row["hand_poses"])
    assert torch.equal(dataset[0]["img"], row["img"])


@pytest.mark.parametrize(
    "kind",
    [
        "axis_angle",
        "packed27",
        "identity20",
        "nonfinite_masked_gt",
        "uint8",
        "crop_roll",
        "nonfinite_confidence",
        "image_size_shape",
        "image_size_nonpositive",
    ],
)
def test_prepared_hand_rejects_incompatible_training_rows(samples, kind):
    row = copy.deepcopy(samples[0])
    if kind == "axis_angle":
        row["hand_poses"] = torch.zeros(25, 3)
    elif kind == "packed27":
        row["hand_scale_params"] = torch.zeros(27)
    elif kind == "identity20":
        row["hand_identity_coeffs"] = torch.zeros(20)
    elif kind == "nonfinite_masked_gt":
        row["has_hand_params"] = torch.tensor(0.0)
        row["hand_poses"][0, 0, 0] = float("nan")
    elif kind == "uint8":
        row["img"] = torch.zeros(3, 256, 256, dtype=torch.uint8)
    elif kind == "crop_roll":
        row["aug_rot"] = torch.tensor(30.0)
    elif kind == "nonfinite_confidence":
        row["keypoints_2d"][0, 2] = float("nan")
    elif kind == "image_size_shape":
        row["img_size"] = torch.tensor([256.0])
    else:
        row["img_size"] = torch.tensor([0.0, 256.0])
    with pytest.raises(ValueError):
        PreparedHandPoseDataset([row])[0]


def test_hand_loader_resume_reuses_shared_consumption_cursor(samples):
    dataset = PreparedHandPoseDataset(samples)
    data = PoseTrainingDataModule(dataset, batch_size=2, num_workers=2, seed=17)
    iterator = iter(data.train_dataloader())
    first = next(iterator)
    assert first["hand_poses"].shape == (2, 25, 3, 3)
    assert first["hand_scale_params"].shape == (2, 26)
    data.mark_batch_consumed(first)
    state = data.state_dict()
    remainder = [batch["sample_id"] for batch in iterator]
    assert data.state_dict() == state  # Worker prefetch does not commit rows.
    resumed = PoseTrainingDataModule(dataset, batch_size=2, num_workers=0, seed=17)
    resumed.load_state_dict(state)
    actual = []
    for batch in resumed.train_dataloader():
        actual.append(batch["sample_id"])
        resumed.mark_batch_consumed(batch)
    assert actual == remainder
    assert resumed.data_epoch == 1 and resumed.consumed_batches == 0


def test_synthetic_hand_example_has_valid_matrix_targets(samples):
    assert samples[0]["sample_id"].startswith("synthetic-hand-example-")
    matrices = torch.stack([row["hand_poses"] for row in samples])
    torch.testing.assert_close(matrices @ matrices.mT, torch.eye(3).expand_as(matrices))
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(6, 25))
    assert all(bool((row["hand_scale_params"] == 0).all()) for row in samples)
