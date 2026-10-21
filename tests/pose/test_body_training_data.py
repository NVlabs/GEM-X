# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-sample contracts and recovery from actual consumption, not prefetch."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gem.pose.data.training import PoseTrainingDataModule, PreparedPoseDataset


@pytest.fixture
def samples():
    path = Path(__file__).resolve().parents[2] / "scripts/make_pose_example.py"
    spec = importlib.util.spec_from_file_location("make_pose_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_samples(8)


def indices(batch):
    return batch["data_cursor"]["sample_index"].tolist()


def drain(module):
    batches = []
    for batch in module.train_dataloader():
        batches.append(indices(batch))
        module.mark_batch_consumed(batch)
    return batches


def test_prepared_file_retains_confidence_and_isolates_rows(tmp_path, samples):
    samples[0]["landmarks_2d"][0] = torch.tensor([0.2, -0.1, 1.7])
    samples[0]["keypoints_2d"][1] = torch.tensor([float("nan"), 0, 0])
    path = tmp_path / "samples.pt"
    torch.save({"samples": samples, "synthetic": True}, path)
    dataset = PreparedPoseDataset(path=path)
    row = dataset[0]
    assert row["landmarks_2d"][0, 2] == pytest.approx(1.7)
    assert torch.isnan(row["keypoints_2d"][1, 0])
    row["img"].zero_()
    row["soma_identity_coeffs"][0] = 8
    assert dataset[0]["img"].abs().sum() > 0
    assert dataset[0]["soma_identity_coeffs"][0] == 0
    assert len(dataset) == 8


@pytest.mark.parametrize(
    "kind", ["packed_scale", "nonfinite_masked_gt", "uint8_rgb", "wrong_global_scale"]
)
def test_prepared_boundary_rejects_wrong_native_training_contract(samples, kind):
    row = copy.deepcopy(samples[0])
    if kind == "packed_scale":
        row["soma_scale_params"] = torch.ones(57)
    elif kind == "nonfinite_masked_gt":
        row["has_soma_params"] = torch.tensor(0.0)
        row["soma_body_pose"][0, 0] = float("nan")
    elif kind == "uint8_rgb":
        row["img"] = torch.zeros(3, 256, 256, dtype=torch.uint8)
    else:
        row["soma_global_scale"] = torch.tensor(1.2)
    with pytest.raises(ValueError):
        PreparedPoseDataset([row])[0]


def test_resume_uses_consumed_cursor_despite_prefetch(samples):
    dataset = PreparedPoseDataset(samples)
    module = PoseTrainingDataModule(dataset, batch_size=2, num_workers=2, seed=9)
    loader = module.train_dataloader()
    iterator = iter(loader)
    first = next(iterator)
    module.mark_batch_consumed(first)
    prefetched = next(iterator)  # Deliberately not consumed by the model.
    state = module.state_dict()
    assert state["epoch"] == 0 and state["consumed_batches"] == 1
    # Advancing the original loader, including worker prefetch, cannot commit.
    rest = [prefetched, *list(iterator)]
    assert module.state_dict() == state
    resumed = PoseTrainingDataModule(dataset, batch_size=2, num_workers=0, seed=9)
    resumed.load_state_dict(state)
    resumed_loader = resumed.train_dataloader()
    assert len(resumed_loader) == 4  # Full epoch size for Lightning loop recovery.
    actual = []
    for batch in resumed_loader:
        actual.append(indices(batch))
        resumed.mark_batch_consumed(batch)
    assert actual == [indices(batch) for batch in rest]
    assert resumed.data_epoch == 1 and resumed.consumed_batches == 0
    assert sorted(indices(first) + sum(actual, [])) == list(range(8))


def test_epoch_shuffle_repeats_after_restore_and_rejects_double_commit(samples):
    data = PreparedPoseDataset(samples)
    module = PoseTrainingDataModule(data, batch_size=2, seed=31)
    first = drain(module)
    checkpoint = module.state_dict()
    second = drain(module)
    restored = PoseTrainingDataModule(data, batch_size=2, seed=31)
    restored.load_state_dict(checkpoint)
    assert drain(restored) == second
    assert first != second
    batch = next(iter(restored.train_dataloader()))
    restored.mark_batch_consumed(batch)
    with pytest.raises(ValueError, match="exactly once"):
        restored.mark_batch_consumed(batch)


def test_distributed_partition_and_resume_world_size(samples):
    data = PreparedPoseDataset(samples)
    ranks = []
    for rank in (0, 1):
        module = PoseTrainingDataModule(data, batch_size=2, seed=11, drop_last=True)
        module.trainer = SimpleNamespace(global_rank=rank, world_size=2)
        ranks.append(sum(drain(module), []))
    assert set(ranks[0]).isdisjoint(ranks[1])
    assert sorted(ranks[0] + ranks[1]) == list(range(8))
    restored = PoseTrainingDataModule(data, batch_size=2, seed=11, drop_last=True)
    restored.load_state_dict(module.state_dict())
    with pytest.raises(ValueError, match="world size"):
        restored.train_dataloader()


def test_resume_rejects_changed_data_order_configuration(samples):
    original = PoseTrainingDataModule(PreparedPoseDataset(samples), batch_size=2, seed=1)
    changed = PoseTrainingDataModule(PreparedPoseDataset(samples), batch_size=4, seed=1)
    with pytest.raises(ValueError, match="configuration"):
        changed.load_state_dict(original.state_dict())
