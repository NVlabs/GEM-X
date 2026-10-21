# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gem.motion.data.batch_pack import BATCH_PACK_SCHEMA, PreparedBatchPack, collate_prepared_batch
from gem.motion.data.mixed import MixedImageVideoDataModule, MixedTaskSchedule


def test_prepared_pack_keeps_collated_masks_metadata_and_independent_rows(tmp_path):
    batch = {
        "length": torch.tensor([3, 2]),
        "f_imgseq": torch.randn(2, 4, 1024),
        "mask": {
            "valid": torch.tensor([[True, True, True, False], [True, True, False, False]]),
            "label_valid": torch.tensor([[True, True, True, False], [True, False, False, False]]),
        },
        "meta": [{"frame_ids": [4, 5, 6]}, {"frame_ids": [19, 20]}],
    }
    path = tmp_path / "batches.pt"
    torch.save(
        {
            "schema": BATCH_PACK_SCHEMA,
            "metadata": {"runtime_version": "test"},
            "batches": {"video": [batch], "image_a": [batch]},
        },
        path,
    )
    dataset = PreparedBatchPack(path, "video")
    first = dataset[0]
    first["mask"]["valid"].zero_()
    first["meta"][0]["frame_ids"].clear()
    again = dataset[0]
    torch.testing.assert_close(again["mask"]["valid"], batch["mask"]["valid"])
    assert again["meta"] == batch["meta"]
    module = MixedImageVideoDataModule(
        {"video": dataset, "image_a": PreparedBatchPack(path, "image_a")},
        {"video": 1, "image_a": 1},
        MixedTaskSchedule({"video": 1, "image_a": 1}),
        attempts_per_epoch=2,
        collate_fns=dict.fromkeys(("video", "image_a"), collate_prepared_batch),
    )
    batches = list(module.train_dataloader())
    assert {row["training_branch"] for row in batches} == {"video", "image"}
    assert all(row["f_imgseq"].shape == (2, 4, 1024) for row in batches)
    with pytest.raises(ValueError, match="batch_size=1"):
        collate_prepared_batch([batch, batch])
