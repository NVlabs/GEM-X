# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite prepared Pose-body samples and consumption-based loader recovery.

Readers prepare normalized RGB and supervision before this boundary. This module
contains no dataset-specific discovery, augmentation, annotation or fitting code.
"""

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

_REQUIRED_SHAPES = {
    "img": (3, 256, 256),
    "bbx_xys": (3,),
    "K_fullimg": (3, 3),
    "soma_body_pose": (76, 3),
    "soma_global_orient": (3,),
    "soma_identity_coeffs": (128,),
    "soma_scale_params": (56,),
    "keypoints_2d": (77, 3),
    "keypoints_3d": (77, 4),
    "pred_cam_gt": (3,),
    "has_soma_params": (),
    "has_cam_gt": (),
}
_FINITE_TARGETS = (
    "bbx_xys",
    "K_fullimg",
    "soma_body_pose",
    "soma_global_orient",
    "soma_identity_coeffs",
    "soma_scale_params",
    "pred_cam_gt",
    "has_soma_params",
    "has_cam_gt",
)


def _prepared_sample(sample):
    """Validate the external prepared-row boundary, retaining confidence values."""
    if not isinstance(sample, Mapping):
        raise TypeError("A prepared Pose sample must be a mapping")
    for name, shape in _REQUIRED_SHAPES.items():
        value = sample[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Prepared {name} must be a tensor shaped {shape}")
    if not sample["img"].is_floating_point():
        raise ValueError("Prepared img must contain normalized floating-point RGB")
    for name in _FINITE_TARGETS:
        if not bool(torch.isfinite(sample[name]).all()):
            raise ValueError(f"Prepared {name} must be finite, including unavailable-label rows")
    if float(sample["bbx_xys"][2]) <= 0:
        raise ValueError("Prepared bbx_xys must have positive square size")
    if not bool((sample["soma_scale_params"] > 0).all()):
        raise ValueError("Prepared soma_scale_params must contain positive raw local56 scales")
    if "landmarks_2d" in sample and tuple(sample["landmarks_2d"].shape) != (308, 3):
        raise ValueError("Prepared landmarks_2d must have shape (308,3)")
    if "soma_global_scale" in sample and float(sample["soma_global_scale"]) != 1.0:
        raise ValueError("Prepared body training uses architectural global scale1")
    if "data_cursor" in sample:
        raise ValueError("data_cursor is reserved for the Pose training loader")
    # Samples can be cached/reused. Returning independent tensors prevents
    # consumer-side target transformations from changing another epoch's rows.
    result = copy.deepcopy(dict(sample))
    result["aug_rot"] = torch.as_tensor(result.get("aug_rot", 0.0), dtype=torch.float32)
    return result


class PreparedPoseDataset(Dataset):
    """Read finite, already prepared body samples from memory or a PyTorch file.

    ``path`` contains ``{'samples': [sample, ...]}``; all required row fields are
    tensors. Images are normalized RGB256, keypoints use crop-centered XY/conf,
    and native shape targets are identity128/local56 with fixed GS1. Optional
    fields must be consistent across rows for PyTorch's standard collation.
    No crop/flip/roll or confidence clipping occurs here.
    """

    def __init__(self, samples: Sequence[Mapping] | None = None, *, path=None):
        if (samples is None) == (path is None):
            raise ValueError("Provide exactly one of samples or path")
        self.path = None if path is None else str(Path(path).resolve())
        if path is not None:
            samples = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["samples"]
        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("Prepared Pose samples must be a nonempty finite sequence")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return _prepared_sample(self.samples[index])


class _CursorDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        epoch, batch_index, sample_index = index
        sample = dict(self.dataset[sample_index])
        sample["data_cursor"] = {
            "epoch": epoch,
            "batch_index": batch_index,
            "sample_index": sample_index,
        }
        return sample


class _ResumeBatchSampler(BatchSampler):
    """Yield the unconsumed suffix while reporting the full epoch length.

    Lightning restores its own within-epoch batch progress. Keeping full length
    prevents treating that restored progress as part of a shorter new epoch.
    """

    def __init__(self, owner, rank, world_size):
        self.owner = owner
        sampler = DistributedSampler(
            owner.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=owner.shuffle,
            seed=owner.seed,
            drop_last=owner.drop_last,
        )
        super().__init__(sampler, owner.batch_size, owner.drop_last)

    def __iter__(self):
        epoch, consumed = self.owner.data_epoch, self.owner.consumed_batches
        self.sampler.set_epoch(epoch)
        for batch_index, indices in enumerate(super().__iter__()):
            if batch_index >= consumed:
                yield [(epoch, batch_index, index) for index in indices]


class PoseTrainingDataModule(pl.LightningDataModule):
    """Finite epochs with deterministic shuffle and a committed batch cursor.

    The training callback calls ``mark_batch_consumed(batch)`` once per processed
    training batch, before checkpoint callbacks. Prefetch and iterator advancement
    never move the cursor. An optimizer/AMP-skipped batch still consumes data.

    Each batch has one ``data_cursor`` mapping with epoch/batch/sample indices.
    Checkpoint state owns that logical cursor, not worker RNG or model counters.
    Use ``Trainer(use_distributed_sampler=False)``: this loader already owns
    DistributedSampler partitioning and standard tail/drop-last semantics.
    """

    def __init__(
        self,
        train_dataset,
        *,
        batch_size=4,
        shuffle=True,
        seed=42,
        num_workers=0,
        drop_last=False,
        val_dataset=None,
    ):
        super().__init__()
        if batch_size < 1 or num_workers < 0 or len(train_dataset) < 1:
            raise ValueError(
                "Pose data requires a nonempty dataset, positive batch size and nonnegative workers"
            )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.data_epoch = 0
        self.consumed_batches = 0
        self._restored_world_size = None
        self._batch_sampler = None

    def _rank_world(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            return trainer.global_rank, trainer.world_size
        return 0, 1

    def train_dataloader(self):
        rank, world_size = self._rank_world()
        if self._restored_world_size is not None and world_size != self._restored_world_size:
            raise ValueError("Pose data resume requires the same world size")
        sampler = _ResumeBatchSampler(self, rank, world_size)
        if len(sampler) == 0:
            raise ValueError("Pose dataset has no complete batch under the configured tail policy")
        if not 0 <= self.consumed_batches < len(sampler):
            raise ValueError("Restored Pose batch cursor is outside its finite epoch")
        self._batch_sampler = sampler
        return DataLoader(
            _CursorDataset(self.train_dataset),
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.seed + self.data_epoch),
        )

    def mark_batch_consumed(self, batch):
        cursor = batch["data_cursor"]
        epoch = int(cursor["epoch"].reshape(-1)[0])
        batch_index = int(cursor["batch_index"].reshape(-1)[0])
        if (epoch, batch_index) != (self.data_epoch, self.consumed_batches):
            raise ValueError("Pose consumed cursor must advance exactly once, in batch order")
        self.consumed_batches += 1
        if self.consumed_batches == len(self._batch_sampler):
            self.data_epoch += 1
            self.consumed_batches = 0

    def _configuration(self):
        return {
            "dataset_size": len(self.train_dataset),
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }

    def state_dict(self):
        _, world_size = self._rank_world()
        return {
            "epoch": self.data_epoch,
            "consumed_batches": self.consumed_batches,
            "world_size": world_size,
            "configuration": self._configuration(),
        }

    def load_state_dict(self, state_dict):
        if state_dict["configuration"] != self._configuration():
            raise ValueError("Pose data resume configuration differs from the saved finite dataset")
        self.data_epoch = int(state_dict["epoch"])
        self.consumed_batches = int(state_dict["consumed_batches"])
        if self.data_epoch < 0 or self.consumed_batches < 0:
            raise ValueError("Pose data epoch and consumed cursor must be nonnegative")
        self._restored_world_size = int(state_dict["world_size"])

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
