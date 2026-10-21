# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balanced source batches addressed by consumed training attempts.

The sampler's prefetch cursor is never checkpoint state. Source selection,
sample permutations and CPU data augmentation are addressed by the attempt and
draw index. Workers can therefore prefetch freely or be recreated on resume.
Use ``Trainer(use_distributed_sampler=False)``; rank sharding is handled here.
"""

import random
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, default_collate

from .collate import collate_evaluation_rows
from .epoch_budget import BudgetedVideoDataset


@contextmanager
def isolated_data_rng(seed):
    """Seed a CPU dataset/collator draw without advancing training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class MixedTaskSchedule:
    """Shuffle blocks with exact configured source counts using a private RNG.

    ``{'video': 16, 'image_a': 1, 'image_b': 2, 'image_c': 1}`` gives 80%
    video and 20% image attempts, with 25/50/25 mixing among image sources.
    Blocks span epoch boundaries; all ranks share the same source schedule.
    """

    def __init__(self, source_weights: Mapping[str, int], seed: int = 42):
        if not source_weights or any(
            type(count) is not int or count <= 0 for count in source_weights.values()
        ):
            raise ValueError("Source weights must be positive integer batch counts")
        self.source_weights = dict(source_weights)
        self.sources = tuple(source_weights)
        self.seed = seed
        self._base = tuple(source for source, count in source_weights.items() for _ in range(count))
        self._cached_index = None
        self._cached_block = None

    def block(self, index):
        if index != self._cached_index:
            generator = torch.Generator().manual_seed(self.seed + index)
            order = torch.randperm(len(self._base), generator=generator).tolist()
            self._cached_block = tuple(self._base[position] for position in order)
            self._cached_index = index
        return self._cached_block

    def source_at(self, attempt):
        block, position = divmod(attempt, len(self._base))
        return self.block(block)[position]

    def source_batch_ordinal(self, attempt, source):
        block, position = divmod(attempt, len(self._base))
        return block * self.source_weights[source] + self.block(block)[:position].count(source)


@dataclass(frozen=True)
class SampleDraw:
    source: str
    index: int
    attempt: int
    seed: int


class MixedBatchSampler(Sampler):
    def __init__(
        self,
        source_sizes,
        batch_sizes,
        schedule,
        attempts_per_epoch,
        rank=0,
        world_size=1,
        video_source="video",
        video_epoch_sampler=None,
    ):
        self.source_sizes = dict(source_sizes)
        self.batch_sizes = dict(batch_sizes)
        self.schedule = schedule
        self.attempts_per_epoch = attempts_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.video_source = video_source
        self.video_epoch_sampler = video_epoch_sampler
        self.consumed_attempts = 0
        # Historical image permutations enumerate image sources independently
        # of the separate video loader. Preserve those one-based seed offsets.
        image_sources = [source for source in source_sizes if source != video_source]
        self.source_seed_offsets = {source: index + 1 for index, source in enumerate(image_sources)}
        self.source_seed_offsets[video_source] = 0
        if set(source_sizes) != set(schedule.sources) or set(batch_sizes) != set(schedule.sources):
            raise ValueError("Sources, batch sizes and schedule must name the same datasets")
        if attempts_per_epoch < 1 or not 0 <= rank < world_size:
            raise ValueError("Require positive epoch length and a valid distributed rank")
        for source, count in source_sizes.items():
            if batch_sizes[source] < 1 or count < batch_sizes[source] * world_size:
                raise ValueError(f"Source {source!r} must contain a complete global batch")

    def __len__(self):
        # Lightning restores the epoch's completed-batch counter separately.
        # Stateful loaders report the full epoch length, even during resume.
        return self.attempts_per_epoch

    def __iter__(self):
        start_attempt = self.consumed_attempts
        epoch = start_attempt // self.attempts_per_epoch
        epoch_start = epoch * self.attempts_per_epoch
        end_attempt = epoch_start + self.attempts_per_epoch
        video_indices = None
        if self.video_epoch_sampler is not None:
            self.video_epoch_sampler.set_epoch(epoch)
            video_indices = list(self.video_epoch_sampler)
            video_ordinal_at_epoch_start = self.schedule.source_batch_ordinal(
                epoch_start, self.video_source
            )
            video_batches = (
                self.schedule.source_batch_ordinal(end_attempt, self.video_source)
                - video_ordinal_at_epoch_start
            )
            if video_batches * self.batch_sizes[self.video_source] > len(video_indices):
                raise ValueError(
                    "Video epoch budget cannot provide all scheduled full video batches"
                )
        permutations = {}
        for attempt in range(start_attempt, end_attempt):
            source = self.schedule.source_at(attempt)
            source_seed_offset = self.source_seed_offsets[source]
            ordinal = self.schedule.source_batch_ordinal(attempt, source)
            batch_size = self.batch_sizes[source]
            global_batch = batch_size * self.world_size
            if source == self.video_source and video_indices is not None:
                # The original wrapper restarts its video DataLoader each mixed
                # epoch and skips only videos already consumed in that epoch.
                start = (ordinal - video_ordinal_at_epoch_start) * batch_size
                indices = video_indices[start : start + batch_size]
            else:
                cycle, offset = divmod(ordinal, self.source_sizes[source] // global_batch)
                if source not in permutations or permutations[source][0] != cycle:
                    seed = self.schedule.seed + 1_000_003 * source_seed_offset + cycle
                    generator = torch.Generator().manual_seed(seed)
                    permutations[source] = (
                        cycle,
                        torch.randperm(self.source_sizes[source], generator=generator),
                    )
                start = offset * global_batch + self.rank * batch_size
                indices = permutations[source][1][start : start + batch_size].tolist()
            draw_start = ordinal * global_batch + self.rank * batch_size
            yield [
                SampleDraw(
                    source,
                    index,
                    attempt,
                    self.schedule.seed + 1_000_003 * source_seed_offset + draw_start + position,
                )
                for position, index in enumerate(indices)
            ]


class _MixedDataset(Dataset):
    def __init__(self, sources):
        self.sources = sources

    def __len__(self):
        return sum(len(source) for source in self.sources.values())

    def __getitem__(self, draw):
        with isolated_data_rng(draw.seed):
            value = self.sources[draw.source][draw.index]
        return draw, value


class _MixedCollator:
    def __init__(self, collate_fns, video_source):
        self.collate_fns = collate_fns
        self.video_source = video_source

    def __call__(self, rows):
        draw = rows[0][0]
        collate = self.collate_fns.get(draw.source, default_collate)
        with isolated_data_rng(draw.seed + 3_000_017):
            batch = collate([value for _, value in rows])
        batch = dict(batch)
        batch["training_branch"] = "video" if draw.source == self.video_source else "image"
        batch["training_attempt"] = draw.attempt
        batch["training_source"] = draw.source
        if draw.source != self.video_source:
            batch["image_source"] = draw.source
        return batch


class _StatefulMixedLoader(DataLoader):
    def state_dict(self):
        return {"consumed_attempts": self.batch_sampler.consumed_attempts}

    def load_state_dict(self, state):
        self.batch_sampler.consumed_attempts = state["consumed_attempts"]


class MixedImageVideoDataModule(pl.LightningDataModule):
    """One worker pool with homogeneous per-source batches and exact resume.

    Datasets and optional per-source collators return dictionaries. Call
    ``mark_consumed(total_attempts)`` after a training batch computes its loss,
    before batch-end checkpoint callbacks run. The count includes attempts
    whose optimizer update AMP skips. Dataset/collator CPU randomness is
    isolated per draw; private dataset-owned RNGs must use that draw's seeds
    or provide their own resumable state.

    Use ``BudgetedVideoDataset`` for the video source to retain the original
    per-source epoch draw and reset. Plain video datasets retain the generic
    continuous permutation stream.
    """

    def __init__(
        self,
        sources,
        batch_sizes,
        schedule,
        *,
        attempts_per_epoch=377,
        video_source="video",
        collate_fns=None,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
        rank=None,
        world_size=None,
        validation_sources=None,
        validation_num_workers=0,
    ):
        super().__init__()
        if video_source not in sources:
            raise ValueError("video_source must name a configured dataset")
        self.sources = dict(sources)
        self.batch_sizes = dict(batch_sizes)
        self.schedule = schedule
        self.attempts_per_epoch = attempts_per_epoch
        self.video_source = video_source
        self.collate_fns = dict(collate_fns or {})
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.rank = rank
        self.world_size = world_size
        self.validation_sources = dict(validation_sources or {})
        self.validation_num_workers = validation_num_workers
        self.consumed_attempts = 0
        self._loader = None

    def mark_consumed(self, consumed_attempts):
        self.consumed_attempts = int(consumed_attempts)
        if self._loader is not None:
            self._loader.batch_sampler.consumed_attempts = self.consumed_attempts

    def state_dict(self):
        return {"consumed_attempts": self.consumed_attempts}

    def load_state_dict(self, state_dict):
        self.mark_consumed(state_dict["consumed_attempts"])

    def train_dataloader(self):
        if self._loader is None:
            trainer = self.trainer
            rank = self.rank if self.rank is not None else (trainer.global_rank if trainer else 0)
            world_size = (
                self.world_size
                if self.world_size is not None
                else (trainer.world_size if trainer else 1)
            )
            video = self.sources[self.video_source]
            video_epoch_sampler = (
                video.make_sampler(rank=rank, world_size=world_size)
                if isinstance(video, BudgetedVideoDataset)
                else None
            )
            sampler = MixedBatchSampler(
                {name: len(dataset) for name, dataset in self.sources.items()},
                self.batch_sizes,
                self.schedule,
                self.attempts_per_epoch,
                rank,
                world_size,
                video_source=self.video_source,
                video_epoch_sampler=video_epoch_sampler,
            )
            sampler.consumed_attempts = self.consumed_attempts
            worker_options = {"prefetch_factor": self.prefetch_factor} if self.num_workers else {}
            self._loader = _StatefulMixedLoader(
                _MixedDataset(self.sources),
                batch_sampler=sampler,
                collate_fn=_MixedCollator(self.collate_fns, self.video_source),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                # Iterator construction consumes this private generator, never
                # the model's RNG. Augmentations use independent draw seeds.
                generator=torch.Generator().manual_seed(self.schedule.seed + rank),
                **worker_options,
            )
        return self._loader

    def val_dataloader(self):
        trainer = self.trainer
        rank = self.rank if self.rank is not None else (trainer.global_rank if trainer else 0)
        world_size = (
            self.world_size
            if self.world_size is not None
            else (trainer.world_size if trainer else 1)
        )
        # Equal rank lengths avoid mismatched distributed forward calls.
        # Metrics deduplicate the standard sampler's padded sequence IDs.
        return [
            DataLoader(
                dataset,
                batch_size=1,
                sampler=DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, shuffle=False
                ),
                num_workers=self.validation_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_evaluation_rows,
                generator=torch.Generator().manual_seed(self.schedule.seed + 2_000_003 + rank),
            )
            for dataset in self.validation_sources.values()
        ]
