# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-named video populations with the original shared epoch draw.

Budgets count global samples before DistributedSampler padding and batch
drop_last. Mixed image/video training consumes a prefix of each epoch's video
stream, so consumed source counts need not equal the full draw's quotas.
"""

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler


class EpochBudgetDistributedSampler(DistributedSampler):
    """Draw each source without replacement, shuffle together, then rank-stride.

    One generator seeded with ``seed + epoch`` drives all source permutations
    and the final shuffle. Source ordering is therefore part of the recipe.
    Omitted rank/world arguments follow the process group lazily, allowing
    DataLoader construction before Lightning initializes distributed training.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        epoch_samples: Sequence[int],
        seed: int = 0,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if len(epoch_samples) != len(dataset.datasets):
            raise ValueError("epoch_samples must provide one quota per concatenated dataset")
        for source, count in zip(dataset.datasets, epoch_samples):
            if type(count) is not int or not 0 < count <= len(source):
                raise ValueError(
                    f"Epoch quota must be an integer in [1, {len(source)}], got {count!r}"
                )
        if (num_replicas is None) != (rank is None):
            raise ValueError("Explicit num_replicas and rank must be supplied together")
        self.epoch_samples = tuple(epoch_samples)
        self.epoch_size = sum(epoch_samples)
        self._explicit_distribution = None if rank is None else (num_replicas, rank)
        super().__init__(
            dataset,
            num_replicas=1 if num_replicas is None else num_replicas,
            rank=0 if rank is None else rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        self._refresh_distribution()

    def _refresh_distribution(self):
        if self._explicit_distribution is None:
            self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0
        else:
            self.num_replicas, self.rank = self._explicit_distribution
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        self._refresh_distribution()
        return self.num_samples

    def __iter__(self):
        self._refresh_distribution()
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = []
        offset = 0
        for source, count in zip(self.dataset.datasets, self.epoch_samples):
            draws.append(torch.randperm(len(source), generator=generator)[:count] + offset)
            offset += len(source)
        indices = torch.cat(draws)
        indices = indices[torch.randperm(self.epoch_size, generator=generator)].tolist()
        padding = self.total_size - self.epoch_size
        indices += (indices * math.ceil(padding / self.epoch_size))[:padding]
        return iter(indices[self.rank : self.total_size : self.num_replicas])


class BudgetedVideoDataset(ConcatDataset):
    """Explicit adapter for epoch-budgeted video within a mixed DataModule.

    ``sources`` preserves the original dataset configuration's insertion order;
    ``epoch_samples`` supplies caller-owned names and quotas. Example usage::

        video = BudgetedVideoDataset(video_sources, video_quotas, seed=42)
        data = MixedImageVideoDataModule(
            {"video": video, **image_sources}, batch_sizes, schedule, ...
        )

    The usual mixed DataModule keeps its per-draw augmentation seeds. This
    adapter preserves legacy dataset indices, not legacy worker RNG histories.
    """

    def __init__(
        self,
        sources: Mapping[str, Dataset],
        epoch_samples: Mapping[str, int],
        seed: int = 42,
    ):
        if set(sources) != set(epoch_samples):
            raise ValueError("Video budgets must name exactly the configured video datasets")
        self.source_names = tuple(sources)
        self.epoch_samples = tuple(epoch_samples[name] for name in sources)
        self.seed = seed
        super().__init__(list(sources.values()))

    def make_sampler(self, *, rank=None, world_size=None):
        return EpochBudgetDistributedSampler(
            self, self.epoch_samples, seed=self.seed, num_replicas=world_size, rank=rank
        )
