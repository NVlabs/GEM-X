# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read small, already-collated real batches for migration comparisons.

This finite replay dataset does not replace source datasets' window sampling or
the full training population. Packs contain input batches before model-side
geometry/observation preparation, with the original collator's mask defaults.
"""

from copy import deepcopy

import torch
from torch.utils.data import Dataset

BATCH_PACK_SCHEMA = "gem_motion_real_batch_pack_v1"


class PreparedBatchPack(Dataset):
    def __init__(self, path, source):
        pack = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if pack["schema"] != BATCH_PACK_SCHEMA:
            raise ValueError("Expected a native motion real-batch comparison pack")
        self.metadata = pack["metadata"]
        self.source = source
        self.batches = pack["batches"][source]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return deepcopy(self.batches[index])


def collate_prepared_batch(rows):
    """Use sampler batch_size=1: each record already contains a full batch."""
    if len(rows) != 1:
        raise ValueError("Prepared batch packs require sampler batch_size=1")
    return rows[0]
