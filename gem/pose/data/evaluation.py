# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared benchmark rows, independent of training-label requirements."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class PreparedPoseEvaluationDataset(Dataset):
    """Consume unaugmented crops/cameras and protocol-specific benchmark GT.

    Each row has a stable string ``sample_id``, normalized CHW ``img``,
    full-image ``bbx_xys`` and ``K_fullimg``. The evaluator owns GT fields.
    Right-hand crops additionally retain the original camera/bbox metadata
    required by PosePredictor. Raw readers and benchmark split selection stay
    with the producer; training FK labels are not a substitute for benchmark GT.
    """

    def __init__(self, samples=None, *, path=None):
        if (samples is None) == (path is None):
            raise ValueError("Provide samples or a prepared evaluation path")
        if path is not None:
            samples = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["samples"]
        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("Prepared evaluation needs a nonempty sample sequence")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        if not isinstance(row["sample_id"], str):
            raise TypeError("Evaluation sample_id must be a stable string")
        if row["img"].ndim != 3 or row["img"].shape[0] != 3:
            raise ValueError("Evaluation img must be normalized CHW RGB")
        if tuple(row["bbx_xys"].shape) != (3,) or tuple(row["K_fullimg"].shape) != (3, 3):
            raise ValueError("Evaluation requires square bbox[3] and full-image camera[3,3]")
        if "aug_rot" in row and bool(torch.as_tensor(row["aug_rot"]).ne(0).any()):
            raise ValueError("Benchmark inputs must not contain training crop rotation")
        return row
