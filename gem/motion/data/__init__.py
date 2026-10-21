# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached conditions and optional training data loaders for GEM-Motion."""

from .epoch_budget import BudgetedVideoDataset

__all__ = ["BudgetedVideoDataset", "MixedImageVideoDataModule", "MixedTaskSchedule"]


def __getattr__(name):
    if name in {"MixedImageVideoDataModule", "MixedTaskSchedule"}:
        from .mixed import MixedImageVideoDataModule, MixedTaskSchedule

        return {
            "MixedImageVideoDataModule": MixedImageVideoDataModule,
            "MixedTaskSchedule": MixedTaskSchedule,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
