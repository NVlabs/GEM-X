# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .layout import MotionLayout


@dataclass(frozen=True)
class MotionStats:
    """Mean/std metadata for a concrete motion layout.

    ``raw_std`` records the measured dataset statistic. ``std`` is the value
    used for normalization after applying explicitly configured per-field
    floors. Keeping both makes clamped dimensions visible in pilot reports.
    """

    layout_name: str
    mean: torch.Tensor
    raw_std: torch.Tensor
    std: torch.Tensor

    @classmethod
    def build(
        cls,
        layout: MotionLayout,
        mean: torch.Tensor,
        raw_std: torch.Tensor,
        std_floor_by_field: Mapping[str, float] | None = None,
    ) -> MotionStats:
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        raw_std = torch.as_tensor(raw_std, dtype=torch.float32).flatten()
        expected = layout.output_dim
        if mean.numel() != expected or raw_std.numel() != expected:
            raise ValueError(
                f"{layout.name} expects {expected} statistics, got "
                f"mean={mean.numel()} and std={raw_std.numel()}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(raw_std).all():
            raise ValueError("motion statistics must be finite")
        if (raw_std < 0).any():
            raise ValueError("raw standard deviations must be non-negative")

        std = raw_std.clone()
        for field_name, floor in (std_floor_by_field or {}).items():
            if floor <= 0:
                raise ValueError(f"std floor for {field_name} must be positive")
            field_slice = (
                slice(0, layout.output_dim) if field_name == "*" else layout.field_slice(field_name)
            )
            std[field_slice].clamp_min_(float(floor))
        if (std <= 0).any():
            zero_fields = [
                name for name, field_slice in layout.fields.items() if (std[field_slice] <= 0).any()
            ]
            raise ValueError(
                "zero-variance motion dimensions require an explicit std floor; "
                f"affected fields={zero_fields}"
            )
        return cls(layout.name, mean, raw_std, std)

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        layout: MotionLayout,
        std_floor_by_field: Mapping[str, float] | None = None,
    ) -> MotionStats:
        payload = np.load(Path(path), allow_pickle=False)
        stored_layout = str(payload["layout_name"].item())
        if stored_layout != layout.name:
            raise ValueError(f"stats layout is {stored_layout}, expected {layout.name}")
        return cls.build(
            layout,
            torch.from_numpy(payload["mean"]),
            torch.from_numpy(payload["std"]),
            std_floor_by_field,
        )

    @property
    def clamped_mask(self) -> torch.Tensor:
        return self.std > self.raw_std

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean.to(motion)) / self.std.to(motion)

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std.to(motion) + self.mean.to(motion)
