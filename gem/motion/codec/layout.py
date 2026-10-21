# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class MotionLayout:
    """Named slices for one motion representation.

    The temporal network only needs to know the total output width and which
    fields are constant over a sequence.  Body-model-specific meanings stay in
    the codec that owns the layout.
    """

    name: str
    fields: Mapping[str, slice]
    sequence_static_fields: tuple[str, ...]
    global_supervision_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        fields = dict(self.fields)
        cursor = 0
        for name, field_slice in fields.items():
            if field_slice.step not in (None, 1):
                raise ValueError(f"{self.name}.{name} must use a unit-step slice")
            if (
                field_slice.start != cursor
                or field_slice.stop is None
                or field_slice.stop <= cursor
            ):
                raise ValueError(
                    f"{self.name}.{name} must be contiguous at offset {cursor}, got {field_slice}"
                )
            cursor = field_slice.stop
        unknown_static = set(self.sequence_static_fields) - set(fields)
        unknown_global = set(self.global_supervision_fields) - set(fields)
        if unknown_static or unknown_global:
            raise ValueError(
                f"{self.name} references unknown fields: "
                f"static={sorted(unknown_static)}, global={sorted(unknown_global)}"
            )
        object.__setattr__(self, "fields", MappingProxyType(fields))

    @property
    def output_dim(self) -> int:
        return max(field_slice.stop for field_slice in self.fields.values())

    @property
    def sequence_static_slices(self) -> tuple[slice, ...]:
        return tuple(self.fields[name] for name in self.sequence_static_fields)

    @property
    def global_supervision_slices(self) -> tuple[slice, ...]:
        return tuple(self.fields[name] for name in self.global_supervision_fields)

    def field_slice(self, name: str) -> slice:
        return self.fields[name]


SOMA_NATIVE_FIXED_GS1_V1_LAYOUT = MotionLayout(
    name="soma_native_fixed_gs1_v1_655",
    fields={
        "body_pose_r6d": slice(0, 456),
        "identity_coeffs": slice(456, 584),
        "log_local_scale": slice(584, 640),
        "global_orient_c_r6d": slice(640, 646),
        "global_orient_gv_r6d": slice(646, 652),
        "local_transl_vel": slice(652, 655),
    },
    sequence_static_fields=("identity_coeffs", "log_local_scale"),
    global_supervision_fields=("global_orient_gv_r6d", "local_transl_vel"),
)
