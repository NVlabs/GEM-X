# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gem.common.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gem.motion.frontend.time import ReferenceMotionTime, TimeRoPEPositions


def inputs():
    rotation = matrix_to_rotation_6d(axis_angle_to_matrix(torch.tensor([0.0, 0.12, 0.0])))
    return {
        "obs": torch.zeros(2, 4, 29, 3),
        "length": torch.tensor([4, 2]),
        "cam_angvel": rotation.expand(2, 4, 6).clone(),
        "sampling_fps": torch.tensor([20.0, 30.0]),
    }


def test_camera_and_velocity_use_reference_units_but_keep_original_intervals():
    batch = inputs()
    prepared = ReferenceMotionTime().prepare_pipeline(batch, train=True, postproc=False)
    torch.testing.assert_close(
        prepared["motion_reference_step_scale"],
        torch.tensor([[1.5, 1.5, 1.5, 1.5], [1.0, 1.0, 1.0, 1.0]]),
    )
    expected = axis_angle_to_matrix(torch.tensor([0.0, 0.08, 0.0]))
    torch.testing.assert_close(rotation_6d_to_matrix(prepared["cam_angvel"][0, 0]), expected)
    assert prepared["cam_angvel_source_interval"] is batch["cam_angvel"]
    repeated = ReferenceMotionTime().prepare_pipeline(prepared, train=True, postproc=False)
    torch.testing.assert_close(repeated["cam_angvel"], prepared["cam_angvel"], atol=0, rtol=0)


def test_large_absolute_frame_times_subtract_before_float32_and_padding_uses_one():
    batch = inputs()
    batch["frame_times"] = torch.tensor(
        [[1e9, 1e9 + 0.05, 1e9 + 0.1, 1e9 + 0.2], [1e9, 1e9 + 0.05, 1e9 + 0.05, 1e9 + 0.05]],
        dtype=torch.float64,
    )
    clock = ReferenceMotionTime()
    positions = clock.clock(batch)
    torch.testing.assert_close(positions[0], torch.tensor([0.0, 1.5, 3.0, 6.0]), atol=3e-6, rtol=0)
    factors = clock.step_factors(batch)
    torch.testing.assert_close(factors[0], torch.tensor([1.5, 1.5, 3.0, 3.0]), atol=4e-6, rtol=0)
    torch.testing.assert_close(factors[1], torch.tensor([1.5, 1.5, 1.0, 1.0]), atol=3e-6, rtol=0)


def test_default30_and_single_image_keep_legacy_position_and_interval_semantics():
    batch = inputs()
    del batch["sampling_fps"]
    assert TimeRoPEPositions()(batch) is None
    batch["obs"] = batch["obs"][:, :1]
    batch["cam_angvel"] = batch["cam_angvel"][:, :1]
    batch["length"] = torch.ones(2, dtype=torch.long)
    batch["sampling_fps"] = torch.tensor([20.0, 24.0])
    torch.testing.assert_close(ReferenceMotionTime().step_factors(batch), torch.ones(2, 1))
