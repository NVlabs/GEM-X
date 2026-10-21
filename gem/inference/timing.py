# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the physical clock recorded in a saved Motion result."""

import torch


def prediction_fps(result):
    """Video export needs one sampling rate for the saved output timeline."""
    conditions = result.get("conditions", {})
    positions = result.get("positions", conditions.get("positions"))
    if positions is None:
        positions = result.get("time_positions", conditions.get("time_positions"))
    times = (
        torch.as_tensor(positions, dtype=torch.float64) / 30
        if positions is not None
        else result.get("frame_times", conditions.get("frame_times"))
    )
    if times is not None:
        times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
        if len(times) > 1:
            intervals = times.diff()
            if intervals[0] <= 0 or not torch.allclose(
                intervals, intervals[:1].expand_as(intervals), atol=1e-6, rtol=1e-5
            ):
                raise ValueError(
                    "Video rendering requires uniformly spaced, increasing frame_times"
                )
            return 1.0 / intervals.mean().item()
    rate = result.get("sampling_fps", conditions.get("sampling_fps"))
    if rate is None:
        release = result.get("release", {})
        timing = release.get("motion", release).get("time_options") or {}
        metadata = result.get("meta", conditions.get("meta")) or {}
        if isinstance(metadata, list):
            metadata = metadata[0]
        rate = timing.get("fps_by_data_name", {}).get(
            metadata.get("data_name"), timing.get("default_fps", 30.0)
        )
    return float(torch.as_tensor(rate).item())
