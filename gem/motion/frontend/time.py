# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical-time positions and reference-frame camera/motion units.

Camera increments and motion output intervals are distinct. Preserve the source
camera intervals for world rollout; normalize network conditions to30Hz units.
"""
import math

import torch

from gem.common.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

TIME_INPUT_KEYS = ("time_positions", "frame_times", "sampling_fps")


class TimeRoPEPositions:
    """Express elapsed seconds in reference-frame units for rotary attention.

    Constant-rate clips use their sampling clock, not the original clock before
    intentional motion-speed augmentation. Explicit frame_times are in seconds.
    The default-rate path returns None to retain legacy attention exactly.
    """

    def __init__(self, reference_fps=30.0, default_fps=30.0, fps_by_data_name=None):
        self.reference_fps = float(reference_fps)
        self.default_fps = float(default_fps)
        self.fps_by_data_name = {
            key: float(value) for key, value in (fps_by_data_name or {}).items()
        }
        rates = (self.reference_fps, self.default_fps, *self.fps_by_data_name.values())
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Time RoPE sampling/reference FPS must be finite and positive")

    def __call__(self, inputs):
        batch_size, length = inputs["obs"].shape[:2]
        device = inputs["obs"].device
        if "time_positions" in inputs:
            return self._timeline(
                inputs["time_positions"], batch_size, length, device, torch.float32
            )
        if "frame_times" in inputs:
            # Subtract absolute timestamps before FP32 conversion.
            times = self._timeline(inputs["frame_times"], batch_size, length, device, torch.float64)
            return ((times - times[:, :1]) * self.reference_fps).float()
        if "sampling_fps" in inputs:
            rates = torch.as_tensor(inputs["sampling_fps"], device=device, dtype=torch.float32)
            if rates.ndim == 0:
                rates = rates.expand(batch_size)
            if rates.shape != (batch_size,) or not bool(
                (torch.isfinite(rates) & (rates > 0)).all()
            ):
                raise ValueError(
                    "sampling_fps must be positive scalar or one finite rate per sample"
                )
        else:
            metadata = inputs.get("meta")
            if metadata is None:
                values = [self.default_fps] * batch_size
            else:
                if isinstance(metadata, dict):
                    metadata = [metadata]
                if len(metadata) != batch_size:
                    raise ValueError("Time RoPE metadata must contain one record per sample")
                values = [
                    self.fps_by_data_name.get(row.get("data_name"), self.default_fps)
                    for row in metadata
                ]
            if all(rate == self.reference_fps for rate in values):
                return None
            rates = torch.tensor(values, device=device, dtype=torch.float32)
        steps = torch.arange(length, device=device, dtype=torch.float32)
        return steps[None, :] * (self.reference_fps / rates[:, None])

    @staticmethod
    def _timeline(value, batch_size, length, device, dtype):
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.shape == (length,):
            value = value.expand(batch_size, -1)
        if value.shape != (batch_size, length) or not bool(torch.isfinite(value).all()):
            raise ValueError("Time RoPE timeline must be finite and shaped (L,) or (B,L)")
        return value


def copy_time_inputs(source, destination):
    """Preserve optional timing through evaluation/flip input whitelists."""
    for key in TIME_INPUT_KEYS:
        if key in source:
            destination[key] = source[key]
    if "meta" in source:
        destination["meta"] = source["meta"]


CAMERA_INTERVAL_KEY = "cam_angvel_source_interval"

CAMERA_REFERENCE_KEY = "cam_angvel_reference_fps"

MOTION_STEP_KEY = "motion_reference_step_scale"


def reference_step_factors(positions, lengths, sequence_length, device):
    """Return30Hz-step counts per outgoing token interval, padded finitely.

    A last real token has no observed future interval. Its factor repeats the
    previous interval for the codec's terminal placeholder; length1 uses one.
    Remaining padding uses one and never divides by repeated padded timestamps.
    """
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    factors = torch.ones(len(lengths), sequence_length, device=device, dtype=torch.float32)
    if positions is None or sequence_length == 1:
        return factors
    difference = positions[:, 1:] - positions[:, :-1]
    valid = torch.arange(sequence_length - 1, device=device)[None] < (lengths - 1)[:, None]
    if bool((valid & (difference <= 0)).any()):
        raise ValueError("Motion timestamps must increase strictly between real output tokens")
    factors[:, :-1] = torch.where(valid, difference, torch.ones_like(difference))
    previous = factors.gather(1, (lengths - 2).clamp_min(0)[:, None])
    previous = torch.where((lengths > 1)[:, None], previous, torch.ones_like(previous))
    factors.scatter_(1, (lengths - 1)[:, None], previous)
    return factors


def camera_to_reference(rotation6d, step_factors):
    """Exp(Log(R_next @ R_current.T)/(reference_fps*dt)), then encode6D.

    Multiplying a6D rotation by a time scale is not a rotation conversion.
    Identity is [1,0,0,0,1,0], whereas an all-zero network null condition is
    applied later, after geometry conversion and condition normalization.
    """
    changed = step_factors != 1
    if not bool(changed.any()):
        return rotation6d
    with torch.autocast(device_type=rotation6d.device.type, enabled=False):
        rotation = rotation_6d_to_matrix(rotation6d[changed].float())
        rotation_vector = matrix_to_axis_angle(rotation) / step_factors[changed][:, None]
        converted = matrix_to_rotation_6d(axis_angle_to_matrix(rotation_vector))
    result = rotation6d.clone()
    result[changed] = converted.to(result)
    return result


def interval_displacement(reference_displacement, step_factors):
    """Convert the network's metres/reference-frame output to actual intervals."""
    if not bool((step_factors != 1).any()):
        return reference_displacement
    return reference_displacement * step_factors[..., None]


def reference_displacement(interval_values, step_factors):
    if not bool((step_factors != 1).any()):
        return interval_values
    return interval_values / step_factors[..., None]


class ReferenceMotionTime:
    def __init__(self, reference_fps=30.0, default_fps=30.0, fps_by_data_name=None):
        if float(reference_fps) != 30.0:
            raise ValueError("The opt-in motion contract currently requires reference_fps=30")
        self.reference_fps = float(reference_fps)
        self.clock = TimeRoPEPositions(reference_fps, default_fps, fps_by_data_name)

    def step_factors(self, inputs):
        batch_size, length = inputs["obs"].shape[:2]
        device = inputs["obs"].device
        lengths = inputs.get("length", torch.full((batch_size,), length, device=device))
        return reference_step_factors(self.clock(inputs), lengths, length, device)

    def prepare_camera(self, inputs):
        if CAMERA_REFERENCE_KEY in inputs:
            if inputs[CAMERA_REFERENCE_KEY] != self.reference_fps:
                raise ValueError("Camera condition reference clock differs from motion contract")
            return inputs
        raw = inputs["cam_angvel"]
        return {
            **inputs,
            "cam_angvel": camera_to_reference(raw, self.step_factors(inputs)),
            CAMERA_INTERVAL_KEY: raw,
            CAMERA_REFERENCE_KEY: self.reference_fps,
        }

    def prepare_pipeline(self, inputs, *, train, postproc):
        prepared = self.prepare_camera(inputs)
        factors = self.step_factors(prepared)
        if not train and postproc:
            self.require_supported_postprocess(prepared)
        return {**prepared, MOTION_STEP_KEY: factors}

    def require_supported_postprocess(self, inputs):
        if bool((self.step_factors(inputs) != 1).any()):
            raise ValueError(
                "30Hz-reference motion supports non30Hz integration, but non30Hz postprocessing "
                "is not supported; use postproc=False or continuous30Hz inputs"
            )
