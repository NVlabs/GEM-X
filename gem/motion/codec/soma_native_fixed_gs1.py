# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from gem.common.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gem.common.geometry.trajectory import get_local_transl_vel

from .layout import SOMA_NATIVE_FIXED_GS1_V1_LAYOUT
from .stats import MotionStats

MOTION_STEP_KEY = "motion_reference_step_scale"


def reference_displacement(interval_values, step_factors):
    return interval_values / step_factors[..., None]


def root_motion_rows(inputs):
    return inputs.get("mask", {}).get("spv_root_motion")


def override_root_targets(inputs, camera_rotation, gravity_rotation, local_displacement):
    """Keep legacy rows unchanged; Selected rows use one coherent root-label source.

    Returns rotations and per-output-interval displacement before the caller's
    existing reference30 conversion. The original camera pose/shape parameters
    remain available separately for observation and sparse geometry losses.
    """
    rows = root_motion_rows(inputs)
    if rows is None:
        return camera_rotation, gravity_rotation, local_displacement
    target = inputs["motion_targets"]
    rotation_c = axis_angle_to_matrix(target["global_orient_c"])
    rotation_gv = axis_angle_to_matrix(target["global_orient_gv"])
    if local_displacement.shape[1] == 1:
        target_displacement = torch.zeros_like(target["transl_w"])
    else:
        target_displacement = get_local_transl_vel(target["transl_w"], target["global_orient_w"])
    target_displacement = target_displacement.masked_fill(
        ~inputs["mask"]["root_interval"][..., None], 0
    )
    return (
        torch.where(rows[:, None, None, None], rotation_c, camera_rotation),
        torch.where(rows[:, None, None, None], rotation_gv, gravity_rotation),
        torch.where(rows[:, None, None], target_displacement, local_displacement),
    )


class SomaNativeFixedGs1EnDecoder(nn.Module):
    """GEM motion codec for full-pose SOMA-native fixed-GS1 parameters.

    The denoised vector contains all 76 non-root rotations, identity128,
    learned local-scale56, camera/global root rotations, and local root
    velocity.  Global scale is deliberately absent from the learned vector;
    decode reconstructs the public ``scale_params=[1, local56...]`` contract.
    """

    parameter_kind = "soma"
    params_key_c = "soma_params_c"
    params_key_w = "soma_params_w"
    # Applied only when the caller requests postproc (test/demo, not fit validation).
    supports_postprocess = True
    # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist] in public SOMA77.
    contact_joint_ids = (69, 70, 74, 75, 14, 42)

    def __init__(
        self,
        stats_path: str | Path | None = None,
        std_floor_by_field: Mapping[str, float] | None = None,
        *,
        stats_mean: torch.Tensor | None = None,
        stats_std: torch.Tensor | None = None,
        body_model: nn.Module | None = None,
    ):
        super().__init__()
        self.layout = SOMA_NATIVE_FIXED_GS1_V1_LAYOUT
        if std_floor_by_field is None:
            std_floor_by_field = {"*": 0.01}
        if stats_path is not None:
            if stats_mean is not None or stats_std is not None:
                raise ValueError("stats_path and in-memory statistics are mutually exclusive")
            stats = MotionStats.load_npz(stats_path, self.layout, std_floor_by_field)
        else:
            if stats_mean is None or stats_std is None:
                raise ValueError(
                    "SOMA motion statistics are not bundled; pass stats_path or both stats_mean/stats_std"
                )
            stats = MotionStats.build(self.layout, stats_mean, stats_std, std_floor_by_field)
        self.register_buffer("mean", stats.mean.clone(), persistent=True)
        self.register_buffer("raw_std", stats.raw_std.clone(), persistent=True)
        self.register_buffer("std", stats.std.clone(), persistent=True)
        self.register_buffer("std_was_clamped", stats.clamped_mask.clone(), persistent=True)

        self.body_model = body_model

    def get_input_params(self, inputs: dict, coordinate: str) -> dict[str, torch.Tensor]:
        key = self.params_key_c if coordinate == "c" else self.params_key_w
        return inputs[key]

    @staticmethod
    def _sequence_feature(
        value: torch.Tensor,
        *,
        batch_size: int,
        length: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        if value.shape == (batch_size, width):
            return value[:, None].expand(-1, length, -1)
        if value.shape == (batch_size, length, width):
            return value
        raise ValueError(
            f"{name} must have shape {(batch_size, width)} or "
            f"{(batch_size, length, width)}, got {tuple(value.shape)}"
        )

    def encode(self, inputs: dict) -> torch.Tensor:
        motion = self.encode_raw(inputs)
        return (motion - self.mean.to(motion)) / self.std.to(motion)

    def encode_raw(self, inputs: dict) -> torch.Tensor:
        params_c = self.get_input_params(inputs, "c")
        body_pose = params_c["body_pose"]
        batch_size, length = body_pose.shape[:2]
        body_pose = body_pose.reshape(batch_size, length, 76, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)

        identity = self._sequence_feature(
            params_c["identity_coeffs"],
            batch_size=batch_size,
            length=length,
            width=128,
            name="identity_coeffs",
        )
        scale57 = self._sequence_feature(
            params_c["scale_params"],
            batch_size=batch_size,
            length=length,
            width=57,
            name="scale_params",
        )
        if not torch.allclose(scale57[..., 0], torch.ones_like(scale57[..., 0])):
            raise ValueError("This codec requires fixed global scale 1")
        local_scale = scale57[..., 1:]
        if (local_scale <= 0).any():
            raise ValueError("Local scales must be positive before log encoding")
        log_local_scale = torch.log(local_scale)

        global_orient_c_R = axis_angle_to_matrix(params_c["global_orient"])
        global_orient_gv_R = inputs["R_c2gv"] @ global_orient_c_R

        params_w = self.get_input_params(inputs, "w")
        if MOTION_STEP_KEY in inputs and length == 1:
            local_transl_vel = torch.zeros_like(params_w["transl"])
        else:
            local_transl_vel = get_local_transl_vel(params_w["transl"], params_w["global_orient"])
        global_orient_c_R, global_orient_gv_R, local_transl_vel = override_root_targets(
            inputs, global_orient_c_R, global_orient_gv_R, local_transl_vel
        )
        global_orient_c_r6d = matrix_to_rotation_6d(global_orient_c_R)
        global_orient_gv_r6d = matrix_to_rotation_6d(global_orient_gv_R)
        if MOTION_STEP_KEY in inputs:
            local_transl_vel = reference_displacement(local_transl_vel, inputs[MOTION_STEP_KEY])
        motion = torch.cat(
            (
                body_pose_r6d,
                identity,
                log_local_scale,
                global_orient_c_r6d,
                global_orient_gv_r6d,
                local_transl_vel,
            ),
            dim=-1,
        )
        return motion

    def decode(self, motion_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, length, width = motion_norm.shape
        if width != self.layout.output_dim:
            raise ValueError(f"expected {self.layout.output_dim}D motion, got {width}D")
        motion = motion_norm * self.std.to(motion_norm) + self.mean.to(motion_norm)
        field = self.layout.field_slice

        body_pose_r6d = motion[..., field("body_pose_r6d")].reshape(batch_size, length, 76, 6)
        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d)).flatten(-2)
        identity = motion[..., field("identity_coeffs")]
        log_local_scale = motion[..., field("log_local_scale")]
        local_scale = torch.exp(log_local_scale)
        # The deployed shape contract publishes scale57 as float32 even when
        # the normalized network output/exp(local) is half precision.
        scale_params = torch.cat(
            (torch.ones_like(local_scale[..., :1]), local_scale), dim=-1
        ).float()
        global_orient_c = matrix_to_axis_angle(
            rotation_6d_to_matrix(motion[..., field("global_orient_c_r6d")])
        )
        global_orient_gv = matrix_to_axis_angle(
            rotation_6d_to_matrix(motion[..., field("global_orient_gv_r6d")])
        )
        return {
            "body_pose": body_pose,
            "identity_coeffs": identity,
            "log_local_scale": log_local_scale,
            "local_scale": local_scale,
            "scale_params": scale_params,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": motion[..., field("local_transl_vel")],
        }

    @staticmethod
    def build_incam_params(
        decode_dict: dict[str, torch.Tensor], transl: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "body_pose": decode_dict["body_pose"],
            "identity_coeffs": decode_dict["identity_coeffs"],
            "scale_params": decode_dict["scale_params"],
            "global_orient": decode_dict["global_orient"],
            "transl": transl,
        }

    @staticmethod
    def build_global_params(
        decode_dict: dict[str, torch.Tensor], root_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "body_pose": decode_dict["body_pose"],
            "identity_coeffs": decode_dict["identity_coeffs"],
            "scale_params": decode_dict["scale_params"],
            **root_params,
        }

    def forward_body(self, **params: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if self.body_model is None:
            raise RuntimeError("FK requires a configured SOMA body model")
        return self.body_model(**params)

    @staticmethod
    def surface_root(body_output: dict[str, torch.Tensor | None]) -> torch.Tensor:
        joints = body_output["joints"]
        assert joints is not None
        return joints[..., :1, :]

    def fk_v2(self, **params: torch.Tensor) -> torch.Tensor:
        joints = self.forward_body(**params, return_joints_only=True)["joints"]
        assert joints is not None
        return joints
