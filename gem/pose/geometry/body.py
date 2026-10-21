# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA body geometry for the frozen native Pose parameter convention."""

from dataclasses import replace

import torch
from torch import nn

from gem.common.geometry.rotations import matrix_to_axis_angle
from gem.common.soma import SomaFixedGs1BodyAdapter, SomaFixedGs1ShapeSpec, SomaRuntimeConfig


class BodyGeometry(nn.Module):
    """Decode native body-head parameters with the legacy Pose FK convention.

    Inputs are ``body_pose[B,76,3,3]``, ``global_orient[B,1,3,3]``,
    ``identity_coeffs[B,128]`` and ``scale_params`` in the declared layout:
    raw local56 by default, or packed ``[GS1,local56]`` for earlier releases.
    Output joints
    use public77 ordering (Hips at index0, virtual Root omitted); vertices use
    MID18056 topology. Coordinates match the original GEP wrapper in metres.
    Without ``transl``, Hips is at the origin. Camera placement is a separate
    predictor operation; no recentering or camera conversion happens here.

    The official SOMA0.3.3 layer is lazy, frozen and always runs in float32.
    Correctives and repose are enabled, with the explicit legacy_pose021
    procedural profile required by the historical Pose checkpoints.
    """

    def __init__(self, soma_assets, *, device="cpu", scale_parameterization="local56"):
        super().__init__()
        if scale_parameterization not in {"local56", "packed"}:
            raise ValueError(f"Unknown body scale parameterization {scale_parameterization!r}")
        self.scale_parameterization = scale_parameterization
        self.register_buffer("_device_anchor", torch.empty(0, device=device), persistent=False)
        self.body_model = SomaFixedGs1BodyAdapter(
            SomaRuntimeConfig(
                data_root=soma_assets,
                lod="mid",
                device=device,
                apply_correctives=True,
                enable_procedural_transforms=True,
                procedural_reference="legacy_pose021",
            ),
            # The Pose head regresses raw local ratios; its original training
            # path permits signed predictions and learns them through FK.
            # Motion's exp-decoded scales retain the positive default contract.
            shape_spec=SomaFixedGs1ShapeSpec(require_positive_scales=False),
        )

    def _apply(self, fn, recurse=True):
        # A surrounding predictor may move/cast its neural model. Geometry
        # follows the destination device while retaining the required FP32 FK.
        self._device_anchor = fn(self._device_anchor)
        self.body_model.to(device=self._device_anchor.device, dtype=torch.float32)
        self.body_model.runtime_config = replace(
            self.body_model.runtime_config, device=self._device_anchor.device
        )
        return self

    @property
    def faces(self):
        """The shared MID triangle tensor; initializes the runtime if needed."""
        return self.body_model.runtime.faces

    def forward(self, parameters, *, transl=None):
        matrices = parameters["body_pose"]
        if not self.body_model.runtime_is_initialized:
            self._device_anchor = self._device_anchor.to(matrices.device)
            self.body_model.runtime_config = replace(
                self.body_model.runtime_config, device=matrices.device
            )
        with torch.autocast(device_type=matrices.device.type, enabled=False):
            # Preserve the original matrix -> AA -> SOMA path, including its
            # gradients. Passing matrices directly has a different derivative
            # for perturbations outside the rotation-matrix manifold.
            body_pose = matrix_to_axis_angle(matrices.float())
            global_orient = matrix_to_axis_angle(parameters["global_orient"].float().squeeze(1))
            packed_scales = parameters["scale_params"].float()
            if self.scale_parameterization == "local56":
                packed_scales = torch.cat(
                    (torch.ones_like(packed_scales[:, :1]), packed_scales), dim=-1
                )
            output = self.body_model(
                body_pose=body_pose,
                global_orient=global_orient,
                identity_coeffs=parameters["identity_coeffs"].float(),
                scale_params=packed_scales,
                transl=None if transl is None else transl.float(),
                pose2rot=True,
                apply_correctives=True,
                repose_to_bind_pose=True,
            )
        return {"joints": output["joints"], "vertices": output["vertices"]}
