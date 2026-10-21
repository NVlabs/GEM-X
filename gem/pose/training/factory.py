# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Pose training from public model settings and caller-owned assets."""

import torch

from gem.pose.api import (
    PoseReleaseConfig,
    body_native_config,
    build_pose_model,
    hand_training_config,
    load_pose_checkpoint,
)
from gem.pose.geometry.body import BodyGeometry

from .losses import BodyPoseObjective
from .module import PoseBodyTrainingModule


def build_body_training_model(
    *,
    soma_assets,
    release=None,
    initialization=None,
    backbone_initialization=None,
    landmark_projector=None,
    optimizer=None,
    scheduler=None,
    objective=None,
):
    """Initialize weights or a new model; full training resume belongs to Trainer.

    ``landmark_projector`` can be a tensor or a file containing a tensor (or
    the established ``M_soma`` entry). Dataset preparation and fitting are not
    performed by this factory. Reference body training regresses raw local56.
    """
    release = body_native_config() if release is None else release
    if not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    if release.task != "body" or release.head.get("scale_parameterization") != "local56":
        raise ValueError("Body training requires the native local56 body release")
    camera_modes = (
        (release.conditioning or {}).get("camera_mode", "sam3db_perspective"),
        release.inference.get("camera_mode", "sam3db_perspective"),
    )
    if (
        any(mode != "sam3db_perspective" for mode in camera_modes)
        or release.inference.get("camera_representation", "legacy") != "legacy"
    ):
        raise ValueError(
            "Body training currently uses calibrated sam3db_perspective and legacy cameras"
        )
    if initialization is not None and backbone_initialization is not None:
        raise ValueError("Choose model initialization or backbone initialization")
    model = (
        build_pose_model(release)
        if initialization is None
        else load_pose_checkpoint(initialization, release)
    )
    if backbone_initialization is not None:
        model.backbone.load_pretrained(backbone_initialization)
    if landmark_projector is not None and not isinstance(landmark_projector, torch.Tensor):
        landmark_projector = torch.load(landmark_projector, map_location="cpu", weights_only=True)
        if isinstance(landmark_projector, dict):
            landmark_projector = landmark_projector["M_soma"]
    geometry = BodyGeometry(soma_assets, scale_parameterization="local56")
    loss = BodyPoseObjective(geometry, landmark_projector, **dict(objective or {}))
    return PoseBodyTrainingModule(
        model, loss, release=release, optimizer=optimizer, scheduler=scheduler
    )


def build_hand_training_model(
    *,
    soma_assets,
    release=None,
    initialization=None,
    backbone_initialization=None,
    optimizer=None,
    objective=None,
    grad_clip_val=1.0,
):
    """Train canonical-left MHR hands; source readers own image/label mirroring."""
    from gem.pose.geometry.hand import HandGeometry

    from .hand_losses import HandPoseObjective
    from .hand_module import PoseHandTrainingModule

    release = hand_training_config() if release is None else release
    if not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    if release.task != "hand":
        raise ValueError("Hand training requires a hand release")
    camera_modes = (
        (release.conditioning or {}).get("camera_mode", "sam3db_perspective"),
        release.inference.get("camera_mode", "sam3db_perspective"),
    )
    depth_mode = release.head.get("camera_depth_mode")
    kp2d = release.kp2d_translation or {}
    if (
        any(mode != "sam3db_perspective" for mode in camera_modes)
        or depth_mode != "log_lambda"
        or release.inference.get("camera_representation") != "log_lambda"
        or not kp2d.get("enabled", False)
        or kp2d.get("topology") != "separate"
        or not release.ray_pe_num_bands
    ):
        raise ValueError("Hand training requires calibrated log-lambda, separate 2D and RayPE")
    if initialization is not None and backbone_initialization is not None:
        raise ValueError("Choose model initialization or backbone initialization")
    structural_options = {
        "camera_depth_mode": depth_mode,
        "camera_log_value_bounds": tuple(release.inference["camera_log_value_bounds"]),
        "kp2d_translation": release.kp2d_translation,
    }
    objective_options = dict(objective or {})
    for name, expected in structural_options.items():
        if name in objective_options:
            requested = objective_options.pop(name)
            if name == "camera_log_value_bounds":
                requested = tuple(requested)
            if requested != expected:
                raise ValueError(f"Hand objective {name} conflicts with the model release")
    loss_options = {
        **structural_options,
        "camera_scalar_loss_type": "smooth_l1",
        "camera_scalar_loss_std": release.head["camera_scalar_std"],
        **objective_options,
    }
    model = (
        build_pose_model(release)
        if initialization is None
        else load_pose_checkpoint(initialization, release)
    )
    if backbone_initialization is not None:
        model.backbone.load_pretrained(backbone_initialization)
    loss = HandPoseObjective(HandGeometry(soma_assets), **loss_options)
    if initialization is not None:
        checkpoint = torch.load(initialization, map_location="cpu", weights_only=False, mmap=True)
        state = checkpoint.get("state_dict", checkpoint)
        prefix = (
            "objective.kp2d_outcoord_loss."
            if "gem_pose_training" in checkpoint
            else "kp2d_outcoord_loss."
        )
        density_state = {
            key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
        }
        loss.kp2d_outcoord_loss.load_state_dict(density_state, strict=True)
    options = {"fused": None, **dict(optimizer or {})}
    return PoseHandTrainingModule(
        model, loss, release=release, optimizer=options, grad_clip_val=grad_clip_val
    )
