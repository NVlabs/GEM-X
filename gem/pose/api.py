# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load GEM-Pose parameter models or complete image predictors.

Architecture, crop and camera conventions come from explicit release settings.
Checkpoint ``hyper_parameters`` never instantiate model classes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import torch
from torch import Tensor

from gem.common.backbones import DinoV3Backbone, convert_dinov3_native_state_dict

from .conditioning import CliffConditioning
from .heads import SOMABodyHeadV2, SOMAHandHeadV2
from .model import PoseModel
from .release_ids import BODY_RELEASE_ID, LEGACY_BODY_RELEASE_ID
from .tasks import BodyTask, HandTask


@dataclass
class PoseReleaseConfig:
    """Architecture and crop semantics supplied by a public release config."""

    task: str
    head: dict[str, Any]
    backbone: dict[str, Any] = field(default_factory=dict)
    conditioning: dict[str, Any] | None = None
    backbone_input_crop_w: tuple[int, int] | None = None
    ray_pe_num_bands: int | None = None
    kp2d_translation: dict[str, Any] | None = None
    pose_release_id: str | None = None
    inference: dict[str, Any] = field(default_factory=dict)


def body_config() -> PoseReleaseConfig:
    """Explicit architecture of the released Body Pose s240000 model.

    This historical release returns packed GS1+local56. Newer body heads may
    return raw local56; both preserve the same native fixed-GS1 geometry.
    """
    return PoseReleaseConfig(
        task="body",
        backbone={"name": "dinov3_vith16plus", "drop_path_rate": 0.1},
        head={
            "num_body_joints": 76,
            "num_identity_coeffs": 128,
            "num_scale_params": 57,
            "scale_comps_dim": 0,
            "scale_param_mode": "direct",
            "scale_parameterization": "packed",
            "fixed_global_scale": 1.0,
            "transformer_dim": 1024,
            "transformer_depth": 6,
            "transformer_heads": 8,
            "transformer_dim_head": 64,
            "transformer_mlp_dim": 1024,
            "transformer_context_dim": 1280,
            "transformer_dropout": 0.0,
            "transformer_emb_dropout": 0.0,
            "ief_iters": 1,
            "learnable_init": True,
            "use_init_to_token": True,
            "two_layer_heads": True,
            "per_layer_predictions": True,
            "cliff_cond_dim": 3,
            "camera_scale_mode": "ste_clamp",
        },
        conditioning={"camera_mode": "sam3db_perspective"},
        pose_release_id=BODY_RELEASE_ID,
        inference={"camera_representation": "legacy", "minimum_depth": 0.05},
    )


def body_sg6_config() -> PoseReleaseConfig:
    """Compatibility preset retaining the historical Body release identifier."""
    release = body_config()
    release.pose_release_id = LEGACY_BODY_RELEASE_ID
    return release


def hand_config() -> PoseReleaseConfig:
    """Final EgoHand GA s060000: RayPE, log-lambda depth and separate 2D joints."""
    return PoseReleaseConfig(
        task="hand",
        backbone={"name": "dinov3_vith16plus", "drop_path_rate": 0.0},
        head={
            "num_finger_joints": 24,
            "num_identity_coeffs": 5,
            "num_scale_params": 27,
            "transformer_dim": 1024,
            "transformer_depth": 6,
            "transformer_heads": 8,
            "transformer_dim_head": 64,
            "transformer_mlp_dim": 1024,
            "transformer_context_dim": 1280,
            "transformer_dropout": 0.0,
            "transformer_emb_dropout": 0.0,
            "ief_iters": 1,
            "learnable_init": True,
            "use_init_to_token": True,
            "two_layer_heads": True,
            "per_layer_predictions": True,
            "cliff_cond_dim": 3,
            "ray_cond_dim": 32,
            "camera_scale_mode": "ste_clamp",
            "camera_depth_mode": "log_lambda",
            "init_global_scale": 1.0,
            "init_camera_scalar": -1.8199350824986398,
            "camera_scalar_std": 0.3735116511306386,
        },
        conditioning={"camera_mode": "sam3db_perspective"},
        ray_pe_num_bands=8,
        kp2d_translation={
            "enabled": True,
            "topology": "separate",
            "num_joints": 25,
            "d_model": 256,
            "feat_hw": [16, 16],
            "decoder_depth": 3,
            "num_heads": 8,
            "presence_threshold": 0.5,
            "negative_presence_weight": 1.0,
            "min_points": 4,
            "min_depth": 1.0e-4,
            "damping": 1.0e-6,
            "max_condition_number": 1.0e6,
        },
        pose_release_id="gem-pose-hand-ga-s060000",
        inference={
            "hand_mirror": "full_image",
            "camera_representation": "log_lambda",
            "camera_log_value_bounds": [-3.5, 3.0],
        },
    )


def hand_training_config(*, pose_release_id="gem-pose-hand-training") -> PoseReleaseConfig:
    """Train the current Hand architecture with reference stochastic depth."""
    release = hand_config()
    release.backbone["drop_path_rate"] = 0.1
    release.pose_release_id = pose_release_id
    return release


def body_native_config(*, pose_release_id="gem-pose-body-native-v1") -> PoseReleaseConfig:
    """Native body training layout: identity128, raw local56 and fixed GS1."""
    release = body_config()
    release.head.update(
        num_scale_params=56, scale_parameterization="local56", direct_scale_init="ones"
    )
    release.pose_release_id = pose_release_id
    return release


def build_pose_model(release: PoseReleaseConfig | Mapping[str, Any]) -> PoseModel:
    """Construct a parameter model without loading or downloading weights."""
    if not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    if release.task not in {"body", "hand"}:
        raise ValueError(f"Unsupported pose task {release.task!r}")
    options = dict(release.backbone)
    if isinstance(options.get("config"), Mapping):
        from transformers import DINOv3ViTConfig

        options["config"] = DINOv3ViTConfig(**options["config"])
    backbone = DinoV3Backbone(**options)
    task = (
        BodyTask(SOMABodyHeadV2(**release.head))
        if release.task == "body"
        else HandTask(SOMAHandHeadV2(**release.head))
    )
    kp2d_head = None
    keypoints = release.kp2d_translation or {}
    if keypoints.get("enabled", False):
        from .heads.rletr_occ_head import RLETransformerOccHead

        if release.task != "hand" or keypoints.get("topology") != "separate":
            raise ValueError("Direct keypoint translation requires a separate Hand decoder")
        kp2d_head = RLETransformerOccHead(
            in_channels=release.head["transformer_context_dim"],
            num_joints=keypoints["num_joints"],
            d_model=keypoints["d_model"],
            depth=keypoints["decoder_depth"],
            n_heads=keypoints["num_heads"],
            feat_hw=tuple(keypoints["feat_hw"]),
        )
    return PoseModel(
        backbone,
        task,
        conditioning=(
            None if release.conditioning is None else CliffConditioning(**release.conditioning)
        ),
        backbone_input_crop_w=release.backbone_input_crop_w,
        ray_pe_num_bands=release.ray_pe_num_bands,
        kp2d_head=kp2d_head,
        kp2d_translation=release.kp2d_translation,
        pose_release_id=release.pose_release_id,
    )


def checkpoint_pose_release(checkpoint_path):
    """Read the architecture saved by the native Pose trainer, if present."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    metadata = checkpoint.get("gem_pose_training")
    return None if metadata is None else PoseReleaseConfig(**metadata["release"])


def convert_pose_state_dict(state_dict: Mapping[str, Tensor], *, task: str) -> dict[str, Tensor]:
    """Map every legacy or native parameter to the unified model namespace.

    The backbone converter consumes the saved RoPE periods and applies the
    original fixed QKV bias masks. Unknown tensors and missing learned weights
    are errors, so a partially initialized model is never returned.
    """
    if task not in {"body", "hand"}:
        raise ValueError(f"Unsupported pose task {task!r}")
    backbone_prefix = "backbone.encoder."
    head_prefix = "soma_head." if task == "body" else "hand_head."
    prefixes = (backbone_prefix, head_prefix, "task.head.")
    if task == "hand":
        # The RLE density is learned by the objective, but unused by inference.
        prefixes += ("kp2d_head.", "kp2d_outcoord_loss.")
    unknown = [key for key in state_dict if not key.startswith(prefixes)]
    if unknown:
        raise ValueError(f"Unsupported pose checkpoint tensors: {unknown}")
    if backbone_prefix + "cls_token" in state_dict:
        backbone = convert_dinov3_native_state_dict(state_dict, prefix=backbone_prefix)
        converted = {backbone_prefix + key: value for key, value in backbone.items()}
    else:
        converted = {
            key: value for key, value in state_dict.items() if key.startswith(backbone_prefix)
        }
    legacy_head = any(key.startswith(head_prefix) for key in state_dict)
    unified_head = any(key.startswith("task.head.") for key in state_dict)
    if legacy_head and unified_head:
        raise ValueError("Checkpoint mixes legacy and unified pose head namespaces")
    for key, value in state_dict.items():
        if key.startswith(head_prefix):
            converted["task.head." + key[len(head_prefix) :]] = value
        elif key.startswith("task.head."):
            converted[key] = value
        elif task == "hand" and key.startswith("kp2d_head."):
            converted[key] = value
    return converted


def load_pose_checkpoint(
    checkpoint_path: str | PathLike[str],
    release: PoseReleaseConfig | Mapping[str, Any] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> PoseModel:
    """Load a trusted local checkpoint using caller-supplied release settings.

    Body releases must supply their actual head/camera configuration. The
    hand preset is available via :func:`load_hand_checkpoint`. Meta-device
    construction plus mmap avoids allocating a second ViT-H copy on the CPU.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    metadata = checkpoint.get("gem_pose_training")
    if release is None:
        if metadata is None:
            raise ValueError("Legacy Pose checkpoints require an explicit release configuration")
        release = PoseReleaseConfig(**metadata["release"])
    elif not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if metadata is not None:
        state_dict = {
            key.removeprefix("pose_model."): value
            for key, value in state_dict.items()
            if key.startswith("pose_model.")
        }
    converted = convert_pose_state_dict(state_dict, task=release.task)
    with torch.device("meta"):
        model = build_pose_model(release)
    model.load_state_dict(converted, strict=True, assign=True)
    return model.to(device).eval()


def load_hand_checkpoint(
    checkpoint_path: str | PathLike[str], *, device: str | torch.device = "cpu"
) -> PoseModel:
    """Load the current Hand checkpoint for normalized canonical-left crops."""
    return load_pose_checkpoint(checkpoint_path, hand_config(), device=device)


def load_pose_predictor(checkpoint_path, release, *, soma_assets, device="cpu"):
    """Load weights plus lazy SOMA FK for RGB/bbox or normalized-crop prediction.

    Assets must be the tested SOMA0.3.3 release, including body correctives or
    the hand MHR files for the selected task. Loading weights does not fetch
    assets or modify the installed runtime. Use ``load_pose_checkpoint`` when
    only parameters/features are needed.
    """
    from .predictor import PoseInferenceConfig, PosePredictor

    if not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    model = load_pose_checkpoint(checkpoint_path, release, device=device)
    if release.task == "body":
        from .geometry.body import BodyGeometry

        geometry = BodyGeometry(
            soma_assets,
            device=device,
            scale_parameterization=release.head.get("scale_parameterization", "packed"),
        )
    else:
        from .geometry.hand import HandGeometry

        geometry = HandGeometry(soma_assets)
    camera = {
        key: value
        for key, value in (release.conditioning or {}).items()
        if key in {"camera_mode", "focal_length", "focal_length_mode"}
    }
    config = PoseInferenceConfig(**{**camera, **release.inference})
    return PosePredictor(model, geometry, config).to(device).eval()


def load_hand_predictor(checkpoint_path, *, soma_assets, device="cpu", hand_mirror="full_image"):
    """Load the current Hand model with an explicit right-hand raster convention.

    ``full_image`` follows the calibrated deployment preprocessing;
    ``legacy_crop`` retains the historical HOT3D crop-then-flip convention.
    """
    release = hand_config()
    release.inference["hand_mirror"] = hand_mirror
    return load_pose_predictor(checkpoint_path, release, soma_assets=soma_assets, device=device)
