# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native observation inputs for the B2 image/video training recipe.

The caller owns the codec, its injected FK implementation and training state.
This helper changes observation inputs and constructs clean video loss geometry;
it does not register another codec, change supervision, or own an optimizer.
"""

from __future__ import annotations

import math

import torch

from gem.common.camera import compute_bbox_info_bedlam, perspective_projection
from gem.motion.frontend.joint_sets import (
    SOMA29_NOFINGER,
    SOMA_OBSERVATION_SCHEMAS,
    get_observation_schema,
    select_observation_joints,
)
from gem.motion.frontend.observation import normalize_keypoints

from .conditions import image_t1_conditions
from .losses import parameter_masks, project_valid_bi01, safe_native_targets
from .observation_augmentation import (
    AUGMENTATION_SEED_OFFSET,
    augment_image_observations,
    observation_augmentation_profile,
)
from .observation_noise import (
    get_observation_aug_kp3d,
    get_observation_lower_body_invisible_mask,
    get_observation_visible_mask,
    make_observation_noise_generator,
    randomly_modify_observation_hands_legs,
)


def generated_bboxes(pixels):
    """The training crop policy, including its three full-shape RNG draws."""
    min_x, max_x = pixels[..., 0].amin(-1), pixels[..., 0].amax(-1)
    min_y, max_y = pixels[..., 1].amin(-1), pixels[..., 1].amax(-1)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    height, width = max_y - min_y, max_x - min_x
    aspect = 192 / 256
    wide = width > aspect * height
    height[wide] = width[wide] / aspect
    tall = width < aspect * height
    width[tall] = height[tall] * aspect
    # Retain the deployed multiply/divide order before random enlargement.
    size = torch.maximum(height, width) * 1.2
    shape, device = size.shape, size.device
    scale = torch.rand(shape, device=device) * 0.3 + 1.05
    shift_x = torch.rand(shape, device=device) * 1.6 - 0.8
    shift_y = torch.rand(shape, device=device) * 1.6 - 0.8
    raw_size = size / 1.2
    center_x += raw_size / 2 * ((scale - 1) * shift_x)
    center_y += raw_size / 2 * ((scale - 1) * shift_y)
    return torch.stack((center_x, center_y, raw_size * scale), -1)


def replace_sparse_soma_gt_geometry(codec, params, body, masks):
    """Sparse per-image fits retain independent shapes within their video clip."""
    if "label_valid" not in masks:
        return
    selected = masks["valid"] & masks["label_valid"] & masks["spv_incam_only"][:, None]
    sample_ids, frame_ids = selected.nonzero(as_tuple=True)
    if sample_ids.numel() == 0:
        return
    independent_params = {
        name: (
            value[sample_ids]
            if name in {"identity_coeffs", "scale_params"} and value.ndim == 2
            else value[sample_ids, frame_ids]
        )
        for name, value in params.items()
    }
    independent = codec.forward_body(**independent_params)
    for name in ("joints", "vertices"):
        body[name][sample_ids, frame_ids] = independent[name]


def apply_sparse_training_conditions(batch):
    """Unlabeled video frames use real detector observations or remain absent."""
    masks = batch["mask"]
    if "label_valid" in masks:
        unlabeled = ~masks["label_valid"]
        detected = unlabeled & masks.get("kp2d", False)
        batch["obs"][unlabeled] = 0
        batch["obs"][detected] = normalize_keypoints(batch["kp2d"], batch["bbx_xys"])[detected]
    if "imgfeat" in masks:
        batch["f_imgseq"] = batch["f_imgseq"].masked_fill(~masks["imgfeat"][..., None], 0)


def apply_unlabeled_detected_observation_noise(batch, config, *, generator):
    """Apply the isolated E/H XY noise stream without changing confidence/GT."""
    obs, masks = batch["obs"], batch["mask"]
    batch_size, length, joints = obs.shape[:3]
    source_scales = config.get("source_joint_scales")
    sources = tuple(config["sources"] if "sources" in config else source_scales)
    scales = None
    if source_scales is not None:
        table = obs.new_tensor(
            [[[1.0, 1.0]] * joints, *(source_scales[source] for source in sources)]
        )
        source_to_id = {source: index + 1 for index, source in enumerate(sources)}
        source_ids = torch.tensor(
            [source_to_id.get(meta["data_name"], 0) for meta in batch["meta"]],
            device=obs.device,
        )
        scales = table[source_ids, None]
    detected_source = torch.tensor(
        [meta["data_name"] in sources for meta in batch["meta"]], device=obs.device
    )
    frames = (
        detected_source[:, None]
        & masks["valid"]
        & ~masks.get("label_valid", masks["valid"])
        & masks.get("kp2d", False)
    )
    selected = frames[..., None] & (obs[..., 2] > 0.5)
    bias = torch.randn(
        batch_size, 1, joints, 2, device=obs.device, dtype=obs.dtype, generator=generator
    )
    temporal = torch.randn(
        batch_size, length, joints, 2, device=obs.device, dtype=obs.dtype, generator=generator
    )
    rho = config.get("rho", 0.95)
    innovation_scale = math.sqrt(1 - rho * rho)
    for frame in range(1, length):
        temporal[:, frame] = rho * temporal[:, frame - 1] + innovation_scale * temporal[:, frame]
    noise = bias * config.get("bias_std", 0.005) + temporal * config.get("temporal_std", 0.002)
    if scales is not None:
        noise = noise * scales
    changed = obs.clone()
    xy = changed[..., :2]
    xy[selected] = xy[selected] + noise[selected]
    batch["obs"] = changed


class TrainingPreparation:
    """Native GT image/T1 and native GT/detected video input construction."""

    def __init__(
        self,
        codec,
        schema=SOMA29_NOFINGER,
        *,
        observation_augmentation="common2d_v1",
        detected_observation_noise=None,
        cache_soma_aug_profiles=False,
    ):
        self.codec = codec
        self.schema = get_observation_schema(schema)
        if self.schema.name not in SOMA_OBSERVATION_SCHEMAS:
            raise ValueError("Native training preparation requires SOMA29 or SOMA77 observations")
        observation_augmentation_profile(observation_augmentation, schema=self.schema)
        self.observation_augmentation = observation_augmentation
        self.cache_soma_aug_profiles = cache_soma_aug_profiles
        if detected_observation_noise is not None:
            config = detected_observation_noise
            if (
                config.get("bias_std", 0.005) < 0
                or config.get("temporal_std", 0.002) < 0
                or not 0 <= config.get("rho", 0.95) < 1
            ):
                raise ValueError("Observation noise requires nonnegative stds and rho in [0,1)")
            if not config.get("sources") and not config.get("source_joint_scales"):
                raise ValueError("Detected noise requires explicit sources or source_joint_scales")
            if config.get("source_joint_scales") is not None:
                for source in config["source_joint_scales"]:
                    table = config["source_joint_scales"][source]
                    if len(table) != self.schema.num_joints or any(len(row) != 2 for row in table):
                        raise ValueError(
                            f"{source} noise scales must match ({self.schema.num_joints},2)"
                        )
        self.detected_observation_noise = detected_observation_noise

    @torch.no_grad()
    def prepare_video(self, batch, *, attempt_step=0, global_rank=0, world_size=1):
        """Return an independent input batch and full77/XLO loss geometry.

        The native recipe uses video clips longer than90 frames. Global Torch
        RNG owns GT noise and bbox augmentation; E/H noise uses restored attempt
        counters without advancing that stream. Pipeline modality dropout occurs
        after this method, as in the established mixed image/video recipe.
        """
        params = self.codec.get_input_params(batch, "c")
        body = self.codec.forward_body(**params)
        replace_sparse_soma_gt_geometry(self.codec, params, body, batch["mask"])
        root = self.codec.surface_root(body)
        prepared = {
            **batch,
            "gt_j3d": body["joints"],
            "gt_c_vertices": body["vertices"],
            "gt_cr_vertices": body["vertices"] - root,
            "bbx_xys": batch["bbx_xys"].clone(),
        }
        clean = select_observation_joints(body["joints"], self.schema)
        with torch.autocast(device_type=clean.device.type, enabled=False):
            vertices = body["vertices"].float().clone()
            vertices[..., 2].clamp_(min=0.3)
            pixels = perspective_projection(vertices, batch["K_fullimg"].float())
        boxes = generated_bboxes(pixels)
        missing_boxes = ~batch["mask"]["bbx_xys"]
        prepared["bbx_xys"][missing_boxes] = boxes[missing_boxes]
        noisy = clean + get_observation_aug_kp3d(
            clean.shape[:2],
            self.schema,
            device=clean.device,
            dtype=clean.dtype,
            cache_profiles=self.cache_soma_aug_profiles,
        )
        noisy = randomly_modify_observation_hands_legs(noisy, self.schema)
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            pixels = perspective_projection(noisy.float(), batch["K_fullimg"].float())
        visible = get_observation_visible_mask(clean.shape[:2], self.schema, device=clean.device)
        visible[noisy[..., 2] < 0.3] = False
        visible[
            get_observation_lower_body_invisible_mask(
                clean.shape[:2],
                self.schema,
                device=clean.device,
            )
        ] = False
        keypoints = torch.cat((pixels, visible[..., None].float()), -1)
        obs = normalize_keypoints(keypoints, prepared["bbx_xys"])
        obs[~visible] = 0
        # Preserve the reference's CPU draw even when batch tensors are on GPU.
        real = (torch.rand(len(obs)).to(keypoints) < 0.5) * batch["mask"]["vitpose"]
        obs[real] = normalize_keypoints(batch["kp2d"], prepared["bbx_xys"])[real]
        prepared["obs"] = obs
        apply_sparse_training_conditions(prepared)
        noise = self.detected_observation_noise
        if noise is not None:
            generator = make_observation_noise_generator(
                obs.device,
                seed=noise.get("seed", 42),
                attempt_step=attempt_step,
                global_rank=global_rank,
                world_size=world_size,
            )
            apply_unlabeled_detected_observation_noise(prepared, noise, generator=generator)
        prepared["obs"][~batch["mask"]["valid"]] = 0
        return prepared

    @torch.no_grad()
    def prepare_image_conditions(self, batch, *, attempt_step=0, global_rank=0, world_size=1):
        """Project reliable native image GT, then common2d and the B2 policy."""
        features = batch["f_imgseq"]
        if features.shape[1] != 1:
            raise ValueError("Native B2 image preparation requires T=1")
        labels = batch["image_labels"]
        masks = parameter_masks(labels)
        rows = masks["geometry"] & masks["camera"]
        obs = torch.zeros(
            len(features), 1, self.schema.num_joints, 3, dtype=torch.float32, device=features.device
        )
        if bool(rows.any()):
            targets = safe_native_targets(labels, masks)
            params = {name: value[rows] for name, value in targets.items()}
            with torch.autocast(device_type=features.device.type, enabled=False):
                body = self.codec.body_model(
                    **params,
                    apply_correctives=False,
                    repose_to_bind_pose=True,
                    return_joints_only=True,
                )
                joints = select_observation_joints(body["joints"], self.schema)
                uv, visible = project_valid_bi01(
                    joints,
                    batch["bbx_xys"][rows, :1].float(),
                    batch["K_fullimg"][rows, :1].float(),
                )
                visible &= torch.isfinite(uv).all(-1) & ((uv >= 0) & (uv <= 1)).all(-1)
                selected = torch.cat((2 * (uv - 0.5), visible[..., None].float()), -1)
                obs[rows] = selected.masked_fill(~visible[..., None], 0)
        if self.observation_augmentation != "none":
            generator = make_observation_noise_generator(
                obs.device,
                seed=42 + AUGMENTATION_SEED_OFFSET,
                attempt_step=attempt_step,
                global_rank=global_rank,
                world_size=world_size,
            )
            obs = augment_image_observations(
                obs,
                self.observation_augmentation,
                generator=generator,
                schema=self.schema,
            )
        cliff = compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"])
        return image_t1_conditions({**batch, "obs": obs}, cliff, "B2")
