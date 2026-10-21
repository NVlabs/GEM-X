# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SOMA video losses with the deployed full-tensor reductions.

The caller prepares finite motion/geometry targets and observation conditions.
Sparse body-label availability stays independent of dense root-motion labels.
The supplied codec owns normalization and the configured differentiable FK.
"""

import torch
from torch import nn
from torch.nn import functional as F

from gem.common.camera import compute_transl_full_cam, get_a_pred_cam, project_to_bi01
from gem.common.geometry.trajectory import rollout_local_transl_vel
from gem.motion.codec.soma_native_fixed_gs1 import MOTION_STEP_KEY, root_motion_rows

from .root_supervision import (
    get_static_joint_mask,
    incam_geometry_root_conditioning,
    override_source_static_time,
    root_motion_simple_mask,
    segmented_root_rollout,
)

DEFAULT_VIDEO_WEIGHTS = {
    "cr_j3d": 500.0,
    "transl_c": 1.0,
    "cr_verts": 500.0,
    "j2d": 1000.0,
    "verts2d": 1000.0,
    "transl_w": 1.0,
    "static_conf_bce": 1.0,
}


def supervision_mask(inputs):
    masks = inputs["mask"]
    valid = masks["valid"]
    return valid & masks["label_valid"] if "label_valid" in masks else valid


def make_motion_field_weights(layout, configured_weights, reference):
    weights = torch.ones(layout.output_dim, device=reference.device, dtype=reference.dtype)
    for field_name, value in configured_weights.items():
        value = float(value)
        if value < 0:
            raise ValueError(f"motion field weight for {field_name} must be non-negative")
        weights[layout.field_slice(field_name)] = value
    return weights


def compute_full_frame_reprojection_loss(pred_points, gt_points, bbx_xys, K_fullimg, frame_mask):
    """Preserve the full-frame projection and denominator of the video recipe."""
    threshold = 0.3
    pred_projectable = pred_points.clone()
    gt_projectable = gt_points.clone()
    pred_projectable[pred_projectable[..., 2].abs() <= threshold] = threshold
    gt_projectable[gt_projectable[..., 2].abs() <= threshold] = threshold
    pred_2d = project_to_bi01(pred_projectable, bbx_xys, K_fullimg)
    gt_2d = project_to_bi01(gt_projectable, bbx_xys, K_fullimg)
    valid_points = (
        (gt_projectable[..., 2] > threshold)
        & (pred_projectable[..., 2] > threshold)
        & (gt_2d[..., 0] > 0.0)
        & (gt_2d[..., 0] < 1.0)
        & (gt_2d[..., 1] > 0.0)
        & (gt_2d[..., 1] < 1.0)
    )
    error = F.mse_loss(pred_2d, gt_2d, reduction="none")
    return (error * (valid_points & frame_mask[..., None])[..., None]).mean()


class VideoLoss(nn.Module):
    """Losses for already prepared video batches.

    Geometry labels ``gt_j3d``, ``gt_c_vertices`` and ``gt_cr_vertices`` are
    prepared by the caller after applying any independent sparse incam fits.
    The compact/77 observation schema does not alter the full public77 loss
    skeleton or the codec's 655D representation.

    ``root_motion_supervision=True`` retains the formal recipe's legacy
    0.033-second contact divisor for ordinary sources and masks contact labels
    on dense-root rows. Explicit ``static_fps_by_source`` overrides individual
    source clocks. With False, motion step factors also scale contact timing.
    """

    def __init__(
        self,
        weights=None,
        *,
        motion_field_weights=None,
        static_velocity_threshold=0.15,
        static_fps_by_source=None,
        root_motion_supervision=True,
    ):
        super().__init__()
        self.weights = dict(DEFAULT_VIDEO_WEIGHTS)
        if weights is not None:
            self.weights.update(
                {name: float(weights[name]) for name in self.weights if name in weights}
            )
        self.motion_field_weights = dict(motion_field_weights or {})
        self.static_velocity_threshold = float(static_velocity_threshold)
        self.static_fps_by_source = dict(static_fps_by_source or {})
        self.root_motion_supervision = root_motion_supervision

    def forward(self, model_output, inputs, *, codec):
        root_rows = root_motion_rows(inputs)
        if root_rows is not None and bool(root_rows.any()) and not self.root_motion_supervision:
            raise ValueError("Root-motion targets require explicit root-motion supervision")
        prediction = model_output["pred_x"]
        decoded = codec.decode(prediction)
        translation = compute_transl_full_cam(
            model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
        )
        camera_params = codec.build_incam_params(decoded, translation)
        target = codec.encode(inputs)
        error = F.mse_loss(prediction, target, reduction="none")
        mask = supervision_mask(inputs)[..., None].expand_as(prediction).clone()
        for region in codec.layout.global_supervision_slices:
            mask[inputs["mask"]["spv_incam_only"], :, region] = False
        mask = root_motion_simple_mask(inputs, mask, codec.layout)
        if self.motion_field_weights:
            error = error * make_motion_field_weights(
                codec.layout, self.motion_field_weights, prediction
            )
        simple_loss = (error * mask).mean()
        result = {
            "model_output": model_output,
            "decode_dict": decoded,
            "simple_loss": simple_loss,
            "pred_body_params_incam": camera_params,
            "pred_soma_params_incam": camera_params,
        }
        incam = self._incam_losses(model_output, inputs, camera_params, codec)
        global_losses = self._global_losses(model_output, inputs, decoded, codec)
        result.update(incam)
        result.update(global_losses)
        # Keep the original per-group accumulation order as well as each
        # loss's full-tensor denominator.
        incam_total = 0
        for name, key in (
            ("cr_j3d", "cr_j3d_loss"),
            ("transl_c", "transl_c_loss"),
            ("j2d", "j2d_loss"),
            ("cr_verts", "cr_vert_loss"),
            ("verts2d", "verts2d_loss"),
        ):
            if key in incam:
                incam_total = incam_total + self.weights[name] * incam[key]
        global_total = 0
        for name, key in (
            ("transl_w", "transl_w_loss"),
            ("static_conf_bce", "static_conf_loss"),
        ):
            if key in global_losses:
                global_total = global_total + self.weights[name] * global_losses[key]
        result["loss"] = simple_loss + incam_total + global_total
        return result

    def _incam_losses(self, output, inputs, camera_params, codec):
        weights = self.weights
        result = {}
        if not any(
            weights[name] > 0 for name in ("cr_j3d", "transl_c", "cr_verts", "j2d", "verts2d")
        ):
            return result
        mask = supervision_mask(inputs)
        reprojection_rows = inputs["mask"].get(
            "spv_reprojection", ~inputs["mask"]["spv_incam_only"]
        )
        reprojection_frames = mask & reprojection_rows[:, None]
        geometry_params = incam_geometry_root_conditioning(inputs, camera_params)
        predicted = codec.forward_body(**geometry_params)
        joints, vertices = predicted["joints"], predicted["vertices"]
        target_joints = inputs["gt_j3d"]
        target_params = codec.get_input_params(inputs, "c")
        if weights["cr_j3d"] > 0:
            error = F.mse_loss(
                joints - joints[:, :, :1],
                target_joints - target_joints[:, :, :1],
                reduction="none",
            )
            result["cr_j3d_loss"] = (error * mask[..., None, None]).mean()
        if weights["transl_c"] > 0:
            target_cam = get_a_pred_cam(
                target_params["transl"], inputs["bbx_xys"], inputs["K_fullimg"]
            )
            target_cam[target_cam.isinf()] = -1
            valid_camera = (
                (target_joints[..., 2].min(dim=-1)[0] > 0.3)
                & (target_cam[..., 0] > 0.3)
                & (target_cam[..., 0] < 5.0)
                & (target_cam[..., 1] > -3.0)
                & (target_cam[..., 1] < 3.0)
                & (target_cam[..., 2] > -3.0)
                & (target_cam[..., 2] < 3.0)
                & (inputs["bbx_xys"][..., 2] > 0)
            )
            error = F.mse_loss(output["pred_cam"], target_cam, reduction="none")
            result["transl_c_loss"] = (error * mask[..., None] * valid_camera[..., None]).mean()
        if weights["j2d"] > 0:
            result["j2d_loss"] = compute_full_frame_reprojection_loss(
                joints, target_joints, inputs["bbx_xys"], inputs["K_fullimg"], reprojection_frames
            )
        if weights["cr_verts"] > 0:
            target_vertices = inputs.get("gt_cr_vertices", inputs.get("gt_cr_verts437"))
            error = F.mse_loss(
                vertices - codec.surface_root(predicted), target_vertices, reduction="none"
            )
            result["cr_vert_loss"] = (error * mask[..., None, None]).mean()
        if weights["verts2d"] > 0:
            result["verts2d_loss"] = compute_full_frame_reprojection_loss(
                vertices,
                inputs.get("gt_c_vertices", inputs.get("gt_c_verts437")),
                inputs["bbx_xys"],
                inputs["K_fullimg"],
                reprojection_frames,
            )
        return result

    def _global_losses(self, output, inputs, decoded, codec):
        result = {}
        mask = supervision_mask(inputs).clone()
        mask[inputs["mask"]["spv_incam_only"]] = False
        root_rows = root_motion_rows(inputs)
        if self.weights["transl_w"] > 0:
            params = codec.get_input_params(inputs, "w")
            target_translation = params["transl"]
            velocity = decoded["local_transl_vel"]
            if MOTION_STEP_KEY in inputs:
                velocity = velocity * inputs[MOTION_STEP_KEY][..., None]
            prediction = rollout_local_transl_vel(
                velocity, params["global_orient"], target_translation[:, [0]]
            )
            position_mask = mask
            if root_rows is not None:
                targets = inputs["motion_targets"]
                position_valid = inputs["mask"]["root_position"] & inputs["mask"]["valid"]
                interval_valid = inputs["mask"]["root_interval"] & inputs["mask"]["valid"]
                root_prediction = segmented_root_rollout(
                    velocity,
                    targets["global_orient_w"],
                    targets["transl_w"],
                    position_valid,
                    interval_valid,
                )
                prediction = torch.where(root_rows[:, None, None], root_prediction, prediction)
                target_translation = torch.where(
                    root_rows[:, None, None], targets["transl_w"], target_translation
                )
                position_mask = torch.where(root_rows[:, None], position_valid, mask)
            error = F.l1_loss(prediction, target_translation, reduction="none")
            result["transl_w_loss"] = (error * position_mask[..., None]).mean()
        if self.weights["static_conf_bce"] > 0:
            joints = codec.fk_v2(**codec.get_input_params(inputs, "w"))
            intervals = None
            if MOTION_STEP_KEY in inputs and not self.root_motion_supervision:
                intervals = inputs[MOTION_STEP_KEY][:, :-1] / 30.0
            static = get_static_joint_mask(
                joints,
                vel_thr=self.static_velocity_threshold,
                repeat_last=True,
                interval_seconds=intervals,
            )
            static = override_source_static_time(
                static, joints, inputs, self.static_velocity_threshold, self.static_fps_by_source
            )
            target = static[:, :, codec.contact_joint_ids].float()
            static_mask = mask if root_rows is None else mask & ~root_rows[:, None]
            error = F.binary_cross_entropy_with_logits(
                output["static_conf_logits"], target, reduction="none"
            )
            result["static_conf_loss"] = (error * static_mask[..., None]).mean()
        return result
