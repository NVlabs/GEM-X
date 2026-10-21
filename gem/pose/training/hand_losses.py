# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-left MHR supervision with log-lambda and direct 2D keypoints."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from gem.common.camera import compute_transl_full_cam, perspective_projection
from gem.pose.conditioning.camera_depth import (
    depth_parameter_to_legacy_camera,
    legacy_camera_to_depth_parameter,
)
from gem.pose.conditioning.fixed_depth_translation import (
    camera_scale_to_translation_z,
    crop_keypoints_to_full_image,
    solve_lateral_translation_full_perspective,
)
from gem.pose.heads.output import PoseHeadOutput

from .rletr_occ_loss import RLEOccOutCoordLoss

DEFAULT_HAND_WEIGHTS = {
    "keypoints_2d": 0.01,
    "keypoints_3d": 0.05,
    "hand_pose": 0.001,
    "wrist_orient": 0.001,
    "identity_coeffs": 0.0005,
    "scale_params": 0.0005,
    "camera": 0.001,
    "kp2d_rle": 1.0,
    "kp2d_presence": 1.0,
    "kp2d_evidence": 0.1,
    "translation": 0.05,
}


class HandPoseObjective(nn.Module):
    """Final-layer, calibrated hand losses with the original masks and reductions.

    Prepared RGB, poses, camera and intrinsics already describe canonical LEFT
    hands, including any full-image camera roll. No side mirroring occurs here.
    GT rotations are matrices ``hand_poses[B,25,3,3]``. Shape is identity5 plus
    a learned global scale scalar and 26 MHR controls; these controls may be negative and are
    neither body local56 ratios nor logarithmic Motion scales.

    Both GT keypoint coordinate sets are rebuilt with frozen, differentiable
    hand geometry: input keypoint tensors supply only confidence. The 25-joint
    layout includes fingertips at 4,9,14,19,24; they take part in the ordinary
    keypoint losses without a separate fingertip term.

    Direct 2D RLE/presence/evidence losses, GT-masked translation and standardized
    log-camera supervision follow the current Hand recipe. Its density network
    is learned training state. ``pred_cam_gt`` remains legacy ``[s, dx, dy]``;
    only the neural camera output and its supervision use log-lambda.
    """

    camera_mode = "sam3db_perspective"
    camera_depth_mode = "log_lambda"
    camera_scalar_loss_type = "smooth_l1"
    projection_denominator_floor = 1e-6
    mask_predicted_depth = True
    mask_nonfinite_gt_projection = True

    def __init__(
        self,
        geometry: nn.Module,
        *,
        kp2d_translation,
        weights=None,
        reduction="sum",
        root_joint_ids=(0,),
        camera_depth_mode="log_lambda",
        camera_scalar_loss_type="smooth_l1",
        camera_scalar_loss_std=0.3735116511306386,
        camera_log_value_bounds=(-3.5, 3.0),
    ):
        super().__init__()
        self.geometry = geometry
        self.camera_depth_mode = camera_depth_mode
        self.camera_scalar_loss_type = camera_scalar_loss_type
        self.camera_scalar_loss_std = float(camera_scalar_loss_std)
        self.camera_log_value_bounds = tuple(camera_log_value_bounds)
        self.kp2d_translation = dict(kp2d_translation)
        if (
            camera_depth_mode != "log_lambda"
            or not self.kp2d_translation.get("enabled", False)
            or self.kp2d_translation.get("topology") != "separate"
        ):
            raise ValueError("Hand supervision requires log-lambda and the separate 2D head")
        # Match the source branch initialization without consuming model RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 103) % (2**63 - 1))
            self.kp2d_outcoord_loss = RLEOccOutCoordLoss(
                feat_hw=tuple(self.kp2d_translation.get("feat_hw", (16, 16))),
                negative_presence_weight=self.kp2d_translation.get("negative_presence_weight", 1.0),
            )
        self.weights = {**DEFAULT_HAND_WEIGHTS, **(weights or {})}
        unsupported = [
            name
            for name, weight in self.weights.items()
            if name not in DEFAULT_HAND_WEIGHTS and weight
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported hand losses {unsupported}; this recipe has final-layer supervision only"
            )
        if reduction not in {"sum", "mean"}:
            raise ValueError("Hand loss reduction must be sum or mean")
        if camera_scalar_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("Unknown hand camera scalar loss type")
        if not self.camera_scalar_loss_std > 0:
            raise ValueError("Camera scalar loss standard deviation must be positive")
        self.reduction = reduction
        self.root_joint_ids = tuple(root_joint_ids)

    def _reduce(self, values):
        return values.sum() if self.reduction == "sum" else values.mean()

    def _parameter_loss(self, prediction, target, available):
        mask = available.to(prediction).reshape(-1, *([1] * (prediction.ndim - 1)))
        return self._reduce(mask * F.mse_loss(prediction, target, reduction="none"))

    def _keypoints_2d(self, predicted, target_xy, confidence):
        # Only zero-confidence rows are made inert. Supervised nonfinite model
        # outputs must still reach the training module's nonfinite-step policy.
        active = confidence != 0
        prediction = torch.where(active, predicted, torch.zeros_like(predicted))
        target = torch.where(active, target_xy, torch.zeros_like(target_xy))
        confidence = torch.where(active, confidence, torch.zeros_like(confidence))
        values = confidence * F.l1_loss(prediction, target, reduction="none")
        return values.mean() if self.reduction == "mean" else values.sum(dim=(1, 2)).sum()

    def _project(self, joints, camera, bbox, intrinsics):
        translation = compute_transl_full_cam(camera, bbox, intrinsics)
        # GT follows the source projection floor. Predicted points use the
        # explicit positive-depth mask after the lateral-translation solve.
        pixels = perspective_projection(joints + translation[:, None], intrinsics)
        return translation, (pixels - bbox[:, None, :2]) / bbox[:, None, 2:]

    def _direct_losses(
        self, direct, joints, camera, batch, target, confidence, has_hand, gt_joints
    ):
        """Use GT geometric masks for training; never gate with predicted presence."""
        options = self.kp2d_translation
        bbox, intrinsics = batch["bbx_xys"].float(), batch["K_fullimg"].float()
        image_wh = batch["img_size"].float()
        has_cam = batch["has_cam_gt"].to(camera).reshape(-1)
        finite = torch.isfinite(target).all(-1)
        labeled = (
            (confidence[..., 0] > 0) & (has_hand[:, None] > 0) & (has_cam[:, None] > 0) & finite
        )
        target_safe = torch.where(finite[..., None], target, torch.zeros_like(target))
        full = crop_keypoints_to_full_image(target, bbox)
        inside_crop = finite & (target >= -0.5).all(-1) & (target < 0.5).all(-1)
        inside_image = (
            torch.isfinite(full).all(-1) & (full >= 0).all(-1) & (full < image_wh[:, None]).all(-1)
        )
        minimum_depth = options.get("min_depth", 1e-4)
        with torch.no_grad():
            gt_translation = compute_transl_full_cam(batch["pred_cam_gt"].float(), bbox, intrinsics)
            positive_gt = gt_joints[..., 2] + gt_translation[:, None, 2] > minimum_depth
        direct_terms = self.kp2d_outcoord_loss(
            {key: value.float() for key, value in direct.items()},
            target_safe,
            labeled,
            inside_crop & inside_image,
        )
        point_mask = labeled & inside_crop & inside_image & positive_gt
        translation_z = camera_scale_to_translation_z(camera[:, 0], bbox, intrinsics)
        solve = solve_lateral_translation_full_perspective(
            joints.float(),
            crop_keypoints_to_full_image(direct["mu"].float(), bbox),
            translation_z,
            intrinsics,
            point_mask=point_mask,
            min_points=options.get("min_points", 4),
            min_depth=minimum_depth,
            damping=options.get("damping", 1e-6),
            max_condition_number=options.get("max_condition_number", 1e6),
        )
        legacy_translation = compute_transl_full_cam(camera, bbox, intrinsics)
        translation = torch.where(
            solve.valid[:, None],
            torch.cat((solve.translation_xy, translation_z[:, None]), -1),
            legacy_translation,
        )
        points = joints.float() + translation[:, None]
        projection_valid = torch.isfinite(points).all(-1) & (points[..., 2] > minimum_depth)
        inert = torch.zeros_like(points)
        inert[..., 2] = 1
        safe_points = torch.where(projection_valid[..., None], points, inert)
        pixels = perspective_projection(safe_points, intrinsics)
        projected = (pixels - bbox[:, None, :2]) / bbox[:, None, 2:]
        translation_valid = solve.valid & (has_hand > 0) & (has_cam > 0)
        safe_pred = torch.where(
            translation_valid[:, None], translation, torch.zeros_like(translation)
        )
        safe_gt = torch.where(
            translation_valid[:, None], gt_translation, torch.zeros_like(gt_translation)
        )
        translation_error = (safe_pred - safe_gt).square()
        divisor = translation_valid.to(translation_error).sum().clamp_min(1.0)
        loss_xy = translation_error[:, :2].sum() / divisor
        loss_z = translation_error[:, 2].sum() / divisor
        terms = {
            "kp2d_rle": direct_terms["rle_loss"],
            "kp2d_presence": direct_terms["presence_loss"],
            "kp2d_evidence": direct_terms["evidence_crps_loss"],
            "translation": loss_xy + loss_z,
        }
        debug = {
            "camera_solve_valid": solve.valid,
            "camera_solve_point_mask": point_mask,
            "camera_solve_num_points": solve.num_points,
            "camera_solve_condition_number": solve.condition_number,
            "projection_valid": projection_valid.detach(),
            "translation_xy_loss": loss_xy,
            "translation_z_loss": loss_z,
        }
        return terms, translation, projected, debug

    def forward(self, head: PoseHeadOutput, batch: Mapping):
        parameters = head.parameters
        with torch.autocast(device_type=head.pred_cam.device.type, enabled=False):
            raw_camera = head.pred_cam.float()
            bbox, intrinsics = batch["bbx_xys"].float(), batch["K_fullimg"].float()
            camera = depth_parameter_to_legacy_camera(
                raw_camera,
                bbox,
                intrinsics,
                self.camera_depth_mode,
                log_value_bounds=self.camera_log_value_bounds,
            )
            count, device = len(camera), camera.device
            predicted = self.geometry(parameters)
            poses = batch["hand_poses"].float()
            global_scale = batch.get("hand_global_scale", camera.new_ones(count, 1))
            local_scale = batch.get("hand_scale_params", camera.new_zeros(count, 26))
            gt_scale = torch.cat((global_scale.float().reshape(count, 1), local_scale.float()), -1)
            gt_parameters = {
                "wrist_orient": poses[:, :1],
                "hand_pose": poses[:, 1:],
                "identity_coeffs": batch["hand_identity_coeffs"].float(),
                "scale_params": gt_scale,
            }
            has_hand = (
                batch.get("has_hand_params", camera.new_ones(count)).to(camera).reshape(count)
            )
            with torch.no_grad():
                target_geometry = self.geometry(gt_parameters)
            gt_camera = batch["pred_cam_gt"].float()
            gt_translation, gt_projected = self._project(
                target_geometry["joints"], gt_camera, bbox, intrinsics
            )
            finite_gt = torch.isfinite(gt_projected).all(-1)
            gt_projected_safe = torch.where(
                finite_gt[..., None], gt_projected, torch.zeros_like(gt_projected)
            )
            confidence2d = (
                batch["keypoints_2d"][..., 2:3].to(camera)
                * has_hand[:, None, None]
                * finite_gt[..., None]
            )
            confidence3d = batch["keypoints_3d"][..., 3:4].to(camera) * has_hand[:, None, None]
            direct_terms, translation, projected, debug = self._direct_losses(
                head.auxiliary["kp2d"],
                predicted["joints"],
                camera,
                batch,
                gt_projected,
                batch["keypoints_2d"][..., 2:3].to(camera),
                has_hand,
                target_geometry["joints"],
            )
            confidence2d = confidence2d * debug["projection_valid"][..., None]
            pred_root = predicted["joints"][:, list(self.root_joint_ids)].mean(1, keepdim=True)
            gt_root = target_geometry["joints"][:, list(self.root_joint_ids)].mean(1, keepdim=True)
            errors3d = confidence3d * F.l1_loss(
                predicted["joints"] - pred_root,
                target_geometry["joints"] - gt_root,
                reduction="none",
            )
            terms = {
                "keypoints_2d": self._keypoints_2d(projected, gt_projected_safe, confidence2d),
                "keypoints_3d": (
                    errors3d.mean() if self.reduction == "mean" else errors3d.sum(dim=(1, 2)).sum()
                ),
                "hand_pose": self._parameter_loss(
                    parameters["hand_pose"].float().reshape(count, -1),
                    poses[:, 1:].reshape(count, -1),
                    has_hand,
                ),
                "wrist_orient": self._parameter_loss(
                    parameters["wrist_orient"].float().reshape(count, -1),
                    poses[:, :1].reshape(count, -1),
                    has_hand,
                ),
                "identity_coeffs": self._parameter_loss(
                    parameters["identity_coeffs"].float(),
                    gt_parameters["identity_coeffs"],
                    has_hand,
                ),
                "scale_params": self._parameter_loss(
                    parameters["scale_params"].float(), gt_scale, has_hand
                ),
            }
            camera_valid = (
                batch.get("has_cam_gt", torch.ones(count, device=device, dtype=torch.bool))
                .to(device=device, dtype=torch.bool)
                .reshape(count)
            )
            camera_zero = (
                torch.where(
                    torch.isfinite(raw_camera), raw_camera, torch.zeros_like(raw_camera)
                ).sum()
                * 0
            )
            gt_parameter = legacy_camera_to_depth_parameter(
                gt_camera, bbox, intrinsics, self.camera_depth_mode
            )
            camera_valid = (
                camera_valid
                & torch.isfinite(gt_parameter).all(-1)
                & torch.isfinite(bbox).all(-1)
                & torch.isfinite(intrinsics).all(dim=(-2, -1))
                & (gt_camera[:, 0] > 0)
                & (bbox[:, 2] > 0)
                & (intrinsics[:, 0, 0] > 0)
            )
            terms["camera"] = camera_zero
            if self.weights["camera"] > 0:
                safe_camera = torch.where(
                    camera_valid[:, None], raw_camera, torch.zeros_like(raw_camera)
                )
                safe_gt_camera = torch.where(
                    camera_valid[:, None], gt_parameter, torch.zeros_like(gt_parameter)
                )
                error = (safe_camera[:, 0] - safe_gt_camera[:, 0]) / self.camera_scalar_loss_std
                depth_error = (
                    F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=1.0)
                    if self.camera_scalar_loss_type == "smooth_l1"
                    else error.square()
                )
                depth_loss = self._reduce(
                    torch.where(camera_valid, depth_error, torch.zeros_like(depth_error))
                )
                offset_error = F.mse_loss(
                    safe_camera[:, 1:], safe_gt_camera[:, 1:], reduction="none"
                )
                offset_loss = self._reduce(
                    torch.where(camera_valid[:, None], offset_error, torch.zeros_like(offset_error))
                )
                terms["camera"] = depth_loss + offset_loss
                debug.update(camera_depth_loss=depth_loss, camera_offset_loss=offset_loss)
            terms.update(direct_terms)
            loss = sum(self.weights[name] * value for name, value in terms.items())
            # Invalid camera outputs invalidate the attempted step even when
            # their camera target or hand labels are unavailable.
            camera_output_valid = torch.isfinite(raw_camera).all(-1)
            loss = (
                loss
                + torch.where(
                    camera_output_valid,
                    torch.zeros_like(camera_output_valid, dtype=loss.dtype),
                    torch.full_like(camera_output_valid, float("nan"), dtype=loss.dtype),
                ).sum()
            )
        names = {"identity_coeffs": "identity", "scale_params": "scale"}
        losses = {f"loss_{names.get(name, name)}": value.detach() for name, value in terms.items()}
        losses["loss"] = loss.detach()
        losses.update(
            loss_camera_depth=debug.get("camera_depth_loss", camera_zero).detach(),
            loss_camera_offset=debug.get("camera_offset_loss", camera_zero).detach(),
            loss_translation_xy=debug["translation_xy_loss"].detach(),
            loss_translation_z=debug["translation_z_loss"].detach(),
        )
        return {
            "loss": loss,
            "losses": losses,
            "predictions": {
                "parameters": parameters,
                "pred_cam": camera,
                "camera_translation": translation,
                "joints": predicted["joints"],
                "vertices": predicted["vertices"],
                "keypoints_2d": projected,
                "target_keypoints_2d": gt_projected_safe,
                "target_projection_valid": finite_gt,
                "target_camera_translation": gt_translation,
                **debug,
            },
        }
