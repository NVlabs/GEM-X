# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native body Pose supervision on prepared, augmented image batches."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gem.common.camera import compute_transl_full_cam
from gem.common.geometry.rotations import aa_to_rotmat
from gem.pose.heads.output import PoseHeadOutput
from gem.pose.heads.soma_body_head import FINGER_BODY_POSE_INDICES_76

DEFAULT_BODY_WEIGHTS = {
    "keypoints_2d": 0.01,
    "landmarks_2d": 0.0025,
    "keypoints_3d": 0.05,
    "body_pose": 0.01,
    "global_orient": 0.01,
    "identity_coeffs": 0.0005,
    "scale_params": 0.0005,
    "camera": 0.001,
}


def _project_training(points: Tensor, intrinsics: Tensor, minimum_depth: float):
    """Pinhole projection with an inert invalid branch, before any division."""
    valid = (torch.isfinite(points).all(-1) & (points[..., 2] > minimum_depth)).detach()
    xy = torch.where(valid[..., None], points[..., :2], torch.zeros_like(points[..., :2]))
    depth = torch.where(valid, points[..., 2], torch.ones_like(points[..., 2]))
    homogeneous = torch.cat((xy / depth[..., None], torch.ones_like(depth)[..., None]), -1)
    pixels = torch.einsum("bij,bkj->bki", intrinsics, homogeneous)
    return pixels[..., :2], valid


def _correct_rotated_targets(targets: Tensor, batch: Mapping, bbox: Tensor, intrinsics: Tensor):
    """Match crop-rotated labels to full-camera projection around its principal point."""
    rotation = batch.get("aug_rot")
    if rotation is None:
        return targets
    theta = torch.deg2rad(torch.as_tensor(rotation, device=targets.device, dtype=targets.dtype))
    if theta.ndim == 0:
        theta = theta.expand(targets.shape[0])
    dx, dy = ((intrinsics[:, :2, 2] - bbox[:, :2]) / bbox[:, 2:]).unbind(-1)
    cosine, sine = theta.cos(), theta.sin()
    correction = torch.stack(((cosine - 1) * dx - sine * dy, sine * dx + (cosine - 1) * dy), -1)
    corrected = targets.clone()
    corrected[..., :2] = corrected[..., :2] + correction[:, None]
    return corrected


class BodyPoseObjective(nn.Module):
    """Final-layer body loss with local56 scales and architectural global scale1.

    ``geometry`` is the differentiable, frozen BodyGeometry adapter. Its runtime
    owns checkpoint exclusion for external SOMA assets. ``landmark_projector``
    is an optional caller-supplied ``[N,V]`` matrix, registered nonpersistently;
    the current dense-label recipe uses N=308. Its prepared labels are named
    ``landmarks_2d`` and are optional independently of native77 labels.

    Prepared inputs keep augmented body/root axis angles, square full-image
    ``bbx_xys``, full ``K_fullimg``, centered-crop 2D labels and confidences,
    identity128, raw local56 scales, and legacy ``[s,tx,ty]`` camera GT. The
    head owns the camera STE floor. This module neither clamps scales nor
    un-rotates augmented predictions. All geometry and loss math is float32.
    """

    def __init__(
        self,
        geometry: nn.Module,
        landmark_projector: Tensor | None = None,
        *,
        weights: Mapping[str, float] | None = None,
        minimum_depth: float = 0.05,
        pelvis_ids: tuple[int, ...] = (0,),
        reduction: str = "sum",
        finger_weight: float = 1.0,
    ):
        super().__init__()
        self.geometry = geometry
        self.weights = {**DEFAULT_BODY_WEIGHTS, **(weights or {})}
        unsupported = [
            name
            for name, value in self.weights.items()
            if name not in DEFAULT_BODY_WEIGHTS and value
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported body loss weights {unsupported}; this recipe uses final-layer supervision"
            )
        if reduction not in {"sum", "mean"}:
            raise ValueError("Body loss reduction must be sum or mean")
        if minimum_depth <= 0:
            raise ValueError("minimum_depth must be positive")
        self.minimum_depth, self.pelvis_ids, self.reduction = minimum_depth, pelvis_ids, reduction
        self.finger_weight = float(finger_weight)
        self.register_buffer(
            "landmark_projector",
            None if landmark_projector is None else torch.as_tensor(landmark_projector).float(),
            persistent=False,
        )
        kp_weight = pose_weight = None
        if finger_weight != 1:
            kp_weight = torch.ones(77)
            kp_weight[[index + 1 for index in FINGER_BODY_POSE_INDICES_76]] = finger_weight
            pose_weight = torch.ones(76)
            pose_weight[FINGER_BODY_POSE_INDICES_76] = finger_weight**0.5
        self.register_buffer("finger_kp_weight", kp_weight, persistent=False)
        self.register_buffer("finger_pose_weight", pose_weight, persistent=False)

    def _reduce(self, value):
        return value.sum() if self.reduction == "sum" else value.mean()

    def _keypoints(self, prediction, target):
        confidence, xyz = target[..., -1:], target[..., :-1]
        valid = (
            (confidence > 0)
            & torch.isfinite(confidence)
            & torch.isfinite(xyz).all(-1, keepdim=True)
        )
        confidence = torch.where(valid, confidence.clamp_min(0), torch.zeros_like(confidence))
        xyz = torch.where(valid.expand_as(xyz), xyz, torch.zeros_like(xyz))
        prediction = torch.where(valid.expand_as(prediction), prediction, xyz)
        loss = confidence * F.l1_loss(prediction, xyz, reduction="none")
        return loss.mean() if self.reduction == "mean" else loss.sum(dim=(1, 2)).sum()

    def _parameter_loss(self, prediction, target, available):
        mask = available.to(prediction).reshape(-1, *([1] * (prediction.ndim - 1)))
        return self._reduce(mask * F.mse_loss(prediction, target, reduction="none"))

    def _keypoints_2d(self, prediction, target, valid, batch, bbox, intrinsics, joint_weight=None):
        target = _correct_rotated_targets(target.float(), batch, bbox, intrinsics)
        target = target.clone()
        if joint_weight is not None:
            target[..., 2] = target[..., 2] * joint_weight
        target[..., 2] = torch.where(valid, target[..., 2], torch.zeros_like(target[..., 2]))
        return self._keypoints(prediction, target)

    def forward(self, head: PoseHeadOutput, batch: Mapping) -> dict:
        parameters = head.parameters
        with torch.autocast(device_type=head.pred_cam.device.type, enabled=False):
            predicted = self.geometry(parameters)
            pred_cam = head.pred_cam.float()
            bbox, intrinsics = batch["bbx_xys"].float(), batch["K_fullimg"].float()
            translation = compute_transl_full_cam(pred_cam, bbox, intrinsics)

            def project(points):
                pixels, valid = _project_training(
                    points + translation[:, None], intrinsics, self.minimum_depth
                )
                return (pixels - bbox[:, None, :2]) / bbox[:, None, 2:], valid

            kp2d, valid = project(predicted["joints"])
            predictions = {
                "parameters": parameters,
                "pred_cam": pred_cam,
                "camera_translation": translation,
                "joints": predicted["joints"],
                "vertices": predicted["vertices"],
                "keypoints_2d": kp2d,
                "projection_valid": valid,
            }
            count = pred_cam.shape[0]
            gt_parameters = {
                "body_pose": aa_to_rotmat(batch["soma_body_pose"].float().reshape(-1, 3)).reshape(
                    count, 76, 3, 3
                ),
                "global_orient": aa_to_rotmat(
                    batch["soma_global_orient"].float().reshape(-1, 3)
                ).reshape(count, 1, 3, 3),
                "identity_coeffs": batch["soma_identity_coeffs"].float(),
                "scale_params": batch["soma_scale_params"].float(),
            }
            with torch.no_grad():
                target_geometry = self.geometry(gt_parameters)
            has_pose = batch.get("has_body_pose", batch.get("has_soma_params"))
            has_shape = batch.get("has_betas", batch.get("has_soma_params"))
            kp3d_conf = (
                batch["keypoints_3d"][..., 3:4].float() * (has_pose * has_shape)[:, None, None]
            )
            if self.finger_kp_weight is not None:
                kp3d_conf = kp3d_conf * self.finger_kp_weight[None, :, None]
            target_joints = target_geometry["joints"]
            pred_center = predicted["joints"][:, list(self.pelvis_ids)].mean(1, keepdim=True)
            target_center = target_joints[:, list(self.pelvis_ids)].mean(1, keepdim=True)
            target3d = torch.cat((target_joints - target_center, kp3d_conf), -1)
            pred_pose, gt_pose = parameters["body_pose"].float(), gt_parameters["body_pose"]
            if self.finger_pose_weight is not None:
                weight = self.finger_pose_weight[None, :, None, None]
                pred_pose, gt_pose = pred_pose * weight, gt_pose * weight
            terms = {
                "keypoints_2d": self._keypoints_2d(
                    kp2d,
                    batch["keypoints_2d"],
                    valid,
                    batch,
                    bbox,
                    intrinsics,
                    self.finger_kp_weight,
                ),
                "landmarks_2d": pred_cam.new_zeros(()),
                "keypoints_3d": self._keypoints(predicted["joints"] - pred_center, target3d),
                "body_pose": self._parameter_loss(
                    pred_pose.reshape(count, -1), gt_pose.reshape(count, -1), has_pose
                ),
                "global_orient": self._parameter_loss(
                    parameters["global_orient"].float().reshape(count, -1),
                    gt_parameters["global_orient"].reshape(count, -1),
                    has_pose,
                ),
                "identity_coeffs": self._parameter_loss(
                    parameters["identity_coeffs"].float(),
                    gt_parameters["identity_coeffs"],
                    has_shape,
                ),
                "scale_params": self._parameter_loss(
                    parameters["scale_params"].float(), gt_parameters["scale_params"], has_pose
                ),
            }
            if self.landmark_projector is not None:
                landmarks = torch.einsum(
                    "kv,bvd->bkd", self.landmark_projector.float(), predicted["vertices"]
                )
                projected, landmark_valid = project(landmarks)
                predictions.update(landmarks_2d=projected, landmark_projection_valid=landmark_valid)
                if "landmarks_2d" in batch:
                    terms["landmarks_2d"] = self._keypoints_2d(
                        projected, batch["landmarks_2d"], landmark_valid, batch, bbox, intrinsics
                    )
            terms["camera"] = pred_cam.new_zeros(())
            if self.weights["camera"] > 0:
                camera_error = F.mse_loss(pred_cam, batch["pred_cam_gt"].float(), reduction="none")
                if "has_cam_gt" in batch:
                    camera_error = camera_error * batch["has_cam_gt"].to(pred_cam).reshape(-1, 1)
                terms["camera"] = self._reduce(camera_error)
            loss = sum(self.weights[name] * value for name, value in terms.items())
        metric_names = {"identity_coeffs": "identity", "scale_params": "scale"}
        losses = {
            f"loss_{metric_names.get(name, name)}": value.detach() for name, value in terms.items()
        }
        losses["loss"] = loss.detach()
        return {"loss": loss, "losses": losses, "predictions": predictions}
