# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera/local image supervision with differentiable MID Sapiens landmarks.

The caller supplies the shared video codec at forward time, so this module does
not register a second codec or change its normalization. Raw cached labels have
no T axis; repeated-image output frames share those same clean targets. World
labels, gravity-view orientation, root velocity, and contact are never constructed.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from gem.common.camera import compute_transl_full_cam, get_a_pred_cam, project_to_bi01
from gem.common.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

COMPACT30 = tuple(range(21)) + (41, 62) + tuple(range(63, 70))
DEFAULT_WEIGHTS = {
    "simple": 1.0,
    "cr_j3d": 500.0,
    "cr_verts": 500.0,
    "transl_c": 1.0,
    "j2d": 1000.0,
    "verts2d": 1000.0,
}


def parameter_masks(labels):
    """Respect source pose/shape masks independently of input-modality dropout."""
    base = labels["has_soma_params"].bool()
    pose = labels.get("has_body_pose", base).bool()
    body = pose[:, None].expand(-1, 76)
    if "body_pose_valid" in labels:
        body = body & labels["body_pose_valid"].bool()
    identity = labels.get("has_betas", base).bool()
    scale = labels.get("has_scale", pose).bool()
    root = labels.get("has_global_orient", pose).bool()
    return {
        "body": body,
        "identity": identity,
        "scale": scale,
        "root": root,
        "camera": labels["has_cam_gt"].bool(),
        "geometry": body.all(-1) & identity & scale & root,
    }


def _masked(value, valid, neutral=0.0):
    return torch.where(
        valid.reshape(*valid.shape, *([1] * (value.ndim - valid.ndim))),
        value.float(),
        torch.full_like(value, neutral, dtype=torch.float32),
    )


def safe_native_targets(labels, masks):
    """Missing fields get finite calculation placeholders, never supervision."""
    scales = _masked(labels["soma_scale_params"], masks["scale"], 1.0)
    return {
        "body_pose": _masked(labels["soma_body_pose"], masks["body"])[:, None],
        "global_orient": _masked(labels["soma_global_orient"], masks["root"])[:, None],
        "identity_coeffs": _masked(labels["soma_identity_coeffs"], masks["identity"]),
        "scale_params": scales,
        "transl": _masked(labels["soma_transl_m"], masks["camera"])[:, None],
    }


def local_target_fields(targets):
    return {
        "body_pose_r6d": matrix_to_rotation_6d(axis_angle_to_matrix(targets["body_pose"])).flatten(
            -2
        ),
        "identity_coeffs": targets["identity_coeffs"][:, None],
        "log_local_scale": targets["scale_params"][:, None, 1:].log(),
        "global_orient_c_r6d": matrix_to_rotation_6d(
            axis_angle_to_matrix(targets["global_orient"])
        ),
    }


def normalized_local_parameter_loss(pred_x, targets, masks, endecoder):
    field_masks = {
        "body_pose_r6d": masks["body"][:, None, :, None].expand(-1, 1, -1, 6).flatten(-2),
        "identity_coeffs": masks["identity"][:, None, None],
        "log_local_scale": masks["scale"][:, None, None],
        "global_orient_c_r6d": masks["root"][:, None, None],
    }
    loss = pred_x[..., :646].sum() * 0
    for name, raw in local_target_fields(targets).items():
        region = endecoder.layout.field_slice(name)
        target = (raw - endecoder.mean[region].to(raw)) / endecoder.std[region].to(raw)
        loss = loss + ((pred_x[..., region].float() - target).square() * field_masks[name]).sum()
    # Match A's complete 655-coordinate denominator, including its masked
    # global channels, without inventing any targets for those channels.
    return loss / pred_x.numel()


def decode_local_motion(pred_x, pred_cam, bbx, intrinsics, endecoder):
    raw = pred_x.float() * endecoder.std.to(pred_x).float() + endecoder.mean.to(pred_x).float()
    field = endecoder.layout.field_slice
    pose = raw[..., field("body_pose_r6d")].reshape(*raw.shape[:2], 76, 6)
    root = raw[..., field("global_orient_c_r6d")]
    local_scale = raw[..., field("log_local_scale")].exp()
    return {
        "body_pose": matrix_to_axis_angle(rotation_6d_to_matrix(pose)).flatten(-2),
        "global_orient": matrix_to_axis_angle(rotation_6d_to_matrix(root)),
        "identity_coeffs": raw[..., field("identity_coeffs")],
        "scale_params": torch.cat((torch.ones_like(local_scale[..., :1]), local_scale), -1),
        "transl": compute_transl_full_cam(pred_cam.float(), bbx, intrinsics),
    }


def project_valid_bi01(points, bbx, intrinsics, minimum_depth=0.3):
    valid = torch.isfinite(points).all(-1) & (points[..., 2] > minimum_depth)
    neutral = torch.zeros_like(points)
    neutral[..., 2] = 1.0
    projected = project_to_bi01(torch.where(valid[..., None], points, neutral), bbx, intrinsics)
    return projected, valid


def sapiens30_l1(points, labels, bbx, intrinsics):
    pred, positive = project_valid_bi01(points, bbx, intrinsics)
    target = labels["keypoints_2d_sapiens"][:, None, COMPACT30].float()
    confidence = target[..., 2]
    # GEP retains confidence exactly0.8; coordinates remain those of its crop.
    valid = positive & torch.isfinite(target).all(-1) & (confidence >= 0.8)
    valid = valid & (target[..., :2].abs() <= 0.5).all(-1)
    if "sapiens2_uv_sample_valid" in labels:
        valid = valid & labels["sapiens2_uv_sample_valid"].bool()[:, None, None]
    safe_target = torch.where(valid[..., None], target[..., :2] + 0.5, pred.detach())
    weight = torch.where(valid, confidence, torch.zeros_like(confidence))
    return ((pred - safe_target).abs() * weight[..., None]).mean(), valid


def _take(params, rows):
    return {name: value[rows] for name, value in params.items()}


class ImageT1Loss(nn.Module):
    def __init__(self, *, mid_body, m_soma, weights=None):
        """Use caller-owned SOMA FK and its MID vertex-to-landmark projection.

        ``mid_body`` must accept native SOMA parameters and return MID18056
        vertices. The shared codec supplies the XLO612/public77 body model.
        Geometry assets and initialization remain the caller's responsibility.
        ``m_soma`` is the deployed [308,18056] linear landmark projector.
        """
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(
                {name: float(weights[name]) for name in self.weights if name in weights}
            )
        self.mid_body = mid_body
        if m_soma.layout != torch.strided:
            m_soma = m_soma.to_dense()
        if tuple(m_soma.shape) != (308, 18056):
            raise ValueError(
                "Image landmark supervision requires the original MID M_soma[308,18056]"
            )
        self.register_buffer("m_soma30", m_soma[list(COMPACT30)].float().clone(), persistent=False)

    def forward(self, model_output, image_labels, *, endecoder):
        pred_x, pred_cam = model_output["pred_x"], model_output["pred_cam"]
        count = len(image_labels["bbx_xys"])
        if (
            pred_x.ndim != 3
            or pred_x.shape[0] != count
            or pred_x.shape[1] not in (1, 3)
            or pred_x.shape[2] != 655
            or tuple(pred_cam.shape) != (*pred_x.shape[:2], 3)
        ):
            raise ValueError("Image loss requires matching outputs[B,T,655] and[B,T,3], T=1 or3")
        with torch.autocast(device_type=pred_x.device.type, enabled=False):
            return self._forward_fp32(pred_x.float(), pred_cam.float(), image_labels, endecoder)

    def _forward_fp32(self, pred_x, pred_cam, labels, endecoder):
        batch_size = len(pred_x)
        frame_count = batch_size * pred_x.shape[1]
        masks = parameter_masks(labels)
        targets = safe_native_targets(labels, masks)
        bbx, K = labels["bbx_xys"].float()[:, None], labels["K_fullimg"].float()[:, None]
        params = decode_local_motion(pred_x, pred_cam, bbx, K, endecoder)
        zero = pred_x[..., :646].sum() * 0 + pred_cam.sum() * 0
        result = {
            "simple_loss": normalized_local_parameter_loss(pred_x, targets, masks, endecoder),
            "cr_j3d_loss": zero,
            "cr_vert_loss": zero,
            "transl_c_loss": zero,
            "verts2d_loss": zero,
        }
        geometry_rows = masks["geometry"]
        camera_depth_ok = targets["transl"][:, 0, 2] > 0.3
        if bool(geometry_rows.any()):
            gt_params, pred_params = _take(targets, geometry_rows), _take(params, geometry_rows)
            # GT stays[B,1,...] and broadcasts to each repeated prediction.
            # Keep prediction[B,T,...] so SOMA prepares identity only once per B.
            with torch.no_grad():
                gt = endecoder.body_model(
                    **gt_params, apply_correctives=False, repose_to_bind_pose=True
                )
            pred = endecoder.body_model(
                **pred_params, apply_correctives=False, repose_to_bind_pose=True
            )
            if pred["vertices"].shape[-2] != 612 or pred["joints"].shape[-2] != 77:
                raise ValueError("Native image losses must retain A's XLO612/public77 geometry")
            pred_root, gt_root = pred["joints"][..., :1, :], gt["joints"][..., :1, :]
            if "native_joint_confidence" in labels:
                confidence3d = labels["native_joint_confidence"][geometry_rows, None].float()
            else:
                confidence3d = labels["keypoints_3d"][geometry_rows, None, :, 3].float()
            j3d_error = ((pred["joints"] - pred_root) - (gt["joints"] - gt_root)).square()
            result["cr_j3d_loss"] = (j3d_error * confidence3d[..., None]).sum() / (
                frame_count * 77 * 3
            )
            verts_error = ((pred["vertices"] - pred_root) - (gt["vertices"] - gt_root)).square()
            result["cr_vert_loss"] = verts_error.sum() / (
                frame_count * pred["vertices"].shape[-2] * 3
            )
            camera_depth_ok[geometry_rows] = gt["joints"][:, 0, :, 2].amin(-1) > 0.3
            pred_uv, pred_ok = project_valid_bi01(
                pred["vertices"], bbx[geometry_rows], K[geometry_rows]
            )
            gt_uv, gt_ok = project_valid_bi01(gt["vertices"], bbx[geometry_rows], K[geometry_rows])
            inside = ((gt_uv > 0) & (gt_uv < 1)).all(-1)
            vertex_valid = pred_ok & gt_ok & inside & masks["camera"][geometry_rows, None, None]
            result["verts2d_loss"] = (
                (pred_uv - gt_uv).square() * vertex_valid[..., None]
            ).sum() / (frame_count * pred_uv.shape[-2] * 2)
        gt_transl = targets["transl"].clone()
        usable_translation = masks["camera"][:, None] & (gt_transl[..., 2] > 0.3)
        gt_transl[..., 2] = torch.where(
            usable_translation, gt_transl[..., 2], torch.ones_like(gt_transl[..., 2])
        )
        target_cam = get_a_pred_cam(gt_transl, bbx, K)
        camera_valid = masks["camera"][:, None] & camera_depth_ok[:, None]
        camera_valid = camera_valid & (target_cam[..., 0] > 0.3) & (target_cam[..., 0] < 5.0)
        camera_valid = camera_valid & (target_cam[..., 1:].abs() < 3.0).all(-1)
        result["transl_c_loss"] = (
            (pred_cam - target_cam).square() * camera_valid[..., None]
        ).mean()

        # Prediction is differentiable; the existing no-grad S70 teacher is not
        # used. MID and native XLO both receive explicit matched pose policies.
        mid = self.mid_body(**params, apply_correctives=False, repose_to_bind_pose=True)
        if mid["vertices"].shape[-2] != 18056:
            raise ValueError("MID prediction path did not return18056vertices")
        with _matmul_without_tf32(mid["vertices"].device):
            points = torch.einsum(
                "jv,blvc->bljc", self.m_soma30.to(mid["vertices"]), mid["vertices"]
            )
        result["j2d_loss"], valid = sapiens30_l1(points, labels, bbx, K)
        names = {
            "simple": "simple_loss",
            "cr_j3d": "cr_j3d_loss",
            "cr_verts": "cr_vert_loss",
            "transl_c": "transl_c_loss",
            "j2d": "j2d_loss",
            "verts2d": "verts2d_loss",
        }
        result["loss"] = sum(self.weights[name] * result[key] for name, key in names.items())
        result["valid_sapiens_points"] = valid.sum().detach()
        result["valid_native_rows"] = geometry_rows.sum().detach()
        result["pred_soma_params_incam"] = params
        return result


@contextmanager
def _matmul_without_tf32(device):
    """Keep the deployed landmark projection in full float32 precision."""
    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
