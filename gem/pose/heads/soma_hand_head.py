# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA hand parameter regression with cross-attention.

The head supports learnable pose, shape and camera initializations, an
initialization-to-query projection, two-layer regression heads and per-layer
intermediate predictions.

The MHR hand layout contains five identity coefficients and 27 scale values:
one global scale followed by 26 local scales. The geometry adapter separates
the global and local scales when calling the SOMA hand runtime.
"""

import math

import einops
import torch
import torch.nn as nn

from gem.common.geometry.rotations import rot6d_to_rotmat

from ..conditioning.camera_depth import canonical_camera_depth_mode
from .output import PoseHeadOutput
from .transformer_decoder import TransformerDecoder


def _make_head(
    in_dim: int, out_dim: int, two_layer: bool, init_xavier: bool, init_gain: float = 0.01
) -> nn.Module:
    """Single Linear or 2-layer FFN head. Final layer starts near zero."""
    if two_layer:
        head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
        if init_xavier:
            nn.init.xavier_uniform_(head[2].weight, gain=init_gain)
            nn.init.zeros_(head[2].bias)
    else:
        head = nn.Linear(in_dim, out_dim)
        if init_xavier:
            nn.init.xavier_uniform_(head.weight, gain=init_gain)
            nn.init.zeros_(head.bias)
    return head


class SOMAHandHeadV2(nn.Module):
    """v2 cross-attention head for canonical left-hand SOMA parameter regression.

    Returns PoseHeadOutput. The global query is always ``token``; optional
    keypoint queries are returned separately in ``joint_tokens``.

    Outputs:
        pred_hand_params:
            ``wrist_orient``: (B, 1, 3, 3)
            ``hand_pose``: (B, 24, 3, 3)
            ``identity_coeffs``: (B, K)
            ``scale_params``: (B, S), with S=27 for MHR
        pred_cam: (B, 3) internal camera parameter
            ``[s|log_Z|log_lambda, dx, dy]``. Here ``s`` is the crop
            weak-perspective scale. Under the calibrated full-perspective
            camera contract, ``Z = 2 f_x / (s b)`` and
            ``lambda = b Z / f_x = 2 / s``, where ``b`` is the square crop
            size in full-image pixels. The lateral ``[dx, dy]`` channels keep
            their metric-offset semantics in every depth parameterization.
        pred_hand_params_list: per-layer/iter predictions stacked on dim 0
    """

    def __init__(
        self,
        num_finger_joints: int = 24,
        num_identity_coeffs: int = 5,
        num_scale_params: int = 27,
        joint_rep_dim: int = 6,
        transformer_dim: int = 1024,
        transformer_depth: int = 6,
        transformer_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_mlp_dim: int = 1024,
        transformer_dropout: float = 0.0,
        transformer_emb_dropout: float = 0.0,
        transformer_context_dim: int = 1280,
        ief_iters: int = 1,
        init_xavier: bool = True,
        cliff_cond_dim: int = 0,
        learnable_init: bool = True,
        use_init_to_token: bool = True,
        two_layer_heads: bool = True,
        per_layer_predictions: bool = True,
        init_global_scale: float = 1.0,
        init_cam_s: float = 1.0,
        init_cam_z: float = 1.0,
        init_camera_scalar: float | None = None,
        camera_scalar_std: float = 1.0,
        camera_scalar_delta_scaled: bool | None = None,
        camera_depth_mode: str = "s",
        camera_scale_mode: str = "hard_clamp",
        num_kp2d_queries: int = 0,
        ray_cond_dim: int = 0,
    ):
        super().__init__()
        if ief_iters != 1 and per_layer_predictions:
            raise ValueError(
                "v2: per_layer_predictions=True replaces the IEF outer loop. Set ief_iters=1."
            )
        if num_scale_params < 1:
            raise ValueError("Hand scale_params must include global_scale as dimension 0.")
        if num_kp2d_queries < 0:
            raise ValueError("num_kp2d_queries must be non-negative.")
        if num_kp2d_queries > 0 and transformer_emb_dropout != 0:
            raise ValueError(
                "Merged ordered kp2d queries require "
                "transformer_emb_dropout=0; token removal would change the "
                "SOMA joint-to-query correspondence."
            )

        self.num_finger_joints = num_finger_joints
        self.num_identity_coeffs = num_identity_coeffs
        self.num_scale_params = num_scale_params
        self.joint_rep_dim = joint_rep_dim
        self.ief_iters = ief_iters
        self.cliff_cond_dim = cliff_cond_dim
        self.use_init_to_token = use_init_to_token
        self.two_layer_heads = two_layer_heads
        self.per_layer_predictions = per_layer_predictions
        self.transformer_depth = transformer_depth
        self.num_kp2d_queries = int(num_kp2d_queries)
        self.ray_cond_dim = int(ray_cond_dim)
        if self.ray_cond_dim < 0:
            raise ValueError(f"ray_cond_dim must be non-negative, got {self.ray_cond_dim}")
        self.camera_depth_mode = canonical_camera_depth_mode(camera_depth_mode)
        self.camera_scale_mode = str(camera_scale_mode).lower()
        if self.camera_scale_mode in {"no_clamp", "unclamped"}:
            self.camera_scale_mode = "none"
        if self.camera_scale_mode not in {
            "hard_clamp",
            "ste_clamp",
            "none",
        }:
            raise ValueError(
                f"camera_scale_mode={self.camera_scale_mode!r} must be "
                "'hard_clamp', 'ste_clamp', or 'none'"
            )
        if camera_scalar_delta_scaled is None:
            # Preserve the historical contracts: log-depth heads predict a
            # normalized decoder residual, while legacy s heads add the raw
            # decoder delta directly.
            camera_scalar_delta_scaled = self.camera_depth_mode != "s"
        self.camera_scalar_delta_scaled = bool(camera_scalar_delta_scaled)
        self.camera_scalar_std = float(camera_scalar_std)
        if not math.isfinite(self.camera_scalar_std) or self.camera_scalar_std <= 0:
            raise ValueError("camera_scalar_std must be finite and positive")
        if init_cam_s <= 0 or not math.isfinite(float(init_cam_s)):
            raise ValueError("init_cam_s must be finite and positive")
        if init_cam_z <= 0 or not math.isfinite(float(init_cam_z)):
            raise ValueError("init_cam_z must be finite and positive")
        if init_camera_scalar is None:
            if self.camera_depth_mode == "s":
                camera_scalar = float(init_cam_s)
            elif self.camera_depth_mode == "log_Z":
                camera_scalar = math.log(float(init_cam_z))
            else:
                # lambda = 2 / s. This preserves the legacy step-zero scale
                # prior when switching only the parameterization.
                camera_scalar = math.log(2.0 / float(init_cam_s))
        else:
            camera_scalar = float(init_camera_scalar)
            if not math.isfinite(camera_scalar):
                raise ValueError("init_camera_scalar must be finite")

        num_hand_joints = num_finger_joints + 1
        self.num_hand_joints = num_hand_joints
        self.npose = joint_rep_dim * num_hand_joints
        self.nshape = num_identity_coeffs + num_scale_params

        init_pose = torch.zeros(1, self.npose)
        init_shape = torch.zeros(1, self.nshape)
        init_shape[0, num_identity_coeffs] = float(init_global_scale)
        init_cam = torch.zeros(1, 3)
        init_cam[0, 0] = camera_scalar

        if learnable_init:
            self.init_pose = nn.Parameter(init_pose)
            self.init_shape = nn.Parameter(init_shape)
            self.init_cam = nn.Parameter(init_cam)
        else:
            self.register_buffer("init_pose", init_pose)
            self.register_buffer("init_shape", init_shape)
            self.register_buffer("init_cam", init_cam)

        if use_init_to_token:
            init_dim = self.npose + self.nshape + 3 + cliff_cond_dim
            self.init_to_token = nn.Linear(init_dim, transformer_dim)
            token_dim = transformer_dim
            skip_token_embedding = True
        else:
            self.init_to_token = None
            token_dim = self.npose + self.nshape + 3 + cliff_cond_dim
            skip_token_embedding = False

        # Keep the transformer's learned positional embedding at its historical
        # single-token shape.  It broadcasts over optional joint queries, whose
        # learned embeddings already provide joint identity.  Besides avoiding
        # redundant position parameters, this lets existing one-token
        # checkpoints load into the merged topology without a shape mismatch.
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=token_dim,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=transformer_dropout,
            emb_dropout=transformer_emb_dropout,
            context_dim=transformer_context_dim,
            skip_token_embedding=skip_token_embedding,
            attention_implementation="eager",
        )
        self.decpose = _make_head(transformer_dim, self.npose, two_layer_heads, init_xavier)
        self.decshape = _make_head(transformer_dim, self.nshape, two_layer_heads, init_xavier)
        self.deccam = _make_head(transformer_dim, 3, two_layer_heads, init_xavier)

        if self.ray_cond_dim > 0:
            self.ray_mlp = nn.Sequential(
                nn.Linear(self.ray_cond_dim, 512),
                nn.GELU(),
                nn.Linear(512, transformer_context_dim),
            )
            nn.init.zeros_(self.ray_mlp[-1].weight)
            nn.init.zeros_(self.ray_mlp[-1].bias)
        else:
            self.ray_mlp = None

        # Construct variant-only parameters after every shared module and
        # restore the caller's RNG state afterwards.  A fixed-seed
        # merged-vs-separate comparison therefore starts all shared parameters
        # bit-identically.
        if self.num_kp2d_queries > 0:
            with torch.random.fork_rng(devices=[]):
                self.kp2d_queries = nn.Embedding(
                    self.num_kp2d_queries,
                    token_dim,
                )
                nn.init.normal_(self.kp2d_queries.weight, std=0.02)
        else:
            self.kp2d_queries = None

    def _build_token(self, batch_size: int, device, dtype, cliff_cond):
        if self.use_init_to_token:
            parts = [
                self.init_pose.expand(batch_size, -1).to(dtype=dtype),
                self.init_shape.expand(batch_size, -1).to(dtype=dtype),
                self.init_cam.expand(batch_size, -1).to(dtype=dtype),
            ]
            if self.cliff_cond_dim > 0:
                cc = (
                    cliff_cond
                    if cliff_cond is not None
                    else torch.zeros(batch_size, self.cliff_cond_dim, device=device, dtype=dtype)
                )
                parts.append(cc.to(dtype=dtype))
            global_token = self.init_to_token(torch.cat(parts, dim=-1))[:, None, :]
            return self._append_kp2d_queries(
                global_token,
                batch_size,
                dtype,
            )

        parts = [
            self.init_pose.expand(batch_size, -1).to(dtype=dtype),
            self.init_shape.expand(batch_size, -1).to(dtype=dtype),
            self.init_cam.expand(batch_size, -1).to(dtype=dtype),
        ]
        if self.cliff_cond_dim > 0:
            cc = (
                cliff_cond
                if cliff_cond is not None
                else torch.zeros(batch_size, self.cliff_cond_dim, device=device, dtype=dtype)
            )
            parts.append(cc.to(dtype=dtype))
        global_token = torch.cat(parts, dim=1)[:, None, :]
        return self._append_kp2d_queries(
            global_token,
            batch_size,
            dtype,
        )

    def _append_kp2d_queries(
        self,
        global_token: torch.Tensor,
        batch_size: int,
        dtype,
    ) -> torch.Tensor:
        """Append optional per-joint 2D queries after the global hand token."""
        if self.kp2d_queries is None:
            return global_token
        joint_queries = self.kp2d_queries.weight.to(dtype=dtype)
        joint_queries = joint_queries.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([global_token, joint_queries], dim=1)

    def _decode_single(self, token_out, batch_size: int):
        pred_pose = self.decpose(token_out) + self.init_pose.expand(batch_size, -1)
        pred_shape = self.decshape(token_out) + self.init_shape.expand(batch_size, -1)
        camera_delta = self.deccam(token_out)
        if self.camera_scalar_delta_scaled:
            camera_delta = torch.cat(
                [
                    camera_delta[..., 0:1] * self.camera_scalar_std,
                    camera_delta[..., 1:],
                ],
                dim=-1,
            )
        pred_cam = camera_delta + self.init_cam.expand(batch_size, -1)
        return pred_pose, pred_shape, pred_cam

    def _params_from_pose_shape(self, pred_pose, pred_shape, batch_size: int):
        pred_pose_6d = pred_pose.view(
            batch_size,
            self.num_hand_joints,
            self.joint_rep_dim,
        )
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose).view(
            batch_size,
            self.num_hand_joints,
            3,
            3,
        )
        return {
            "wrist_orient": pred_pose_rotmat[:, :1],
            "hand_pose": pred_pose_rotmat[:, 1:],
            "identity_coeffs": pred_shape[:, : self.num_identity_coeffs],
            "scale_params": pred_shape[:, self.num_identity_coeffs :],
            "pose_6d": pred_pose_6d,
        }

    def apply_camera_scale_constraint(
        self,
        camera_parameter: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured final physical-scale constraint.

        Temporal camera refinement calls this again after adding its residual,
        so every final ``s`` camera obeys the same contract as the image head.
        Log-depth scalars and explicitly unconstrained ``s`` cameras pass
        through unchanged.
        """
        if self.camera_depth_mode != "s" or self.camera_scale_mode == "none":
            return camera_parameter

        raw_cam_s = camera_parameter[..., 0:1]
        clamped_cam_s = raw_cam_s.clamp(min=0.25)
        if self.camera_scale_mode == "ste_clamp":
            # Keep the camera floor representable when the raw scale is large
            # and negative, with the same identity gradient as the body head.
            cam_s = clamped_cam_s.detach() + (raw_cam_s - raw_cam_s.detach())
        else:
            cam_s = clamped_cam_s
        return torch.cat(
            [cam_s, camera_parameter[..., 1:]],
            dim=-1,
        )

    def forward(self, x, cliff_cond=None, ray_feats=None, **kwargs):
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        feat = einops.rearrange(x, "b c h w -> b (h w) c")
        if self.ray_mlp is not None:
            expected_ray_shape = (batch_size, feat.shape[1], self.ray_cond_dim)
            if ray_feats is None or tuple(ray_feats.shape) != expected_ray_shape:
                actual_shape = None if ray_feats is None else tuple(ray_feats.shape)
                raise ValueError(
                    f"ray_feats must have shape {expected_ray_shape}, got {actual_shape}"
                )
            ray_embedding = self.ray_mlp(
                ray_feats.to(
                    device=device,
                    dtype=self.ray_mlp[0].weight.dtype,
                )
            )
            feat = feat + ray_embedding.to(dtype=feat.dtype)

        if self.per_layer_predictions:
            token = self._build_token(batch_size, device, dtype, cliff_cond)
            intermediates = self.transformer(
                token,
                context=feat,
                return_intermediate=True,
            )
            pred_pose_list, pred_shape_list, pred_cam_list = [], [], []
            for h in intermediates:
                p, s, c = self._decode_single(h[:, 0], batch_size)
                pred_pose_list.append(p)
                pred_shape_list.append(s)
                pred_cam_list.append(c)
            pred_pose = pred_pose_list[-1]
            pred_shape = pred_shape_list[-1]
            pred_cam = pred_cam_list[-1]
            kp2d_joint_tokens = intermediates[-1][:, 1:] if self.num_kp2d_queries > 0 else None
        else:
            pred_pose = self.init_pose.expand(batch_size, -1)
            pred_shape = self.init_shape.expand(batch_size, -1)
            pred_cam = self.init_cam.expand(batch_size, -1)
            pred_pose_list, pred_shape_list, pred_cam_list = [], [], []
            for _ in range(self.ief_iters):
                token = self._build_token(batch_size, device, dtype, cliff_cond)
                h = self.transformer(token, context=feat)
                global_h = h[:, 0]
                pred_pose = self.decpose(global_h) + pred_pose
                pred_shape = self.decshape(global_h) + pred_shape
                camera_delta = self.deccam(global_h)
                if self.camera_scalar_delta_scaled:
                    camera_delta = torch.cat(
                        [
                            camera_delta[..., 0:1] * self.camera_scalar_std,
                            camera_delta[..., 1:],
                        ],
                        dim=-1,
                    )
                pred_cam = camera_delta + pred_cam
                pred_pose_list.append(pred_pose)
                pred_shape_list.append(pred_shape)
                pred_cam_list.append(pred_cam)
            kp2d_joint_tokens = h[:, 1:] if self.num_kp2d_queries > 0 else None

        # Constrain only the final prediction; intermediate cameras remain
        # raw, matching the GEP2 head contract.
        pred_cam = self.apply_camera_scale_constraint(pred_cam)
        pred_hand_params = self._params_from_pose_shape(pred_pose, pred_shape, batch_size)

        def _layer_params(pp, ps):
            params = self._params_from_pose_shape(pp, ps, batch_size)
            return params

        layer_params = [_layer_params(pp, ps) for pp, ps in zip(pred_pose_list, pred_shape_list)]
        pred_hand_params_list = {
            "wrist_orient": torch.cat([p["wrist_orient"] for p in layer_params], dim=0),
            "hand_pose": torch.cat([p["hand_pose"] for p in layer_params], dim=0),
            "identity_coeffs": torch.cat([p["identity_coeffs"] for p in layer_params], dim=0),
            "scale_params": torch.cat([p["scale_params"] for p in layer_params], dim=0),
            "cam": torch.cat(pred_cam_list, dim=0),
        }

        final_token = intermediates[-1][:, 0] if self.per_layer_predictions else global_h
        return PoseHeadOutput(
            pred_hand_params,
            pred_cam,
            pred_hand_params_list,
            final_token,
            joint_tokens=kp2d_joint_tokens,
        )
