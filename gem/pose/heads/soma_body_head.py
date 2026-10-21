# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA body parameter regression with cross-attention.

The decoder supports learnable pose, shape and camera initializations.
Parameter heads can predict after each decoder layer and return named parameters,
camera values, auxiliary predictions and the final body query token without
running forward kinematics. An optional second query predicts global-scale
uncertainty through a Laplace or Gaussian head.

Scale prediction uses PCA coefficients expanded to scale parameters,
or direct native scale parameters.
``fixed_global_scale`` fixes the global component; ``scale_parameterization="local56"``
returns 56 local scales and a separate fixed global-scale field.

Per-layer predictions stack on dim 0 as ``depth * B``. ``final_only`` returns
only the final ``B`` rows for training without auxiliary-layer losses.
"""

from collections.abc import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gem.common.geometry.rotations import aa_to_rotmat, rot6d_to_rotmat

from .output import PoseHeadOutput
from .transformer_decoder import TransformerDecoder

FINGER_BODY_POSE_INDICES_76 = list(range(14, 38)) + list(range(42, 66))

_NON_FINGER_BODY_POSE_INDICES_76 = [i for i in range(76) if i not in FINGER_BODY_POSE_INDICES_76]


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"head_activation={name!r} must be 'gelu' or 'relu'")


def _normalize_head_init(head_init, init_xavier: bool) -> str:
    if head_init is None or str(head_init).strip() == "":
        return "xavier" if init_xavier else "default"
    mode = str(head_init).lower()
    if mode in {"xavier", "xavier_small", "legacy"}:
        return "xavier"
    if mode in {"default", "sam", "sam_default", "torch_default", "pytorch_default"}:
        return "default"
    raise ValueError(
        f"head_init={mode!r} must be one of 'xavier'/'legacy' or " "'default'/'sam_default'"
    )


def _make_head(
    in_dim: int,
    out_dim: int,
    two_layer: bool,
    init_xavier: bool,
    init_gain: float = 0.01,
    activation: str = "gelu",
    head_init=None,
) -> nn.Module:
    """Single Linear or 2-layer FFN head.

    The legacy init applies final-layer Xavier(gain) and zero bias so step-0
    outputs stay near the learned init params. SAM3DB-style init leaves Linear
    layers at PyTorch defaults.
    """
    init_mode = _normalize_head_init(head_init, init_xavier)
    if two_layer:
        head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            _make_activation(activation),
            nn.Linear(in_dim, out_dim),
        )
        if init_mode == "xavier":
            nn.init.xavier_uniform_(head[2].weight, gain=init_gain)
            nn.init.zeros_(head[2].bias)
    else:
        head = nn.Linear(in_dim, out_dim)
        if init_mode == "xavier":
            nn.init.xavier_uniform_(head.weight, gain=init_gain)
            nn.init.zeros_(head.bias)
    return head


class SOMABodyHeadV2(nn.Module):
    """SOMA cross-attention head with learnable initialization and per-layer predictions.

    Returns a :class:`PoseHeadOutput` with:
        parameters: native SOMA final-layer predictions.
        pred_cam: (B, 3) camera vector. Its first channel is clamped only when
            ``camera_scale_mode`` is enabled.
        auxiliary: dict whose tensors are layer-wise concatenations
            (dim-0 = depth × B) for auxiliary supervision, or one final B slab
            when ``final_only`` is requested.
        token: (B, transformer_dim) final decoder body query.
    """

    def __init__(
        self,
        # === v1 args (same names, same defaults) ===
        num_body_joints: int = 76,
        num_identity_coeffs: int = 45,
        num_scale_params: int = 69,
        scale_comps_dim: int = 28,
        joint_rep_dim: int = 6,
        soma_data_dir: str = "inputs/soma_data",
        transformer_dim: int = 1024,
        transformer_depth: int = 6,
        transformer_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_mlp_dim: int = 2048,
        transformer_dropout: float = 0.0,
        transformer_emb_dropout: float = 0.0,
        transformer_context_dim: int = 1280,
        ief_iters: int = 1,
        init_xavier: bool = True,
        mean_params: str = "",
        input_is_mean_shape: bool = False,
        cliff_cond_dim: int = 0,
        ray_cond_dim: int = 0,
        skip_finger_joints: bool = False,
        mean_finger_pose_path: str = "",
        # === v2 knobs ===
        learnable_init: bool = True,
        use_init_to_token: bool = True,
        two_layer_heads: bool = True,
        per_layer_predictions: bool = True,
        warm_init_from_ckpt: str = "",
        warm_init_transfer_pose_scale: bool = False,
        warm_init_transfer_camera: bool = False,
        camera_scale_mode: str = "hard_clamp",
        head_activation: str = "gelu",
        head_init=None,
        scale_param_mode: str = "pca",
        direct_scale_init="ones",
        global_scale_b_query: bool = False,
        global_scale_sigma_query: bool = False,
        fixed_global_scale: float | None = None,
        camera_first_init: float | None = None,
        scale_parameterization: str = "packed",
    ):
        super().__init__()
        if use_init_to_token and input_is_mean_shape:
            raise ValueError(
                "v2: use_init_to_token=True is mutually exclusive with input_is_mean_shape=True "
                "(both routes inject init params into the token; pick one)."
            )
        if ief_iters != 1 and per_layer_predictions:
            raise ValueError(
                "v2: per_layer_predictions=True replaces the IEF outer loop. Set ief_iters=1."
            )
        self.joint_rep_dim = joint_rep_dim
        self.num_body_joints = num_body_joints
        self.num_identity_coeffs = num_identity_coeffs
        self.num_scale_params = num_scale_params
        self.scale_comps_dim = scale_comps_dim
        self.scale_param_mode = str(scale_param_mode)
        self.scale_parameterization = str(scale_parameterization)
        if self.scale_parameterization not in {"packed", "local56"}:
            raise ValueError(f"Unsupported scale_parameterization={scale_parameterization!r}")
        if self.scale_parameterization == "local56" and (
            num_identity_coeffs != 128
            or num_scale_params != 56
            or scale_param_mode != "direct"
            or fixed_global_scale != 1.0
        ):
            raise ValueError(
                "local56 requires identity128, direct scale56 and fixed_global_scale=1"
            )
        if self.scale_param_mode not in {"pca", "direct"}:
            raise ValueError(
                f"scale_param_mode={self.scale_param_mode!r} must be 'pca' or 'direct'"
            )
        self.fixed_global_scale = None if fixed_global_scale is None else float(fixed_global_scale)
        if self.fixed_global_scale is not None:
            if not np.isfinite(self.fixed_global_scale) or self.fixed_global_scale <= 0:
                raise ValueError("fixed_global_scale must be finite and positive")
            if self.num_scale_params < 1:
                raise ValueError(
                    "fixed_global_scale requires num_scale_params >= 1 so the "
                    "returned scale vector has a global-scale slot"
                )
            if self.scale_param_mode == "direct" and self.num_scale_params < 2:
                raise ValueError(
                    "fixed_global_scale with scale_param_mode='direct' requires "
                    "num_scale_params >= 2 (one fixed global and at least one "
                    "learned local scale)"
                )
            if self.scale_param_mode == "pca" and scale_comps_dim <= 0:
                raise ValueError(
                    "fixed_global_scale with scale_param_mode='pca' requires " "scale_comps_dim > 0"
                )
        self.scale_latent_dim = (
            scale_comps_dim
            if self.scale_param_mode == "pca"
            else num_scale_params
            - (
                1
                if self.fixed_global_scale is not None and self.scale_parameterization == "packed"
                else 0
            )
        )
        self.ief_iters = ief_iters
        self.input_is_mean_shape = input_is_mean_shape
        self.cliff_cond_dim = cliff_cond_dim
        self.ray_cond_dim = int(ray_cond_dim)
        if self.ray_cond_dim < 0:
            raise ValueError(f"ray_cond_dim must be non-negative, got {self.ray_cond_dim}")
        # Preserve the no-scale sentinel for configurations without a scale branch:
        # scale_comps_dim=0 means no PCA scale branch even if an older/minimal
        # config omitted num_scale_params and inherited its legacy default.
        self.has_scale = self.scale_latent_dim > 0
        self.skip_finger_joints = skip_finger_joints
        self.use_init_to_token = use_init_to_token
        self.two_layer_heads = two_layer_heads
        self.per_layer_predictions = per_layer_predictions
        self.transformer_depth = transformer_depth
        self.global_scale_b_query = bool(global_scale_b_query)
        self.global_scale_sigma_query = bool(global_scale_sigma_query)
        if self.global_scale_b_query and self.global_scale_sigma_query:
            raise ValueError(
                "global_scale_b_query and global_scale_sigma_query are mutually exclusive"
            )
        if self.fixed_global_scale is not None and (
            self.global_scale_b_query or self.global_scale_sigma_query
        ):
            raise ValueError(
                "global-scale uncertainty queries are invalid when fixed_global_scale is set"
            )
        if (self.global_scale_b_query or self.global_scale_sigma_query) and not self.has_scale:
            raise ValueError("global-scale uncertainty queries require scale parameters")
        self.camera_scale_mode = str(camera_scale_mode).lower()
        if self.camera_scale_mode in {"no_clamp", "unclamped"}:
            self.camera_scale_mode = "none"
        if self.camera_scale_mode not in {"hard_clamp", "ste_clamp", "none"}:
            raise ValueError(
                f"camera_scale_mode={self.camera_scale_mode!r} must be "
                "'hard_clamp', 'ste_clamp', or 'none'"
            )
        self.camera_first_init = None if camera_first_init is None else float(camera_first_init)
        if self.camera_first_init is not None and not np.isfinite(self.camera_first_init):
            raise ValueError("camera_first_init must be finite")
        self.head_activation = str(head_activation).lower()
        self.head_init = _normalize_head_init(head_init, init_xavier)

        if skip_finger_joints:
            self._predicted_body_joints = 76 - len(FINGER_BODY_POSE_INDICES_76)  # 28
            self.register_buffer(
                "_non_finger_indices",
                torch.tensor(_NON_FINGER_BODY_POSE_INDICES_76, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_finger_indices",
                torch.tensor(FINGER_BODY_POSE_INDICES_76, dtype=torch.long),
                persistent=False,
            )
            if not mean_finger_pose_path:
                raise ValueError(
                    "skip_finger_joints=True requires `mean_finger_pose_path` "
                    "(an .npy with shape (48, 3) axis-angle finger pose)."
                )
            mean_finger_aa = torch.from_numpy(np.load(mean_finger_pose_path).astype(np.float32))
            assert mean_finger_aa.shape == (
                48,
                3,
            ), f"mean_finger_pose must be (48, 3) axis-angle, got {tuple(mean_finger_aa.shape)}"
            self.register_buffer(
                "_mean_finger_pose_rotmat",
                aa_to_rotmat(mean_finger_aa),  # (48, 3, 3)
                persistent=False,
            )
        else:
            self._predicted_body_joints = num_body_joints

        npose = joint_rep_dim * (self._predicted_body_joints + 1)
        self.npose = npose
        nshape = num_identity_coeffs + self.scale_latent_dim
        self.nshape = nshape

        # ======= Initial param values (loaded same way as v1) =======
        if mean_params:
            mean_data = np.load(mean_params)
            init_pose = torch.from_numpy(mean_data["pose"].astype(np.float32)).unsqueeze(0)
            init_shape = torch.from_numpy(mean_data["shape"].astype(np.float32)).unsqueeze(0)
            init_cam = torch.from_numpy(mean_data["cam"].astype(np.float32)).unsqueeze(0)
            if skip_finger_joints:
                full_npose = joint_rep_dim * (num_body_joints + 1)
                assert (
                    init_pose.shape[-1] == full_npose
                ), f"mean_params pose dim {init_pose.shape[-1]} != {full_npose} (full)"
                init_pose_full = init_pose.view(1, num_body_joints + 1, joint_rep_dim)
                keep_idx = torch.tensor(
                    [0] + [i + 1 for i in _NON_FINGER_BODY_POSE_INDICES_76],
                    dtype=torch.long,
                )
                init_pose = init_pose_full[:, keep_idx, :].reshape(1, -1)
            assert (
                init_pose.shape[-1] == npose
            ), f"mean_params pose dim {init_pose.shape[-1]} != {npose}"
            assert (
                init_shape.shape[-1] == nshape
            ), f"mean_params shape dim {init_shape.shape[-1]} != {nshape}"
        else:
            init_pose = torch.zeros(1, npose)
            init_shape = torch.zeros(1, nshape)
            if self.has_scale and self.scale_param_mode == "direct":
                init_shape[:, self.num_identity_coeffs :] = self._make_direct_scale_init(
                    direct_scale_init,
                )
            init_cam = torch.zeros(1, 3)
            init_cam[0, 0] = 1.0

        if self.camera_first_init is not None:
            init_cam[:, 0] = self.camera_first_init

        # ======= Learnable vs frozen init =======
        if learnable_init:
            self.init_pose = nn.Parameter(init_pose)
            self.init_shape = nn.Parameter(init_shape)
            self.init_cam = nn.Parameter(init_cam)
        else:
            self.register_buffer("init_pose", init_pose)
            self.register_buffer("init_shape", init_shape)
            self.register_buffer("init_cam", init_cam)

        # ======= Transformer + token construction =======
        if use_init_to_token:
            init_dim = npose + nshape + 3 + cliff_cond_dim
            self.init_to_token = nn.Linear(init_dim, transformer_dim)
            base_token_dim = transformer_dim
            skip_token_embed = True
        else:
            base_token_dim = (npose + nshape + 3) if input_is_mean_shape else 1
            base_token_dim = base_token_dim + cliff_cond_dim
            skip_token_embed = False
            self.init_to_token = None

        self.transformer = TransformerDecoder(
            # Build the shared one-token module first. The optional second
            # positional embedding is appended after all shared heads are
            # initialized so enabling uncertainty does not shift their seeded
            # RNG state.
            num_tokens=1,
            token_dim=base_token_dim,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=transformer_dropout,
            emb_dropout=transformer_emb_dropout,
            context_dim=transformer_context_dim,
            skip_token_embedding=skip_token_embed,
        )

        # ======= Regression heads =======
        self.decpose = _make_head(
            transformer_dim,
            npose,
            two_layer_heads,
            init_xavier,
            activation=self.head_activation,
            head_init=self.head_init,
        )
        self.decshape = _make_head(
            transformer_dim,
            nshape,
            two_layer_heads,
            init_xavier,
            activation=self.head_activation,
            head_init=self.head_init,
        )
        self.deccam = _make_head(
            transformer_dim,
            3,
            two_layer_heads,
            init_xavier,
            activation=self.head_activation,
            head_init=self.head_init,
        )
        if self.global_scale_b_query:
            first_pos = self.transformer.pos_embedding
            extra_pos = torch.randn_like(first_pos[:, :1])
            self.transformer.pos_embedding = nn.Parameter(
                torch.cat([first_pos.detach(), extra_pos], dim=1)
            )
            self.b_query = nn.Parameter(torch.empty(1, 1, base_token_dim))
            nn.init.normal_(self.b_query, std=0.02)
            # Match Argmax RLETR-occ's uncertainty readout: one plain Linear
            # with a small-gain Xavier initialization, followed by sigmoid.
            self.decb = nn.Linear(transformer_dim, 1)
            nn.init.xavier_uniform_(self.decb.weight, gain=0.01)
            nn.init.zeros_(self.decb.bias)
            self.register_parameter("sigma_query", None)
            self.decsigma = None
        elif self.global_scale_sigma_query:
            first_pos = self.transformer.pos_embedding
            extra_pos = torch.randn_like(first_pos[:, :1])
            self.transformer.pos_embedding = nn.Parameter(
                torch.cat([first_pos.detach(), extra_pos], dim=1)
            )
            self.sigma_query = nn.Parameter(torch.empty(1, 1, base_token_dim))
            nn.init.normal_(self.sigma_query, std=0.02)
            self.decsigma = nn.Linear(transformer_dim, 1)
            nn.init.xavier_uniform_(self.decsigma.weight, gain=0.01)
            nn.init.zeros_(self.decsigma.bias)
            self.register_parameter("b_query", None)
            self.decb = None
        else:
            self.register_parameter("b_query", None)
            self.decb = None
            self.register_parameter("sigma_query", None)
            self.decsigma = None

        # ======= Scale expansion =======
        if self.has_scale and self.scale_param_mode == "pca":
            init_scale_mean = torch.load(f"{soma_data_dir}/scale_mean.pth", map_location="cpu")
            init_scale_comps = torch.load(f"{soma_data_dir}/scale_comps.pth", map_location="cpu")
            global_scale_mean = 0.9 if self.fixed_global_scale is None else self.fixed_global_scale
            scale_mean = torch.cat(
                [torch.tensor([global_scale_mean], dtype=init_scale_mean.dtype), init_scale_mean]
            )  # (69,)
            global_scale_row = torch.zeros(scale_comps_dim, 1)
            if self.fixed_global_scale is None:
                global_scale_row[0, 0] = 1
            scale_comps = torch.cat([global_scale_row, init_scale_comps], dim=1)  # (28, 69)
            if tuple(scale_mean.shape) != (self.num_scale_params,):
                raise ValueError(
                    f"PCA scale_mean has shape {tuple(scale_mean.shape)}, "
                    f"expected ({self.num_scale_params},)"
                )
            if tuple(scale_comps.shape) != (self.scale_comps_dim, self.num_scale_params):
                raise ValueError(
                    f"PCA scale_comps has shape {tuple(scale_comps.shape)}, "
                    f"expected ({self.scale_comps_dim}, {self.num_scale_params})"
                )
            self.register_buffer("scale_comps", scale_comps)
            self.register_buffer("scale_mean", scale_mean)

        # ======= Optional warm-start from sam-3d-body checkpoint =======
        if warm_init_from_ckpt:
            self._apply_warm_init(
                warm_init_from_ckpt,
                transfer_pose_scale=warm_init_transfer_pose_scale,
                transfer_camera=warm_init_transfer_camera,
            )

        # Construct this after all pre-existing head modules so enabling Ray PE
        # does not perturb their seeded initialization. The zero output layer
        # makes the new residual branch an exact no-op at initialization.
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

    def _make_direct_scale_init(self, direct_scale_init):
        """Initial value for direct scale prediction.

        SOMA-native scale is represented as [global_scale, local ratios], whose
        identity value is all ones. When global scale is fixed, the learned
        direct vector contains only the local ratios. MHR configs keep the
        default PCA mode and do not enter this path.
        """
        if isinstance(direct_scale_init, str):
            value = direct_scale_init.lower()
            if value == "ones":
                init = torch.ones(self.scale_latent_dim)
            elif value == "zeros":
                init = torch.zeros(self.scale_latent_dim)
            else:
                raise ValueError(
                    f"direct_scale_init={direct_scale_init!r} must be 'ones', 'zeros', "
                    "a scalar, or a sequence"
                )
        elif isinstance(direct_scale_init, int | float):
            init = torch.full((self.scale_latent_dim,), float(direct_scale_init))
        elif isinstance(direct_scale_init, Sequence):
            init = torch.tensor(list(direct_scale_init), dtype=torch.float32)
        else:
            raise TypeError(
                f"Unsupported direct_scale_init type: {type(direct_scale_init).__name__}"
            )
        if tuple(init.shape) != (self.scale_latent_dim,):
            raise ValueError(
                f"direct_scale_init has shape {tuple(init.shape)}, "
                f"expected ({self.scale_latent_dim},)"
            )
        return init.float()

    @torch.no_grad()
    def _apply_warm_init(self, ckpt_path: str, transfer_pose_scale: bool, transfer_camera: bool):
        """Copy learned init values from sam-3d-body checkpoint.

        sam-3d-body's init_pose layout (519D): rot6d(6) + body_cont(260) + shape(45)
        + scale(28) + lhand(54) + rhand(54) + face(72). The 28D scale slice [311:339]
        is in the SAME PCA basis as ours (verified byte-identical).

        Note: empirically these learned values are essentially zero (max abs ~0.0035
        for scale, init_camera ~ -0.01). Flag is for completeness; expect ~no benefit.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if transfer_pose_scale and self.has_scale:
            if self.scale_param_mode != "pca":
                raise ValueError(
                    "warm_init_transfer_pose_scale is only valid for "
                    "scale_param_mode='pca' because the sam-3d-body checkpoint "
                    "stores MHR PCA scale coefficients."
                )
            src = sd["init_pose.weight"][:, 311:339]
            dst_lo = self.num_identity_coeffs
            dst_hi = self.num_identity_coeffs + self.scale_comps_dim
            self.init_shape.data[:, dst_lo:dst_hi].copy_(src)
        if transfer_camera:
            self.init_cam.data.copy_(sd["init_camera.weight"])

    def _splice_mean_fingers(self, predicted_body_pose_rotmat):
        """Insert constant mean finger pose into the 28 predicted non-finger slots."""
        B = predicted_body_pose_rotmat.shape[0]
        device = predicted_body_pose_rotmat.device
        dtype = predicted_body_pose_rotmat.dtype
        full = torch.zeros(B, 76, 3, 3, device=device, dtype=dtype)
        full = full.index_copy(1, self._non_finger_indices, predicted_body_pose_rotmat)
        mean_finger = self._mean_finger_pose_rotmat.unsqueeze(0).expand(B, -1, -1, -1)
        full = full.index_copy(1, self._finger_indices, mean_finger.to(dtype=dtype))
        return full

    def _append_uncertainty_query(self, body_token, batch_size: int, dtype):
        if self.global_scale_b_query:
            query = self.b_query
        elif self.global_scale_sigma_query:
            query = self.sigma_query
        else:
            return body_token
        query = query.expand(batch_size, -1, -1).to(dtype=dtype)
        return torch.cat([body_token, query], dim=1)

    def _build_token(self, batch_size: int, device, dtype, cliff_cond):
        """Build the body token and optional global-scale uncertainty query."""
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
            init_input = torch.cat(parts, dim=-1)
            token = self.init_to_token(init_input)[:, None, :]
            return self._append_uncertainty_query(token, batch_size, dtype)

        if self.input_is_mean_shape:
            base_token = torch.cat(
                [
                    self.init_pose.expand(batch_size, -1).to(dtype=dtype),
                    self.init_shape.expand(batch_size, -1).to(dtype=dtype),
                    self.init_cam.expand(batch_size, -1).to(dtype=dtype),
                ],
                dim=1,
            )
        else:
            base_token = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if self.cliff_cond_dim > 0:
            cc = (
                cliff_cond
                if cliff_cond is not None
                else torch.zeros(batch_size, self.cliff_cond_dim, device=device, dtype=dtype)
            )
            base_token = torch.cat([base_token, cc.to(dtype=dtype)], dim=1)
        body_token = base_token[:, None, :]
        return self._append_uncertainty_query(body_token, batch_size, dtype)

    def _decode_single(self, token_out, batch_size: int):
        """Run regression heads on a (B, transformer_dim) token output and return
        (pred_pose, pred_shape, pred_cam) with init residual added."""
        pred_pose = self.decpose(token_out) + self.init_pose.expand(batch_size, -1)
        pred_shape = self.decshape(token_out) + self.init_shape.expand(batch_size, -1)
        pred_cam = self.deccam(token_out) + self.init_cam.expand(batch_size, -1)
        return pred_pose, pred_shape, pred_cam

    def _decode_global_scale_b(self, token_out):
        """Decode an unclamped sigmoid b in fp32 from the dedicated query."""
        if self.decb is None:
            raise RuntimeError("global-scale b decoder is disabled")
        with torch.amp.autocast(token_out.device.type, enabled=False):
            raw_b = self.decb(token_out.float())
            b = torch.sigmoid(raw_b)
        return raw_b, b

    def _decode_global_scale_sigma(self, token_out):
        """Decode an unclamped Gaussian standard deviation in fp32."""
        if self.decsigma is None:
            raise RuntimeError("global-scale sigma decoder is disabled")
        with torch.amp.autocast(token_out.device.type, enabled=False):
            raw_sigma = self.decsigma(token_out.float())
            sigma = torch.sigmoid(raw_sigma)
            log_sigma = F.logsigmoid(raw_sigma)
        return raw_sigma, sigma, log_sigma

    def _params_from_pose_shape(self, pred_pose, pred_shape, batch_size: int):
        """Convert raw (rot6d pose, shape) to soma_params dict."""
        pred_body_pose_rotmat = rot6d_to_rotmat(pred_pose).view(
            batch_size,
            self._predicted_body_joints + 1,
            3,
            3,
        )
        pred_global_orient_rotmat = pred_body_pose_rotmat[:, [0]]
        pred_body_pose_rotmat = pred_body_pose_rotmat[:, 1:]
        if self.skip_finger_joints:
            pred_body_pose_rotmat = self._splice_mean_fingers(pred_body_pose_rotmat)
        out = {
            "body_pose": pred_body_pose_rotmat,
            "global_orient": pred_global_orient_rotmat,
            "identity_coeffs": pred_shape[:, : self.num_identity_coeffs],
        }
        if self.has_scale:
            out["scale_params"] = self._scale_params_from_shape(pred_shape)
            if self.scale_parameterization == "local56":
                out["global_scale"] = torch.ones_like(pred_shape[:, :1])
        return out

    def _scale_params_from_shape(self, pred_shape):
        scale_raw = pred_shape[:, self.num_identity_coeffs :]
        if self.scale_parameterization == "local56":
            return scale_raw
        if self.scale_param_mode == "pca":
            if self.fixed_global_scale is not None:
                # Do not rely on column 0 of the registered PCA buffers here:
                # an older checkpoint may contain the historical learned
                # global column.  Ignoring it makes the architectural contract
                # survive strict checkpoint loading while all 28 coefficients
                # continue to parameterize the 68 local scales.
                local_scale = scale_raw @ self.scale_comps[:, 1:] + self.scale_mean[1:]
                global_scale = torch.full(
                    (pred_shape.shape[0], 1),
                    self.fixed_global_scale,
                    dtype=local_scale.dtype,
                    device=local_scale.device,
                )
                return torch.cat([global_scale, local_scale], dim=1)
            return scale_raw @ self.scale_comps + self.scale_mean
        if self.scale_param_mode == "direct":
            if self.fixed_global_scale is not None:
                global_scale = torch.full(
                    (pred_shape.shape[0], 1),
                    self.fixed_global_scale,
                    dtype=scale_raw.dtype,
                    device=scale_raw.device,
                )
                return torch.cat([global_scale, scale_raw], dim=1)
            return scale_raw
        raise RuntimeError(f"Unhandled scale_param_mode={self.scale_param_mode!r}")

    def forward(self, x, cliff_cond=None, ray_feats=None, final_only=False, **kwargs):
        """Args:
            x: (B, C, H, W) backbone feature map
            cliff_cond: (B, cliff_cond_dim) optional CLIFF bbox conditioning
            ray_feats: (B, H*W, ray_cond_dim) camera-ray Fourier features
            final_only: decode and retain only the final transformer layer;
                valid only for a per-layer head whose caller disables auxiliary loss
        Returns:
            PoseHeadOutput containing parameters, camera, stacked auxiliary
            predictions and the final global query token.
        """
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        if final_only and not self.per_layer_predictions:
            raise ValueError("final_only requires per_layer_predictions=true")
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
                return_intermediate=not final_only,
            )
            if final_only:
                intermediates = [intermediates]
            pred_pose_list, pred_shape_list, pred_cam_list = [], [], []
            raw_b_list, b_list = [], []
            raw_sigma_list, sigma_list, log_sigma_list = [], [], []
            for h in intermediates:
                p, s, c = self._decode_single(h[:, 0], batch_size)
                pred_pose_list.append(p)
                pred_shape_list.append(s)
                pred_cam_list.append(c)
                if self.global_scale_b_query:
                    raw_b, b = self._decode_global_scale_b(h[:, 1])
                    raw_b_list.append(raw_b)
                    b_list.append(b)
                elif self.global_scale_sigma_query:
                    raw_sigma, sigma, log_sigma = self._decode_global_scale_sigma(h[:, 1])
                    raw_sigma_list.append(raw_sigma)
                    sigma_list.append(sigma)
                    log_sigma_list.append(log_sigma)
            pred_pose, pred_shape, pred_cam = (
                pred_pose_list[-1],
                pred_shape_list[-1],
                pred_cam_list[-1],
            )
        else:
            # IEF-style outer loop (kept for backward compat / ablation)
            pred_pose = self.init_pose.expand(batch_size, -1)
            pred_shape = self.init_shape.expand(batch_size, -1)
            pred_cam = self.init_cam.expand(batch_size, -1)
            pred_pose_list, pred_shape_list, pred_cam_list = [], [], []
            raw_b_list, b_list = [], []
            raw_sigma_list, sigma_list, log_sigma_list = [], [], []
            for _ in range(self.ief_iters):
                token = self._build_token(batch_size, device, dtype, cliff_cond)
                token_out = self.transformer(token, context=feat)
                body_token_out = token_out[:, 0]
                pred_pose = self.decpose(body_token_out) + pred_pose
                pred_shape = self.decshape(body_token_out) + pred_shape
                pred_cam = self.deccam(body_token_out) + pred_cam
                pred_pose_list.append(pred_pose)
                pred_shape_list.append(pred_shape)
                pred_cam_list.append(pred_cam)
                if self.global_scale_b_query:
                    raw_b, b = self._decode_global_scale_b(token_out[:, 1])
                    raw_b_list.append(raw_b)
                    b_list.append(b)
                elif self.global_scale_sigma_query:
                    raw_sigma, sigma, log_sigma = self._decode_global_scale_sigma(token_out[:, 1])
                    raw_sigma_list.append(raw_sigma)
                    sigma_list.append(sigma)
                    log_sigma_list.append(log_sigma)

        if self.camera_scale_mode != "none":
            # Clamp s >= 0.25 on FINAL prediction only — intermediate cams in
            # pred_soma_params_list are unclamped, matching v1's contract.
            # ste_clamp keeps the same forward value but lets 2D reprojection loss
            # push raw scale back out of the clamp floor if early training collapses.
            raw_cam_s = pred_cam[..., 0:1]
            clamped_cam_s = raw_cam_s.clamp(min=0.25)
            if self.camera_scale_mode == "ste_clamp":
                # Cancel identical values first to preserve the forward floor
                # for extreme negative raw scales while retaining STE gradients.
                cam_s = clamped_cam_s.detach() + (raw_cam_s - raw_cam_s.detach())
            else:
                cam_s = clamped_cam_s
            pred_cam = torch.cat(
                [
                    cam_s,
                    pred_cam[..., 1:],
                ],
                dim=-1,
            )

        pred_soma_params = self._params_from_pose_shape(pred_pose, pred_shape, batch_size)

        # Per-layer / per-iter intermediate predictions, stacked along dim 0
        # → (N*B, ...) where N = depth (per_layer_predictions) or ief_iters.
        def _layer_body_pose(pp):
            full = rot6d_to_rotmat(pp).view(batch_size, -1, 3, 3)[:, 1:]
            if self.skip_finger_joints:
                full = self._splice_mean_fingers(full)
            return full

        if final_only:
            pred_soma_params_list = {
                "body_pose": pred_soma_params["body_pose"],
                "global_orient": pred_soma_params["global_orient"],
                "identity_coeffs": pred_soma_params["identity_coeffs"],
                "cam": pred_cam_list[-1],
            }
            if self.has_scale:
                pred_soma_params_list["scale_params"] = pred_soma_params["scale_params"]
        else:
            pred_soma_params_list = {
                "body_pose": torch.cat([_layer_body_pose(pp) for pp in pred_pose_list], dim=0),
                "global_orient": torch.cat(
                    [
                        rot6d_to_rotmat(pp).view(batch_size, -1, 3, 3)[:, [0]]
                        for pp in pred_pose_list
                    ],
                    dim=0,
                ),
                "identity_coeffs": torch.cat(
                    [ps[:, : self.num_identity_coeffs] for ps in pred_shape_list],
                    dim=0,
                ),
                "cam": torch.cat(pred_cam_list, dim=0),
            }
            if self.has_scale:
                pred_soma_params_list["scale_params"] = torch.cat(
                    [self._scale_params_from_shape(ps) for ps in pred_shape_list],
                    dim=0,
                )
        if self.scale_parameterization == "local56":
            pred_soma_params_list["global_scale"] = torch.ones_like(
                pred_soma_params_list["scale_params"][:, :1]
            )
        if self.global_scale_b_query:
            pred_soma_params_list["global_scale_b_raw"] = torch.cat(raw_b_list, dim=0)
            pred_soma_params_list["global_scale_b"] = torch.cat(b_list, dim=0)
        elif self.global_scale_sigma_query:
            pred_soma_params_list["global_scale_sigma_raw"] = torch.cat(raw_sigma_list, dim=0)
            pred_soma_params_list["global_scale_sigma"] = torch.cat(sigma_list, dim=0)
            pred_soma_params_list["global_scale_log_sigma"] = torch.cat(log_sigma_list, dim=0)

        final_token = intermediates[-1][:, 0] if self.per_layer_predictions else body_token_out
        return PoseHeadOutput(pred_soma_params, pred_cam, pred_soma_params_list, final_token)
