# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import gem.utils.matrix as matrix
from gem.utils.motion_utils import get_local_transl_vel, get_static_joint_mask
from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gem.utils.soma_utils.soma_layer import SomaLayer

from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(
        self,
        stats_name="DEFAULT_01",
        encode_type="soma",
        feature_arr=None,
        stats_arr=None,
        noise_pose_k=10,
        clip_std=False,
        feat_dim=None,
    ):
        super().__init__()

        if encode_type in ["soma", "soma_v2"]:
            feature_arr = [encode_type]
            stats_arr = [stats_name]

        # Define feature dimensions as a class attribute
        self.FEATURE_DIMS = {
            "soma": 591,
            "soma_v2": 585,
        }
        if feat_dim is not None:
            self.FEATURE_DIMS[encode_type] = feat_dim

        # Store stats for each feature type
        self.stats_dict = {}

        for feature, stats_name in zip(feature_arr, stats_arr):
            stats = getattr(stats_compose, stats_name)
            mean = torch.tensor(stats["mean"]).float()
            std = torch.tensor(stats["std"]).float()

            feature_dim = self.FEATURE_DIMS[feature]
            if stats_name != "DEFAULT_01":
                assert mean.shape[-1] == feature_dim
                assert std.shape[-1] == feature_dim

            if clip_std:
                std[std < 1] = 1

            self.stats_dict[feature] = {"mean": mean, "std": std}

        # Store feature configuration
        self.feature_arr = feature_arr
        self.stats_arr = stats_arr
        self.clip_std = clip_std

        # option
        self.noise_pose_k = noise_pose_k
        self.encode_type = encode_type
        self.obs_indices_dict = None

        self.soma_model = None

    def normalize(self, x, feature_type):
        """Normalize input using stats for specific feature type"""
        stats = self.stats_dict[feature_type]
        return (x - stats["mean"].to(x)) / stats["std"].to(x)

    def denormalize(self, x_norm, feature_type):
        """Denormalize input using stats for specific feature type"""
        stats = self.stats_dict[feature_type]
        return x_norm * stats["std"].to(x_norm) + stats["mean"].to(x_norm)

    def get_static_gt(self, inputs, vel_thr):
        if "soma_params_w" in inputs:
            # SOMA77: [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
            joint_ids = [69, 70, 74, 75, 14, 42]
            self._ensure_soma_model()
            soma_params_w = {k: v.float().cpu() for k, v in inputs["soma_params_w"].items()}
            gt_w_j3d = self.soma_model(**soma_params_w)["joints"].to(
                inputs["soma_params_w"]["body_pose"].device
            )
        else:
            B, L = inputs["target_x"].shape[:2]
            device = inputs["target_x"].device
            return torch.zeros((B, L, 6), device=device)

        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        return static_gt

    def _ensure_soma_model(self):
        """Lazily initialize the SOMA body model."""
        if self.soma_model is None:
            self.soma_model = SomaLayer(
                data_root="inputs/soma_assets",
                low_lod=True,
                device="cuda",
                identity_model_type="mhr",
                mode="warp",
            )

    def fk_v2(
        self,
        body_pose,
        identity_coeffs=None,
        scale_params=None,
        global_orient=None,
        transl=None,
        get_intermediate=False,
        **kwargs,
    ):
        """Forward kinematics using SOMA body model.

        Args:
            body_pose: (B, L, (J-1)*3) axis-angle
            identity_coeffs: (B, L, C)
            scale_params: (B, L, S)
            global_orient: (B, L, 3) axis-angle
            transl: (B, L, 3)
            get_intermediate: if True, return (joints, local_mat, fk_mat)

        Returns:
            joints: (B, L, 77, 3), or (joints, local_mat, fk_mat) when get_intermediate=True
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, J, 3, 3)

        self._ensure_soma_model()
        skeleton = self.soma_model.get_skeleton(
            identity_coeffs.float(), scale_params.float()
        )  # (B, L, 77, 3)
        parents = self.soma_model.parents
        parents_tensor = torch.tensor(parents, device=body_pose.device)

        local_skeleton = skeleton - skeleton[:, :, parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # (B, L, 77, 3)

        mat = matrix.get_TRS(rotmat, local_skeleton)  # (B, L, 77, 4, 4)
        fk_mat = matrix.forward_kinematics(mat, parents)  # (B, L, 77, 4, 4)
        joints = matrix.get_position(fk_mat)  # (B, L, 77, 3)
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def build_obs_indices_dict(self):
        """
        Initialize observation index mapping for decode-time use.
        This mirrors the legacy behavior where eval/demo could decode without
        a preceding encode() call.
        """
        for feature in self.feature_arr:
            if feature == "soma":
                self.obs_indices_dict = {
                    "body_pose": (0, 456),
                    "identity_coeffs": (456, 501),
                    "scale_params": (501, 576),
                    "global_orient": (576, 582),
                    "global_orient_gv": (582, 588),
                    "local_transl_vel": (588, 591),
                }
            elif feature == "soma_v2":
                self.obs_indices_dict = {
                    "body_pose": (0, 456),
                    "identity_coeffs": (456, 501),
                    "scale_params": (501, 570),
                    "global_orient": (570, 576),
                    "global_orient_gv": (576, 582),
                    "local_transl_vel": (582, 585),
                }

    def encode(self, inputs):
        """Composite encoder that combines multiple feature types"""
        encoded_features = []

        for feature in self.feature_arr:
            if feature == "soma":
                encoded = self.encode_soma(inputs)
            elif feature == "soma_v2":
                encoded = self.encode_soma_v2(inputs)
            encoded_features.append(encoded)

        return torch.cat(encoded_features, dim=-1)

    def encode_soma(self, inputs):
        J = 77
        self.obs_indices_dict = {
            "body_pose": (0, (J - 1) * 6),
            "identity_coeffs": ((J - 1) * 6, (J - 1) * 6 + 45),
            "scale_params": ((J - 1) * 6 + 45, (J - 1) * 6 + 45 + 75),
            "global_orient": ((J - 1) * 6 + 45 + 75, (J - 1) * 6 + 45 + 75 + 6),
            "global_orient_gv": (
                (J - 1) * 6 + 45 + 75 + 6,
                (J - 1) * 6 + 45 + 75 + 6 + 6,
            ),
            "local_transl_vel": (
                (J - 1) * 6 + 45 + 75 + 6 + 6,
                (J - 1) * 6 + 45 + 75 + 6 + 6 + 3,
            ),
        }
        B, L = inputs["soma_params_c"]["body_pose"].shape[:2]
        soma_params_c = inputs["soma_params_c"]
        body_pose = soma_params_c["body_pose"].reshape(B, L, J - 1, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        identity_coeffs = soma_params_c["identity_coeffs"]
        scale_params = soma_params_c["scale_params"]
        global_orient_R = axis_angle_to_matrix(soma_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)
        R_c2gv = inputs["R_c2gv"]
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)
        soma_params_w = inputs["soma_params_w"]
        local_transl_vel = get_local_transl_vel(
            soma_params_w["transl"], soma_params_w["global_orient"]
        )
        x = torch.cat(
            [
                body_pose_r6d,
                identity_coeffs,
                scale_params,
                global_orient_r6d,
                global_orient_gv_r6d,
                local_transl_vel,
            ],
            dim=-1,
        )
        return self.normalize(x, "soma")

    def encode_soma_v2(self, inputs):
        J = 77
        self.obs_indices_dict = {
            "body_pose": (0, (J - 1) * 6),
            "identity_coeffs": ((J - 1) * 6, (J - 1) * 6 + 45),
            "scale_params": ((J - 1) * 6 + 45, (J - 1) * 6 + 45 + 69),
            "global_orient": ((J - 1) * 6 + 45 + 69, (J - 1) * 6 + 45 + 69 + 6),
            "global_orient_gv": (
                (J - 1) * 6 + 45 + 69 + 6,
                (J - 1) * 6 + 45 + 69 + 6 + 6,
            ),
            "local_transl_vel": (
                (J - 1) * 6 + 45 + 69 + 6 + 6,
                (J - 1) * 6 + 45 + 69 + 6 + 6 + 3,
            ),
        }
        B, L = inputs["soma_params_c"]["body_pose"].shape[:2]
        soma_params_c = inputs["soma_params_c"]
        body_pose = soma_params_c["body_pose"].reshape(B, L, J - 1, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        identity_coeffs = soma_params_c["identity_coeffs"]
        scale_params = soma_params_c["scale_params"]
        global_orient_R = axis_angle_to_matrix(soma_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)
        R_c2gv = inputs["R_c2gv"]
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)
        soma_params_w = inputs["soma_params_w"]
        local_transl_vel = get_local_transl_vel(
            soma_params_w["transl"], soma_params_w["global_orient"]
        )
        x = torch.cat(
            [
                body_pose_r6d,
                identity_coeffs,
                scale_params,
                global_orient_r6d,
                global_orient_gv_r6d,
                local_transl_vel,
            ],
            dim=-1,
        )
        return self.normalize(x, "soma_v2")

    def decode(self, x_norm):
        """Composite decoder that handles multiple feature types"""
        current_idx = 0
        decoded_outputs = {}

        for feature in self.feature_arr:
            feature_size = self.FEATURE_DIMS[feature]
            feature_norm = x_norm[..., current_idx : current_idx + feature_size]

            if feature == "soma":
                decoded = self.decode_soma(feature_norm)
            elif feature == "soma_v2":
                decoded = self.decode_soma_v2(feature_norm)

            decoded_outputs.update(decoded)
            current_idx += feature_size

        return decoded_outputs

    def decode_soma(self, x_norm):
        B, L, _ = x_norm.shape
        x = self.denormalize(x_norm, "soma")
        body_pose_r6d = x[:, :, : self.obs_indices_dict["body_pose"][1]]
        identity_coeffs = x[
            :,
            :,
            self.obs_indices_dict["identity_coeffs"][0] : self.obs_indices_dict["identity_coeffs"][
                1
            ],
        ]
        scale_params = x[
            :,
            :,
            self.obs_indices_dict["scale_params"][0] : self.obs_indices_dict["scale_params"][1],
        ]
        global_orient_r6d = x[
            :,
            :,
            self.obs_indices_dict["global_orient"][0] : self.obs_indices_dict["global_orient"][1],
        ]
        global_orient_gv_r6d = x[
            :,
            :,
            self.obs_indices_dict["global_orient_gv"][0] : self.obs_indices_dict[
                "global_orient_gv"
            ][1],
        ]
        local_transl_vel = x[
            :,
            :,
            self.obs_indices_dict["local_transl_vel"][0] : self.obs_indices_dict[
                "local_transl_vel"
            ][1],
        ]
        body_pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
        ).flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))
        offset = torch.zeros((B, L, 3), device=x.device)
        return {
            "body_pose": body_pose,
            "identity_coeffs": identity_coeffs,
            "scale_params": scale_params,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
            "offset": offset,
        }

    def decode_soma_v2(self, x_norm):
        B, L, _ = x_norm.shape
        x = self.denormalize(x_norm, "soma_v2")
        body_pose_r6d = x[:, :, : self.obs_indices_dict["body_pose"][1]]
        identity_coeffs = x[
            :,
            :,
            self.obs_indices_dict["identity_coeffs"][0] : self.obs_indices_dict["identity_coeffs"][
                1
            ],
        ]
        scale_params = x[
            :,
            :,
            self.obs_indices_dict["scale_params"][0] : self.obs_indices_dict["scale_params"][1],
        ]
        global_orient_r6d = x[
            :,
            :,
            self.obs_indices_dict["global_orient"][0] : self.obs_indices_dict["global_orient"][1],
        ]
        global_orient_gv_r6d = x[
            :,
            :,
            self.obs_indices_dict["global_orient_gv"][0] : self.obs_indices_dict[
                "global_orient_gv"
            ][1],
        ]
        local_transl_vel = x[
            :,
            :,
            self.obs_indices_dict["local_transl_vel"][0] : self.obs_indices_dict[
                "local_transl_vel"
            ][1],
        ]
        body_pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
        ).flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))
        offset = torch.zeros((B, L, 3), device=x.device)
        return {
            "body_pose": body_pose,
            "identity_coeffs": identity_coeffs,
            "scale_params": scale_params,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
            "offset": offset,
        }

    def get_motion_dim(self):
        """Calculate total dimension based on enabled features"""
        return sum(self.FEATURE_DIMS[feature] for feature in self.feature_arr)

    def get_obs_indices(self, obs):
        return self.obs_indices_dict[obs]
