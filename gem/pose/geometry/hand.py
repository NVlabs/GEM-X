# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MHR hand FK and rig-aware reconstruction from canonical left predictions.

The physical head contract is wrist1 + fingers24 rotation matrices, identity5,
and global1 + local26 scales. Outputs retain SOMA's 25-joint order, including
the five finger ends at indices4,9,14,19,24. Geometry is in meters.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from gem.common.soma.body import _NonPersistentFrozenRuntime


class HandGeometry(nn.Module):
    """Lazy SOMA-X0.3.3 MHR hands with rig-aware left/right reconstruction.

    ``runtime_factory(side, device)`` can inject compatible hand layers for
    tests. Normal construction needs caller-owned ``data_root`` assets. Fixed
    runtime weights remain frozen and do not appear in checkpoints.
    """

    num_joints = 25
    tip_indices = (4, 9, 14, 19, 24)

    def __init__(self, data_root: str | Path | None = None, *, mode="warp", runtime_factory=None):
        super().__init__()
        if data_root is None and runtime_factory is None:
            raise ValueError("Hand geometry requires an explicit SOMA asset directory")
        self.data_root = Path(data_root) if data_root is not None else None
        self.mode = mode
        self._runtime_factory = runtime_factory
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._layers = nn.ModuleDict()
        self._pose_mirror = None

    def _apply(self, fn, recurse=True):
        # Neural predictor precision is independent of the FP32 hand rig/MHR
        # runtime. Follow only the destination device for fixed geometry.
        previous_device = self._device_anchor.device
        self._device_anchor = fn(self._device_anchor)
        for layer in self._layers.values():
            layer.to(device=self._device_anchor.device, dtype=torch.float32)
        if self._pose_mirror is not None and previous_device != self._device_anchor.device:
            self._pose_mirror._cache.clear()
        return self

    def _layer(self, side, device):
        if side not in self._layers:
            self._device_anchor = self._device_anchor.to(device)
            if self._runtime_factory is not None:
                layer = self._runtime_factory(side, device)
            else:
                import soma
                from soma.hand import SOMAHandLayer
                from soma.units import Unit

                if soma.__version__ != "0.3.3":
                    raise RuntimeError("HandGeometry requires the validated SOMA-X0.3.3 runtime")
                layer = SOMAHandLayer(
                    data_root=str(self.data_root),
                    hand_type=side,
                    device=device,
                    identity_model_type="mhr",
                    mode=self.mode,
                    lod="mid",
                    output_unit=Unit.METERS,
                    correctives_model_path=None,
                )
            self._layers[side] = _NonPersistentFrozenRuntime(layer)
        return self._layers[side].module

    def _right_parameters(self, poses, identity, global_scale, local_scale, translation):
        # Reconstruct right-hand poses through the two hand rigs.
        # A matrix X-flip alone loses
        # the wrist/finger T-pose orientations and their chirality corrections.
        from soma.geometry.rig_utils import (
            PoseMirror_SOMA,
            joint_world_to_local,
            precompute_joint_orient,
            remove_joint_orient_local,
        )

        left = self._layer("left", poses.device)
        right = self._layer("right", poses.device)
        left.prepare_identity(identity, global_scale=global_scale, scale_params=local_scale)
        left_output = left.pose(
            poses=poses,
            pose2rot=False,
            absolute_pose=False,
            global_translation=translation,
            fk_only=True,
        )
        transforms = left_output.transforms
        if self._pose_mirror is None:
            names = (
                ["Root"] + list(left.rig_data["joint_names"]) + list(right.rig_data["joint_names"])
            )
            self._pose_mirror = PoseMirror_SOMA(names, root_name="Root")
        paired = (
            torch.eye(4, device=transforms.device, dtype=transforms.dtype)
            .expand(
                len(poses),
                self.num_joints * 2 + 1,
                4,
                4,
            )
            .clone()
        )
        paired[:, 1 : 1 + self.num_joints] = transforms
        mirrored = self._pose_mirror(paired)[:, 1 + self.num_joints :]
        local = joint_world_to_local(mirrored, right.joint_parent_ids)
        translation = mirrored[:, 0, :3, 3].clone()
        orient, parent = precompute_joint_orient(right.t_pose_world, right.joint_parent_ids)
        poses = remove_joint_orient_local(local[:, :, :3, :3], orient, parent)
        return poses, translation

    def rotate_wrist_orientation(self, wrist_orient, camera_rotation, *, right):
        """Rotate physical-side root poses through the SOMA wrist joint basis.

        Call after right-hand reconstruction. Finger poses are parent-relative
        and remain unchanged when the entire hand changes camera axes.
        """
        output = wrist_orient.clone()
        for side, selected in (("left", ~right), ("right", right)):
            if not bool(selected.any()):
                continue
            layer = self._layer(side, wrist_orient.device)
            orientation = torch.linalg.solve(
                layer.bind_pose_world[0].double(), layer.t_pose_world[0].double()
            )[:3, :3]
            rotation = camera_rotation[selected].double()
            output[selected, 0] = (
                orientation @ rotation @ orientation.T @ wrist_orient[selected, 0].double()
            ).to(wrist_orient.dtype)
        return output

    def forward(self, parameters, *, right: torch.Tensor | None = None, transl=None):
        """Return actual-side camera geometry, physical parameters and faces.

        Inputs describe canonical LEFT hands. ``right[B]`` identifies rows
        whose complete rigs must be mirrored. ``transl[B,3]`` is also in that
        canonical camera frame; it is mirrored exactly once for right rows.
        Omitting translation yields the corresponding wrist-local geometry.

        Returned ``parameters`` contains the four physical head fields in the
        reconstructed side convention. Raw non-orthonormal ``pose_6d`` tokens
        belong to the original head output and are deliberately not relabeled.
        """
        return self._decode(parameters, right=right, transl=transl, canonical=True)

    def forward_physical(self, parameters, *, right: torch.Tensor | None = None, transl=None):
        """Decode saved physical-side parameters without another mirror.

        ``right[B]`` selects the matching hand rig. Poses and translation are
        already expressed in that side's source-camera convention. This is the
        FK path for filtered or externally edited saved Pose parameters.
        """
        return self._decode(parameters, right=right, transl=transl, canonical=False)

    def _decode(self, parameters, *, right, transl, canonical):
        wrist = parameters["wrist_orient"]
        count, device = len(wrist), wrist.device
        shapes = {
            "wrist_orient": (count, 1, 3, 3),
            "hand_pose": (count, 24, 3, 3),
            "identity_coeffs": (count, 5),
            "scale_params": (count, 27),
        }
        if any(tuple(parameters[name].shape) != shape for name, shape in shapes.items()):
            raise ValueError(
                "Hand geometry requires wrist1/fingers24 matrices, identity5 and scale27"
            )
        if right is None:
            right = torch.zeros(count, dtype=torch.bool, device=device)
        elif right.dtype != torch.bool or right.shape != (count,):
            raise ValueError("right must be a bool[B] side mask")
        else:
            right = right.to(device)
        if transl is not None and tuple(transl.shape) != (count, 3):
            raise ValueError("Hand camera translation must have shape [B,3]")
        with torch.autocast(device_type=device.type, enabled=False):
            poses = torch.cat((wrist.float(), parameters["hand_pose"].float()), dim=1)
            identity = parameters["identity_coeffs"].float()
            scales = parameters["scale_params"].float()
            translation = poses.new_zeros(count, 3) if transl is None else transl.float()
            groups, order = [], []
            for side, selected in (("left", ~right), ("right", right)):
                indices = selected.nonzero(as_tuple=True)[0]
                if len(indices) == 0:
                    continue
                subset_poses, subset_translation = poses[indices], translation[indices]
                subset_identity, subset_scales = identity[indices], scales[indices]
                if canonical and side == "right":
                    subset_poses, subset_translation = self._right_parameters(
                        subset_poses,
                        subset_identity,
                        subset_scales[:, 0],
                        subset_scales[:, 1:],
                        subset_translation,
                    )
                layer = self._layer(side, device)
                output = layer(
                    subset_poses,
                    subset_identity,
                    pose2rot=False,
                    global_translation=None,
                    global_scale=subset_scales[:, 0],
                    scale_params=subset_scales[:, 1:],
                    apply_correctives=False,
                )
                faces = torch.as_tensor(layer.faces, device=device, dtype=torch.long)
                groups.append(
                    {
                        "joints": output["joints"] + subset_translation[:, None],
                        "vertices": output["vertices"] + subset_translation[:, None],
                        "poses": subset_poses,
                        "transl": subset_translation,
                        "faces": faces[None].expand(len(indices), -1, -1),
                    }
                )
                order.append(indices)
            inverse = torch.cat(order).argsort()
            result = {
                name: torch.cat([group[name] for group in groups])[inverse] for name in groups[0]
            }
            actual_poses = result.pop("poses")
            result["parameters"] = {
                "wrist_orient": actual_poses[:, :1],
                "hand_pose": actual_poses[:, 1:],
                "identity_coeffs": identity,
                "scale_params": scales,
            }
            return result
