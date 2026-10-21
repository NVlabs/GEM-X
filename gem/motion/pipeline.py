# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native parameter inference from prepared motion conditions.

Detection, crop normalization and the frozen pose feature extractor are caller
responsibilities. Legacy compact30 checkpoints retain their original observation
semantics; the new native-joint training path is integrated separately.
"""

import torch
from torch import nn

from gem.common.camera import compute_bbox_info_bedlam, compute_transl_full_cam
from gem.motion.codec.soma_native_fixed_gs1 import SomaNativeFixedGs1EnDecoder
from gem.motion.codec.world import rollout_world
from gem.motion.network import R3MatchedGEMNetwork


class MotionPipeline(nn.Module):
    def __init__(self, network: R3MatchedGEMNetwork, codec: SomaNativeFixedGs1EnDecoder):
        super().__init__()
        self.denoiser3d = network
        self.endecoder = codec
        self.register_buffer(
            "cam_angvel_mean", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), persistent=False
        )
        self.register_buffer(
            "cam_angvel_std", torch.tensor([0.001, 0.1, 0.1, 0.1, 0.001, 0.1]), persistent=False
        )

    def forward_train(self, inputs):
        """Run the video network with the original normalized-condition policy.

        Geometry and clean supervision remain in ``inputs`` for the video loss.
        Modality draws intentionally use the CPU global RNG, as trained.
        """
        masks = inputs["mask"]
        cliff = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])
        if "bbox_condition" in masks:
            cliff = cliff.masked_fill(~masks["bbox_condition"][..., None], 0)
        conditions = {
            "obs": inputs["obs"],
            "f_cliffcam": cliff,
            "f_cam_angvel": (inputs["cam_angvel"] - self.cam_angvel_mean) / self.cam_angvel_std,
            "f_imgseq": inputs["f_imgseq"],
        }
        if "input_condition" in masks:
            keep = masks["input_condition"]
            conditions = {
                name: value.masked_fill(~keep.reshape(*keep.shape, *([1] * (value.ndim - 2))), 0)
                for name, value in conditions.items()
            }
        for name, value in conditions.items():
            dropped = torch.rand(value.shape[:2]) < 0.1
            conditions[name] = value.clone()
            conditions[name][dropped] = 0
        return self.denoiser3d(
            length=inputs["length"], positions=inputs.get("positions"), **conditions
        )

    def forward(self, inputs):
        """Predict raw camera/world SOMA parameters without contact processing.

        ``positions`` optionally gives reference-frame times for RoPE. If the
        sampling interval differs from one reference frame, supply
        ``motion_reference_step_scale`` and camera rotations for the actual
        output intervals as ``cam_angvel_source_interval``. The latter defaults
        to ``cam_angvel`` when no separate interval conversion is needed.
        """
        output = self.denoiser3d(
            length=inputs["length"],
            obs=inputs["obs"],
            f_cliffcam=compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"]),
            f_cam_angvel=(inputs["cam_angvel"] - self.cam_angvel_mean) / self.cam_angvel_std,
            f_imgseq=inputs["f_imgseq"],
            positions=inputs.get("positions"),
        )
        # The deployed decoder/world geometry runs in float32 even under AMP.
        with torch.autocast(device_type=output["pred_x"].device.type, enabled=False):
            decoded = self.endecoder.decode(output["pred_x"].float())
            transl = compute_transl_full_cam(
                output["pred_cam"].float(), inputs["bbx_xys"], inputs["K_fullimg"]
            )
            camera = self.endecoder.build_incam_params(decoded, transl)
            velocity = decoded["local_transl_vel"]
            if "motion_reference_step_scale" in inputs:
                velocity = velocity * inputs["motion_reference_step_scale"][..., None]
            roots = rollout_world(
                decoded["global_orient_gv"],
                velocity,
                decoded["global_orient"],
                inputs.get("cam_angvel_source_interval", inputs["cam_angvel"]).float(),
            )
            world = self.endecoder.build_global_params(decoded, roots)
        return {
            "model_output": output,
            "decode_dict": decoded,
            "pred_body_params_incam": camera,
            "pred_soma_params_incam": camera,
            "pred_body_params_global": world,
            "pred_soma_params_global": world,
            "static_conf_logits": output["static_conf_logits"],
        }

    @classmethod
    def from_legacy_checkpoint(cls, checkpoint_path, stats_path, *, network_config=None):
        """Load the original network plus its required normalization sidecar.

        Old checkpoints did not store normalization statistics. The sidecar
        must be the one used during training; no neutral statistics are inferred.
        New state_dicts include those buffers through the codec.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        state = checkpoint["state_dict"]
        prefix = "pipeline.denoiser3d."
        network = R3MatchedGEMNetwork(**(network_config or {}))
        network.load_state_dict(
            {
                name[len(prefix) :]: tensor
                for name, tensor in state.items()
                if name.startswith(prefix)
            },
            strict=True,
        )
        codec = SomaNativeFixedGs1EnDecoder(stats_path=stats_path)
        return cls(network, codec).eval()
