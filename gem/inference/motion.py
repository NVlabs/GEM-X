# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached-condition Motion inference with explicit release and clock settings."""

import torch
from torch import nn

from gem.motion.api import MotionConditionSpec, MotionPredictor
from gem.motion.codec.soma_native_fixed_gs1 import SomaNativeFixedGs1EnDecoder
from gem.motion.frontend.joint_sets import get_observation_schema, get_soma_observation_schema
from gem.motion.frontend.time import TIME_INPUT_KEYS, ReferenceMotionTime
from gem.motion.network import R3MatchedGEMNetwork
from gem.motion.pipeline import MotionPipeline
from gem.pose.release_ids import canonical_pose_release_id

from .serialization import tensor_tree


class TimedMotionPipeline(nn.Module):
    """Apply the trained clock after MotionPredictor prepares observations."""

    def __init__(self, pipeline, time_options):
        super().__init__()
        self.pipeline = pipeline
        self.timing = ReferenceMotionTime(**time_options)

    @property
    def denoiser3d(self):
        return self.pipeline.denoiser3d

    def forward(self, inputs):
        if inputs.get("positions") is not None:
            inputs = {**inputs, "time_positions": inputs["positions"]}
        prepared = self.timing.prepare_pipeline(inputs, train=False, postproc=False)
        return self.pipeline({**prepared, "positions": self.timing.clock(prepared)})


def load_predictor(checkpoint_path, config, *, stats_path=None, device="cpu"):
    """Native checkpoints carry codec buffers; old checkpoints need --stats.

    Architecture, including obs_num_joints, and expected native conditions are
    caller-owned release settings.
    Native codec buffers retain the exact effective std saved during training.
    """
    conditions = MotionConditionSpec(**config["condition_spec"])
    schema = get_soma_observation_schema(conditions.observation_joint_set)
    if config["network"]["obs_num_joints"] != schema.num_joints:
        raise ValueError("Release network joint count differs from its observation schema")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    for metadata_key in ("gem_motion_training", "hyper_parameters"):
        metadata = checkpoint.get(metadata_key, {})
        if "observation_schema" in metadata:
            saved_schema = get_observation_schema(metadata["observation_schema"])
            if saved_schema.name != schema.name:
                raise ValueError("Checkpoint observation schema differs from the requested release")
        if (
            "pose_release_id" in metadata
            and canonical_pose_release_id(metadata["pose_release_id"]) != conditions.pose_release_id
        ):
            raise ValueError("Checkpoint pose feature release differs from the requested release")
    state = checkpoint["state_dict"]
    network = R3MatchedGEMNetwork(**config["network"])
    prefix = "pipeline.denoiser3d."
    network.load_state_dict(
        {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    codec_prefix = "pipeline.endecoder."
    codec_state = {
        key[len(codec_prefix) :]: value
        for key, value in state.items()
        if key.startswith(codec_prefix)
    }
    if codec_state:
        if stats_path is not None:
            raise ValueError("Native checkpoint includes normalization statistics; omit --stats")
        codec = SomaNativeFixedGs1EnDecoder(
            stats_mean=codec_state["mean"], stats_std=codec_state["raw_std"]
        )
        codec.load_state_dict(codec_state, strict=True)
    else:
        if stats_path is None:
            raise ValueError("This legacy checkpoint requires its training --stats NPZ sidecar")
        codec = SomaNativeFixedGs1EnDecoder(stats_path=stats_path)
    pipeline = MotionPipeline(network, codec).to(device).eval()
    if config.get("time_options") is not None:
        pipeline = TimedMotionPipeline(pipeline, config["time_options"]).eval()
    return MotionPredictor(pipeline, conditions)


def run_motion(predictor, sequence, *, device="cpu"):
    output = predictor.run(tensor_tree(sequence, device=device))
    return tensor_tree(
        {
            "format": "gem_motion_predictions_v1",
            "condition_spec": sequence["condition_spec"],
            "conditions": sequence,
            "length": sequence["length"],
            **{
                key: sequence[key]
                for key in (*TIME_INPUT_KEYS, "positions", "meta")
                if key in sequence
            },
            "coordinates": {"incam": "camera_meters", "global": "raw_world_rollout_meters"},
            "pred_soma_params_incam": output["pred_soma_params_incam"],
            "pred_soma_params_global": output["pred_soma_params_global"],
            "static_conf_logits": output["static_conf_logits"],
        }
    )
