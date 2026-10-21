# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion inference on cached, explicitly identified pose/2D conditions."""

from dataclasses import dataclass

import torch

from gem.pose.release_ids import canonical_pose_release_id

from .frontend.joint_sets import get_soma_observation_schema
from .frontend.observation import prepare_native_observations
from .pipeline import MotionPipeline


@dataclass(frozen=True)
class MotionConditionSpec:
    """Semantic compatibility cannot be inferred from tensor dimensions alone."""

    pose_release_id: str
    observation_source: str
    observation_joint_set: str

    def __post_init__(self):
        object.__setattr__(self, "pose_release_id", canonical_pose_release_id(self.pose_release_id))

    @classmethod
    def from_pose(cls, condition, *, observation_source, observation_joint_set):
        """Build the cache-compatible contract from a Pose ``ImageCondition``.

        The historical ``pose_release_id`` field stores a qualified feature
        version (``release:pose_tokens``), while Pose's own release ID names
        only the model. Keep that distinction explicit at this boundary.
        """
        if condition.feature_version is None:
            raise ValueError("Motion conditioning requires an identified Pose release")
        return cls(condition.feature_version, observation_source, observation_joint_set)


class MotionPredictor:
    """Run a native pipeline with the feature contract used during training.

    Both the release and cached input must identify the frozen pose model and
    detector/joint schema. Changing those inputs requires a trained compatible
    checkpoint, even when the tensors have the same shape.
    """

    def __init__(self, pipeline: MotionPipeline, conditions: MotionConditionSpec):
        schema = get_soma_observation_schema(conditions.observation_joint_set)
        if schema.num_joints != pipeline.denoiser3d.obs_num_joints:
            raise ValueError(
                f"Observation schema {schema.name} has {schema.num_joints} joints, "
                f"but the network expects {pipeline.denoiser3d.obs_num_joints}"
            )
        self.pipeline = pipeline.eval()
        self.conditions = conditions

    @torch.inference_mode()
    def run(self, sequence_dict):
        """Use prepared ``obs`` or select/normalize cached ``kp2d_native77``.

        Native cache inputs include the same ``condition_spec`` as prepared
        inputs, plus full-image keypoints and the original ``bbx_xys``.
        Camera, pose features and the full77 cache are never modified.
        """
        actual = MotionConditionSpec(**sequence_dict["condition_spec"])
        if actual != self.conditions:
            raise ValueError(
                f"Motion conditions do not match the trained release: "
                f"expected {self.conditions}, received {actual}"
            )
        if "obs" not in sequence_dict:
            sequence_dict = {
                **sequence_dict,
                "obs": prepare_native_observations(
                    sequence_dict["kp2d_native77"],
                    sequence_dict["bbx_xys"],
                    self.conditions.observation_joint_set,
                ),
            }
        return self.pipeline(sequence_dict)
