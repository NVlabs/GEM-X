# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gem.motion.api import MotionConditionSpec, MotionPredictor


class RecordingPipeline(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.called = False
        self.denoiser3d = SimpleNamespace(obs_num_joints=num_joints)

    def forward(self, inputs):
        self.called = True
        assert not torch.is_grad_enabled()
        return inputs["f_imgseq"]


@pytest.fixture(params=[("soma29_nofinger_v1", 29), ("soma77_native_v1", 77)])
def native_conditions(request):
    joint_set, num_joints = request.param
    spec = dict(
        pose_release_id="pose-body-test",
        observation_source="vitpose-test",
        observation_joint_set=joint_set,
    )
    return spec, num_joints


@pytest.mark.parametrize(
    "field", ["pose_release_id", "observation_source", "observation_joint_set"]
)
def test_incompatible_cached_conditions_fail_before_network(field, native_conditions):
    spec, num_joints = native_conditions
    pipeline = RecordingPipeline(num_joints)
    predictor = MotionPredictor(pipeline, MotionConditionSpec(**spec))
    altered = {**spec, field: "different-release"}
    with pytest.raises(ValueError, match="do not match"):
        predictor.run({"condition_spec": altered})
    assert not pipeline.called


def test_matching_conditions_are_passed_to_eval_pipeline(native_conditions):
    spec, num_joints = native_conditions
    pipeline = RecordingPipeline(num_joints)
    predictor = MotionPredictor(pipeline, MotionConditionSpec(**spec))
    token = torch.randn(1, 2, 1024)
    assert (
        predictor.run(
            {"condition_spec": spec, "f_imgseq": token, "obs": torch.zeros(1, 2, num_joints, 3)}
        )
        is token
    )
    assert pipeline.called and not pipeline.training


def test_pose_tokens_use_the_existing_motion_cache_feature_version(native_conditions):
    from gem.pose.api import body_config
    from gem.pose.heads.output import ImageCondition

    native_spec, num_joints = native_conditions
    release = body_config()
    token = torch.randn(2, 1024)
    condition = ImageCondition(token, torch.zeros(2, 3), {}, "body", release.pose_release_id)
    spec = MotionConditionSpec.from_pose(
        condition,
        observation_source=native_spec["observation_source"],
        observation_joint_set=native_spec["observation_joint_set"],
    )
    assert spec.pose_release_id == "gem-pose-body-s240000-v1:pose_tokens"
    pipeline = RecordingPipeline(num_joints)
    expected = MotionConditionSpec(**{**native_spec, "pose_release_id": spec.pose_release_id})
    predictor = MotionPredictor(pipeline, expected)
    actual = predictor.run(
        {
            "condition_spec": asdict(spec),
            "f_imgseq": token[None],
            "obs": torch.zeros(1, 2, num_joints, 3),
        }
    )
    torch.testing.assert_close(actual, token[None], atol=0, rtol=0)


def test_anonymous_pose_features_cannot_define_trained_motion_conditions(native_conditions):
    spec, _ = native_conditions
    with pytest.raises(ValueError, match="identified Pose release"):
        MotionConditionSpec.from_pose(
            SimpleNamespace(feature_version=None),
            observation_source=spec["observation_source"],
            observation_joint_set=spec["observation_joint_set"],
        )


def test_compact30_conditioning_is_not_a_released_motion_model():
    pipeline = RecordingPipeline(30)
    conditions = MotionConditionSpec("pose-body-test", "sapiens-test", "sapiens70_nofinger30_v1")
    with pytest.raises(ValueError, match="Unknown native SOMA observation schema"):
        MotionPredictor(pipeline, conditions)
    assert not pipeline.called


@pytest.mark.parametrize("legacy_expected", [False, True])
def test_public_body_features_accept_the_exact_historical_cache_name(
    native_conditions, legacy_expected
):
    from gem.pose.release_ids import BODY_RELEASE_ID, LEGACY_BODY_RELEASE_ID

    spec, num_joints = native_conditions
    public = f"{BODY_RELEASE_ID}:pose_tokens"
    legacy = f"{LEGACY_BODY_RELEASE_ID}:pose_tokens"
    expected, cached = (legacy, public) if legacy_expected else (public, legacy)
    predictor = MotionPredictor(
        RecordingPipeline(num_joints), MotionConditionSpec(**{**spec, "pose_release_id": expected})
    )
    cached_spec = {**spec, "pose_release_id": cached}
    tokens = torch.randn(1, 2, 1024)
    output = predictor.run(
        {"condition_spec": cached_spec, "f_imgseq": tokens, "obs": torch.zeros(1, 2, num_joints, 3)}
    )
    assert output is tokens
    assert predictor.conditions.pose_release_id == public
    assert cached_spec["pose_release_id"] == cached
