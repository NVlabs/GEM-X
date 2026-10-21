# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject camera contracts incompatible with the canonical Hand objective."""

import pytest

from gem.pose.api import hand_training_config
from gem.pose.training import factory


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("conditioning", "camera_mode", "perspective"),
        ("conditioning", "camera_mode", "weak_perspective"),
        ("inference", "camera_mode", "perspective"),
        ("inference", "camera_representation", "crop_2d"),
        ("inference", "camera_representation", "legacy"),
        ("head", "camera_depth_mode", "log_z"),
        ("head", "camera_depth_mode", "s"),
        ("kp2d_translation", "enabled", False),
        ("kp2d_translation", "topology", "merged"),
    ],
)
def test_camera_mismatch_rejected_before_model_construction_or_weight_loading(
    monkeypatch, section, field, value
):
    release = hand_training_config()
    getattr(release, section)[field] = value

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Incompatible camera reached model construction or weight loading")

    monkeypatch.setattr(factory, "build_pose_model", unexpected_model)
    monkeypatch.setattr(factory, "load_pose_checkpoint", unexpected_model)
    for initialization in (None, "unused-weights.ckpt"):
        with pytest.raises(ValueError, match="calibrated log-lambda, separate 2D and RayPE"):
            factory.build_hand_training_model(
                soma_assets="unused-assets", release=release, initialization=initialization
            )


@pytest.mark.parametrize(
    "objective",
    [
        {"camera_depth_mode": "s"},
        {"camera_log_value_bounds": [-10.0, 10.0]},
        {"kp2d_translation": {"enabled": False}},
    ],
)
def test_objective_cannot_override_release_geometry_before_model_loading(monkeypatch, objective):
    def unexpected_model(*args, **kwargs):
        raise AssertionError("Conflicting objective reached model construction")

    monkeypatch.setattr(factory, "build_pose_model", unexpected_model)
    monkeypatch.setattr(factory, "load_pose_checkpoint", unexpected_model)
    with pytest.raises(ValueError, match="conflicts with the model release"):
        factory.build_hand_training_model(soma_assets="unused", objective=objective)
