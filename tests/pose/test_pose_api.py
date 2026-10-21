# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from gem.common.backbones import DinoV3Backbone, dinov3_vith16plus_config
from gem.pose import BodyTask, HandTask, PoseModel
from gem.pose.api import PoseReleaseConfig, load_pose_checkpoint, load_pose_predictor
from gem.pose.conditioning import CliffConditioning
from gem.pose.heads import SOMABodyHeadV2, SOMAHandHeadV2


def tiny_model(task):
    backbone_config = dinov3_vith16plus_config()
    backbone_config.hidden_size = 16
    backbone_config.num_hidden_layers = 1
    backbone_config.num_attention_heads = 2
    backbone_config.intermediate_size = 32
    head = dict(
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=2,
        transformer_dim_head=8,
        transformer_mlp_dim=48,
        transformer_context_dim=16,
        cliff_cond_dim=3,
    )
    if task == "body":
        head.update(
            num_identity_coeffs=128,
            num_scale_params=56,
            scale_comps_dim=0,
            scale_param_mode="direct",
            scale_parameterization="local56",
            fixed_global_scale=1.0,
        )
        pose_task = BodyTask(SOMABodyHeadV2(**head))
    else:
        pose_task = HandTask(SOMAHandHeadV2(**head))
    release = PoseReleaseConfig(
        task=task,
        backbone={"config": backbone_config},
        head=head,
        conditioning={"camera_mode": "sam3db_perspective"},
        pose_release_id="test-v1",
    )
    model = PoseModel(
        DinoV3Backbone(**release.backbone),
        pose_task,
        conditioning=CliffConditioning(),
        pose_release_id=release.pose_release_id,
    ).eval()
    return model, release


@pytest.mark.parametrize("task", ["body", "hand"])
@pytest.mark.parametrize("legacy_head", [False, True])
def test_checkpoint_roundtrip_uses_explicit_release(task, legacy_head, tmp_path):
    original, release = tiny_model(task)
    state_dict = original.state_dict()
    if legacy_head:
        prefix = "soma_head." if task == "body" else "hand_head."
        state_dict = {key.replace("task.head.", prefix): value for key, value in state_dict.items()}
    path = tmp_path / "pose.ckpt"
    torch.save(
        {"state_dict": state_dict, "hyper_parameters": {"cfg": {"_target_": "unused.WrongModel"}}},
        path,
    )
    loaded = load_pose_checkpoint(path, release)
    batch = {
        "img": torch.randn(1, 3, 32, 32),
        "bbx_xys": torch.tensor([[350.0, 210.0, 160.0]]),
        "K_fullimg": torch.tensor([[[600.0, 0.0, 320.0], [0.0, 625.0, 240.0], [0.0, 0.0, 1.0]]]),
    }
    expected, actual = original.encode(batch), loaded.encode(batch)
    torch.testing.assert_close(actual.pose_tokens, expected.pose_tokens, atol=0, rtol=0)
    torch.testing.assert_close(actual.pred_cam, expected.pred_cam, atol=0, rtol=0)
    assert actual.pose_release_id == "test-v1"
    assert not loaded.training


def test_loader_rejects_missing_learned_head_weight(tmp_path):
    original, release = tiny_model("hand")
    state_dict = original.state_dict()
    del state_dict["task.head.deccam.2.weight"]
    path = tmp_path / "missing.ckpt"
    torch.save(state_dict, path)
    with pytest.raises(RuntimeError, match="deccam.2.weight"):
        load_pose_checkpoint(path, release)


def test_loader_rejects_unconsumed_checkpoint_tensors(tmp_path):
    original, release = tiny_model("hand")
    state_dict = original.state_dict()
    state_dict["other_head.weight"] = torch.ones(2)
    path = tmp_path / "extra.ckpt"
    torch.save(state_dict, path)
    with pytest.raises(ValueError, match="other_head.weight"):
        load_pose_checkpoint(path, release)


def test_rgb_encode_reuses_release_crop_and_keeps_geometry_uninitialized(tmp_path):
    original, release = tiny_model("body")
    release.inference = {"image_size": 32}
    checkpoint = tmp_path / "body.ckpt"
    torch.save(original.state_dict(), checkpoint)
    predictor = load_pose_predictor(checkpoint, release, soma_assets=tmp_path / "unused-assets")
    image = np.random.default_rng(8).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    box = [[39.5, 30.5, 45.0]]
    expected = original.encode(predictor.prepare(image, box))
    observed = predictor.encode(image, box)
    torch.testing.assert_close(observed.pose_tokens, expected.pose_tokens, atol=0, rtol=0)
    assert not predictor.geometry.body_model.runtime_is_initialized
