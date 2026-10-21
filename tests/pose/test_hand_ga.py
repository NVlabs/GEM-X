# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Hand GA neural branch and checkpoint behavior; no source/assets needed."""

import math

import pytest
import torch
from transformers import DINOv3ViTConfig

from gem.pose.api import build_pose_model, hand_config, load_pose_checkpoint
from gem.pose.heads.rletr_occ_head import RLEOccReadout


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_direct_coordinates_remain_unbounded_and_presence_uses_spatial_evidence():
    readout = RLEOccReadout(d_model=8, num_joints=25)
    with torch.no_grad():
        readout.coord.linear.weight.zero_()
        readout.coord.linear.bias.copy_(torch.tensor([1.25, -0.75]))
        readout.sigma.linear.weight.zero_()
        readout.sigma.linear.bias.copy_(torch.logit(torch.tensor([0.2, 0.8])))
        readout.map_q.weight.zero_()
        readout.map_k.weight.zero_()
    tokens = torch.randn(2, 25, 8)
    memory = torch.randn(2, 16, 8)
    output = readout(tokens, memory)
    torch.testing.assert_close(output["mu"], torch.tensor([1.25, -0.75]).expand(2, 25, 2))
    torch.testing.assert_close(output["sigma"], torch.tensor([0.2, 0.8]).expand(2, 25, 2))
    torch.testing.assert_close(output["evidence"], torch.full((2, 25, 16), math.log(2)))
    mass = 16 * math.log(2)
    torch.testing.assert_close(output["mass"], torch.full((2, 25), mass))
    torch.testing.assert_close(output["pres_logit"], torch.full((2, 25), math.log(mass + 1e-8)))
    torch.testing.assert_close(output["presence"], torch.full((2, 25), mass / (mass + 1)))
    larger_memory = readout(tokens, memory.repeat(1, 2, 1))
    assert bool((larger_memory["presence"] > output["presence"]).all())


def tiny_ga_release():
    release = hand_config()
    config = DINOv3ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_register_tokens=2,
        use_gated_mlp=True,
        hidden_act="silu",
    )
    release.backbone = {"config": config.to_dict()}
    release.head.update(
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=4,
        transformer_dim_head=8,
        transformer_mlp_dim=64,
        transformer_context_dim=32,
    )
    release.kp2d_translation.update(d_model=16, decoder_depth=1, num_heads=4, feat_hw=[2, 2])
    return release


def input_batch():
    return {
        "img": torch.randn(2, 3, 32, 32),
        "bbx_xys": torch.tensor([[330.0, 200, 190], [240, 190, 160]]),
        "K_fullimg": torch.tensor([[[500.0, 12, 310], [0, 720, 225], [0, 0, 1]]]).expand(2, -1, -1),
        "img_size": torch.tensor([[640.0, 480], [640, 480]]),
        "aug_rot": torch.zeros(2),
    }


def test_separate_2d_branch_receives_raw_features_and_backpropagates():
    model = build_pose_model(tiny_ga_release()).train()
    backbone_outputs, direct_inputs = [], []
    backbone_hook = model.backbone.register_forward_hook(
        lambda _module, _inputs, output: backbone_outputs.append(output)
    )
    direct_hook = model.kp2d_head.register_forward_pre_hook(
        lambda _module, inputs: direct_inputs.append(inputs[0])
    )
    output = model(input_batch())
    backbone_hook.remove()
    direct_hook.remove()
    assert direct_inputs[0] is backbone_outputs[0]
    direct = output.auxiliary["kp2d"]
    assert direct["mu"].shape == (2, 25, 2)
    assert direct["evidence"].shape == (2, 25, 4)
    assert output.token.shape == (2, 32)
    assert output.parameters["scale_params"].shape == (2, 27)
    assert model.task.head.num_kp2d_queries == 0
    loss = direct["mu"].square().mean() + direct["sigma"].mean() + direct["presence"].mean()
    loss.backward()
    assert model.kp2d_head.token_producer.query.weight.grad is not None
    assert all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in model.kp2d_head.parameters()
    )
    backbone_gradients = [
        value.grad for value in model.backbone.parameters() if value.grad is not None
    ]
    assert backbone_gradients and any(bool(value.abs().sum() > 0) for value in backbone_gradients)
    # Independent direct2D supervision must not route through the SOMA query decoder.
    assert all(value.grad is None for value in model.task.head.parameters())


def legacy_checkpoint(model):
    state = {
        (
            "hand_head." + key.removeprefix("task.head.") if key.startswith("task.head.") else key
        ): value
        for key, value in model.state_dict().items()
    }
    # The learned density belongs to training, not the image parameter predictor.
    state["kp2d_outcoord_loss.rle_loss.flow.test_parameter"] = torch.ones(2)
    return {"state_dict": state}


def test_legacy_ga_checkpoint_preserves_direct_head_and_omits_training_density(tmp_path):
    release = tiny_ga_release()
    model = build_pose_model(release).eval()
    checkpoint = tmp_path / "hand_ga.ckpt"
    torch.save(legacy_checkpoint(model), checkpoint)
    restored = load_pose_checkpoint(checkpoint, release)
    assert not restored.training
    assert list(restored.state_dict()) == list(model.state_dict())
    batch = input_batch()
    with torch.inference_mode():
        before, after = model(batch), restored(batch)
    torch.testing.assert_close(before.pred_cam, after.pred_cam, rtol=0, atol=0)
    torch.testing.assert_close(before.token, after.token, rtol=0, atol=0)
    for key in before.auxiliary["kp2d"]:
        torch.testing.assert_close(
            before.auxiliary["kp2d"][key], after.auxiliary["kp2d"][key], rtol=0, atol=0
        )


def test_missing_direct_head_weights_fail_strict_loading(tmp_path):
    release = tiny_ga_release()
    checkpoint = legacy_checkpoint(build_pose_model(release))
    checkpoint["state_dict"].pop("kp2d_head.token_producer.query.weight")
    path = tmp_path / "missing_query.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="kp2d_head.token_producer.query.weight"):
        load_pose_checkpoint(path, release)
