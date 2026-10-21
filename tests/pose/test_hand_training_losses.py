# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable direct-keypoint supervision, learned density and native recovery."""

import pytest
import torch

from gem.pose.api import build_pose_model, load_pose_checkpoint
from gem.pose.training.factory import build_hand_training_model
from gem.pose.training.hand_losses import HandPoseObjective
from gem.pose.training.hand_module import PoseHandTrainingModule

from . import test_hand_training_module as training_fixture
from .hand_training_fixtures import HandFixtureGeometry, hand_case

STD = 0.3735116511306386
OPTIONS = {
    "enabled": True,
    "topology": "separate",
    "feat_hw": [16, 16],
    "min_points": 4,
    "min_depth": 1e-4,
    "damping": 1e-6,
    "max_condition_number": 1e6,
    "negative_presence_weight": 1.0,
}


def objective(reduction="sum", options=None):
    return HandPoseObjective(
        HandFixtureGeometry(),
        reduction=reduction,
        camera_depth_mode="log_lambda",
        camera_scalar_loss_type="smooth_l1",
        camera_scalar_loss_std=STD,
        camera_log_value_bounds=(-3.5, 3.0),
        kp2d_translation=OPTIONS if options is None else options,
    )


def case_inputs(case="normal"):
    head, batch = hand_case()
    raw = head.pred_cam.detach().clone()
    raw[:, 0] = torch.log(2.0 / raw[:, 0])
    head.pred_cam = raw.requires_grad_()
    batch["has_hand_params"].fill_(1)
    batch["has_cam_gt"].fill_(1)
    generator = torch.Generator().manual_seed(550)
    mu = torch.randn(3, 25, 2, generator=generator) * 0.12
    evidence = torch.rand(3, 25, 256, generator=generator) + 0.1
    head.auxiliary["kp2d"] = {
        "mu": mu.requires_grad_(),
        "sigma": torch.full_like(mu, 0.2, requires_grad=True),
        "evidence": evidence.requires_grad_(),
        "mass": evidence.detach().sum(-1).requires_grad_(),
        "pres_logit": torch.zeros(3, 25, requires_grad=True),
        "presence": torch.full((3, 25), 0.5, requires_grad=True),
    }
    if case == "outside":
        batch["pred_cam_gt"][0, 1] += 5.0
        batch["bbx_xys"][1, 0] = -300
    elif case == "masked":
        batch["pred_cam_gt"][0] = float("nan")
        batch["has_cam_gt"][0] = 0
        batch["has_hand_params"][1] = 0
        batch["keypoints_2d"][2, 3:, 2] = 0
    elif case == "bounds":
        with torch.no_grad():
            head.pred_cam[0, 0] = -12
            head.pred_cam[1, 0] = 8
    return head, batch


def test_density_initialization_preserves_rng_and_presence_does_not_gate_training_solve():
    before = torch.get_rng_state()
    loss = objective()
    assert torch.equal(torch.get_rng_state(), before)
    head, batch = case_inputs()
    first = loss(head, batch)
    head.auxiliary["kp2d"]["presence"] = torch.zeros_like(head.auxiliary["kp2d"]["presence"])
    head.auxiliary["kp2d"]["sigma"] = head.auxiliary["kp2d"]["sigma"] * 20
    second = loss(head, batch)
    for name in ("camera_translation", "camera_solve_point_mask"):
        torch.testing.assert_close(
            first["predictions"][name], second["predictions"][name], rtol=0, atol=0
        )
    assert second["losses"]["loss_kp2d_rle"] != first["losses"]["loss_kp2d_rle"]


def tiny_ga_release():
    release = training_fixture.tiny_release()
    release.head.update(
        camera_depth_mode="log_lambda",
        init_camera_scalar=-1.8199350824986398,
        camera_scalar_std=STD,
        ray_cond_dim=8,
    )
    release.ray_pe_num_bands = 2
    release.kp2d_translation = {
        **OPTIONS,
        "num_joints": 25,
        "d_model": 16,
        "decoder_depth": 1,
        "num_heads": 2,
        "feat_hw": [8, 8],
    }
    release.inference = {
        "hand_mirror": "full_image",
        "camera_representation": "log_lambda",
        "camera_log_value_bounds": [-3.5, 3.0],
    }
    return release


def tiny_training():
    release = tiny_ga_release()
    return PoseHandTrainingModule(
        build_pose_model(release),
        objective(options=release.kp2d_translation),
        release=release,
        optimizer={"lr": 1e-4, "backbone_lr": 5e-5},
    )


def test_ga_native_recovery_restores_density_optimizer_rng_and_inference_ignores_only_loss(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training_fixture, "build_training", tiny_training)
    rows = training_fixture.samples(4)
    for row in rows:
        row["img_size"] = torch.tensor([640.0, 480.0])
    full = training_fixture.train(tmp_path / "full", rows, steps=3)
    resume = training_fixture.train(
        tmp_path / "resume", rows, steps=3, seed=88, resume=tmp_path / "full/step-1.ckpt"
    )
    for name in ("state", "optimizer", "rng"):
        training_fixture.assert_tree_equal(full[name], resume[name])
    training_fixture.assert_tree_equal(full["trace"].rows[1:], resume["trace"].rows)
    for prefix in (
        "pose_model.backbone.",
        "pose_model.task.head.",
        "pose_model.kp2d_head.",
        "objective.kp2d_outcoord_loss.",
    ):
        assert any(
            not torch.equal(value, full["initial"][name])
            for name, value in full["state"].items()
            if name.startswith(prefix)
        )
    model = full["model"]
    optimizer = full["trainer"].optimizers[0]
    groups = {group["name"]: {id(p) for p in group["params"]} for group in optimizer.param_groups}
    assert all(id(p) in groups["head"] for p in model.pose_model.kp2d_head.parameters())
    assert all(id(p) in groups["head"] for p in model.objective.kp2d_outcoord_loss.parameters())
    assert all(p in optimizer.state for p in model.objective.kp2d_outcoord_loss.parameters())
    loaded = load_pose_checkpoint(full["checkpoint"].last_model_path)
    batch = torch.utils.data.default_collate(rows[:1])
    model.eval()
    with torch.no_grad():
        ours, saved = loaded(batch), model(batch)
    training_fixture.assert_tree_equal(ours.auxiliary["kp2d"], saved.auxiliary["kp2d"])


def test_hand_resume_requires_the_direct_keypoint_release():
    model = training_fixture.build_training()
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    checkpoint["gem_pose_training"]["release"].pop("kp2d_translation")
    with pytest.raises(ValueError, match="Resume release configuration differs"):
        model.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("native", [False, True])
def test_finetune_loads_complete_source_or_native_density_and_rejects_missing_tensor(
    tmp_path, native
):
    torch.set_num_threads(2)
    trained = tiny_training()
    with torch.no_grad():
        next(trained.objective.kp2d_outcoord_loss.parameters()).add_(0.15)
    state = trained.state_dict()
    if native:
        checkpoint = {"state_dict": state}
        trained.on_save_checkpoint(checkpoint)
        prefix = "objective.kp2d_outcoord_loss."
    else:
        state = {
            name.removeprefix("pose_model.")
            .replace("task.head.", "hand_head.")
            .removeprefix("objective."): value
            for name, value in state.items()
        }
        checkpoint = {"state_dict": state}
        prefix = "kp2d_outcoord_loss."
    path = tmp_path / "initialization.ckpt"
    torch.save(checkpoint, path)
    loaded = build_hand_training_model(
        soma_assets=tmp_path / "not-needed-for-loading",
        release=trained.release,
        initialization=path,
    )
    training_fixture.assert_tree_equal(
        trained.pose_model.state_dict(), loaded.pose_model.state_dict()
    )
    training_fixture.assert_tree_equal(
        trained.objective.kp2d_outcoord_loss.state_dict(),
        loaded.objective.kp2d_outcoord_loss.state_dict(),
    )
    del state[prefix + "flow.translation.0.0.weight"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_hand_training_model(
            soma_assets=tmp_path / "not-needed-for-loading",
            release=trained.release,
            initialization=path,
        )


def test_density_and_direct_head_participate_in_unscaled_clip_and_resume_semantics(monkeypatch):
    torch.set_num_threads(2)
    model = tiny_training()
    optimizer = model.configure_optimizers()
    monkeypatch.setattr(model, "_amp_found_inf", lambda optimizer: None)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    captured = []

    def clip(parameters, *args, **kwargs):
        captured.extend(parameters)
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    model.on_before_optimizer_step(optimizer)
    assert {id(p) for p in captured} == {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert len(captured) == len({id(p) for p in captured})
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.objective.camera_scalar_loss_std *= 2
    with pytest.raises(ValueError, match="Resume objective differs"):
        model.on_load_checkpoint(checkpoint)
