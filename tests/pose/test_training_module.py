# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CPU Lightning training, native checkpoint inference and consumed-cursor resume."""

import importlib.util
import random
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import default_collate
from transformers import DINOv3ViTConfig

from gem.pose.api import body_native_config, build_pose_model, load_pose_checkpoint
from gem.pose.data.training import PoseTrainingDataModule, PreparedPoseDataset
from gem.pose.training.losses import BodyPoseObjective
from gem.pose.training.module import PoseBodyTrainingModule
from gem.pose.training.recovery import PoseTrainingState, _capture_rng


class TinyGeometry(nn.Module):
    """Differentiable geometry fixture with fixed, nonpersistent external assets."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(812)
        self.register_buffer(
            "template", torch.randn(77, 3, generator=generator) * 0.1, persistent=False
        )
        self.register_buffer(
            "shape_basis", torch.randn(128, 231, generator=generator) * 0.002, persistent=False
        )
        self.register_buffer(
            "scale_basis", torch.randn(56, 231, generator=generator) * 0.004, persistent=False
        )

    def forward(self, parameters):
        rotations = torch.cat((parameters["global_orient"], parameters["body_pose"]), 1).float()
        local = self.template + (
            parameters["identity_coeffs"].float() @ self.shape_basis
            + (parameters["scale_params"].float() - 1) @ self.scale_basis
        ).reshape(-1, 77, 3)
        joints = torch.einsum("bjik,bjk->bji", rotations, local)
        return {"joints": joints, "vertices": torch.cat((joints, joints[:, :23] + 0.01), 1)}


def tiny_release():
    release = body_native_config(pose_release_id="tiny-body-training-test")
    backbone = DINOv3ViTConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=256,
        patch_size=32,
        num_register_tokens=2,
        drop_path_rate=0.2,
        use_gated_mlp=True,
        hidden_act="silu",
        key_bias=False,
        layerscale_value=1e-5,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
        layer_norm_eps=1e-5,
    )
    release.backbone = {"config": backbone.to_dict()}
    release.head.update(
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=2,
        transformer_dim_head=8,
        transformer_mlp_dim=48,
        transformer_context_dim=16,
        transformer_dropout=0.15,
        # The body query must remain present; test stochastic training with
        # attention/feed-forward dropout and backbone drop path instead.
        transformer_emb_dropout=0.0,
    )
    return release


def prepared_samples():
    path = Path(__file__).resolve().parents[2] / "scripts/make_pose_example.py"
    spec = importlib.util.spec_from_file_location("pose_training_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = module.make_samples(6, seed=21)
    for sample in samples:
        sample["keypoints_2d"][:, 2] = 1
        sample["keypoints_3d"][:, 3] = 1
        sample["landmarks_2d"][:, 2] = 1
    return samples


def build_training(release, scheduler=None):
    projector = torch.zeros(308, 100)
    projector[torch.arange(308), torch.arange(308) % 100] = 1
    objective = BodyPoseObjective(TinyGeometry(), projector)
    return PoseBodyTrainingModule(
        build_pose_model(release),
        objective,
        release=release,
        optimizer={"lr": 1e-4, "backbone_lr": 5e-5, "fused": False},
        scheduler=scheduler,
    )


class TraceBatches(pl.Callback):
    def __init__(self):
        self.rows = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        cursor = batch["data_cursor"]
        self.rows.append(
            {
                "epoch": int(cursor["epoch"][0]),
                "batch": int(cursor["batch_index"][0]),
                "indices": cursor["sample_index"].tolist(),
                "ids": list(batch["sample_id"]),
                "python": random.random(),
                "numpy": float(np.random.random()),
                "torch": torch.rand(3).clone(),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.rows[-1]["loss"] = float(outputs["loss"])
        self.rows[-1]["consumed"] = deepcopy(trainer.datamodule.state_dict())


def train(tmp_path, samples, release, *, workers, seed, resume=None, scheduler=None):
    pl.seed_everything(seed, workers=True)
    model = build_training(release, scheduler=scheduler)
    initial = {name: value.clone() for name, value in model.pose_model.state_dict().items()}
    data = PoseTrainingDataModule(
        PreparedPoseDataset(samples), batch_size=2, shuffle=True, seed=71, num_workers=workers
    )
    trace = TraceBatches()
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        filename="step-{step}",
        auto_insert_metric_name=False,
        every_n_train_steps=1,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=7,
        max_epochs=4,
        callbacks=[PoseTrainingState(), trace, checkpoint],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume, weights_only=False)
    return {
        "trainer": trainer,
        "model": model,
        "data": data,
        "trace": trace.rows,
        "checkpoint": checkpoint,
        "initial": initial,
        "state": deepcopy(model.state_dict()),
        "optimizer": deepcopy(trainer.optimizers[0].state_dict()),
        "rng": _capture_rng(),
    }


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, tuple | list):
        assert len(first) == len(second)
        for value, other in zip(first, second):
            assert_tree_equal(value, other)
    else:
        assert first == second


def test_eval_initialization_enters_training_mode_and_backpropagates_backbone_and_head():
    release = tiny_release()
    pose_model = build_pose_model(release).eval()
    assert all(not module.training for module in pose_model.modules())
    model = PoseBodyTrainingModule(
        pose_model,
        BodyPoseObjective(TinyGeometry()),
        release=release,
        optimizer={"fused": False},
    )
    assert model.pose_model is pose_model
    assert all(module.training for module in pose_model.backbone.modules())
    assert all(module.training for module in pose_model.task.head.modules())
    batch = default_collate([PreparedPoseDataset(prepared_samples())[0]])
    before = torch.get_rng_state().clone()
    output = model(batch)
    # The tiny release has real dropout and positional augmentation enabled.
    assert not torch.equal(before, torch.get_rng_state())
    loss = model.objective(output, batch)["loss"]
    loss.backward()
    for component in (pose_model.backbone, pose_model.task.head):
        gradients = [
            parameter.grad for parameter in component.parameters() if parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)


@pytest.mark.parametrize("workers", [0, 2])
def test_full_training_resume_crosses_epochs_and_matches_model_optimizer_rng_and_samples(
    tmp_path, workers
):
    torch.set_num_threads(2)
    samples, release = prepared_samples(), tiny_release()
    continuous = train(tmp_path / "continuous", samples, release, workers=workers, seed=135)
    assert continuous["trainer"].global_step == 7
    assert [(row["epoch"], row["batch"]) for row in continuous["trace"]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
    ]
    assert any(
        not torch.equal(value, continuous["initial"][name])
        for name, value in continuous["model"].pose_model.state_dict().items()
    )
    for cut, expected_cursor in ((2, (0, 2)), (3, (1, 0))):
        checkpoint_path = tmp_path / "continuous" / f"step-{cut}.ckpt"
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert saved["global_step"] == cut
        assert saved["optimizer_states"] and saved["optimizer_states"][0]["state"]
        assert "gem_pose_training" in saved
        assert tuple(saved["gem_pose_training"]["pelvis_ids"]) == (0,)
        assert saved["gem_pose_training"]["finger_weight"] == 1.0
        data_state = saved["PoseTrainingDataModule"]
        assert (data_state["epoch"], data_state["consumed_batches"]) == expected_cursor
        assert all(name.startswith("pose_model.") for name in saved["state_dict"])
        assert not any(
            "objective" in name or "projector" in name or "geometry" in name
            for name in saved["state_dict"]
        )
        assert "rng_by_rank" in saved["callbacks"]["PoseTrainingState"]
        resumed = train(
            tmp_path / f"resumed-{cut}",
            samples,
            release,
            workers=workers,
            seed=900 + cut,
            resume=checkpoint_path,
        )
        assert resumed["trainer"].global_step == 7
        assert_tree_equal(resumed["trace"], continuous["trace"][cut:])
        assert_tree_equal(resumed["state"], continuous["state"])
        assert_tree_equal(resumed["optimizer"], continuous["optimizer"])
        assert_tree_equal(resumed["rng"], continuous["rng"])
        assert_tree_equal(resumed["data"].state_dict(), continuous["data"].state_dict())

    # A full training checkpoint is directly usable by the public parameter loader.
    loaded = load_pose_checkpoint(continuous["checkpoint"].last_model_path, release=None)
    batch = default_collate([PreparedPoseDataset(samples)[0], PreparedPoseDataset(samples)[1]])
    continuous["model"].eval()
    with torch.inference_mode():
        expected, actual = continuous["model"](batch), loaded(batch)
    torch.testing.assert_close(actual.token, expected.token, atol=0, rtol=0)
    torch.testing.assert_close(actual.pred_cam, expected.pred_cam, atol=0, rtol=0)
    for field in expected.parameters:
        torch.testing.assert_close(
            actual.parameters[field], expected.parameters[field], atol=0, rtol=0
        )
    assert not loaded.training and loaded.pose_release_id == release.pose_release_id


def test_cosine_tail_resumes_through_lightning_with_model_and_adam(tmp_path):
    torch.set_num_threads(2)
    schedule = {"constant_steps": 2, "decay_steps": 4, "eta_min": 1e-7}
    samples, release = prepared_samples(), tiny_release()
    continuous = train(
        tmp_path / "continuous", samples, release, workers=0, seed=135, scheduler=schedule
    )
    checkpoint = tmp_path / "continuous" / "step-3.ckpt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["lr_schedulers"][0]["last_epoch"] == 3
    resumed = train(
        tmp_path / "resumed",
        samples,
        release,
        workers=0,
        seed=999,
        resume=checkpoint,
        scheduler=schedule,
    )
    assert_tree_equal(resumed["state"], continuous["state"])
    assert_tree_equal(resumed["optimizer"], continuous["optimizer"])
    assert_tree_equal(resumed["trace"], continuous["trace"][3:])
    assert [group["lr"] for group in resumed["optimizer"]["param_groups"]] == [1e-7, 1e-7]
    assert resumed["trainer"].lr_scheduler_configs[0].scheduler.last_epoch == 7

    changed = build_training(release, scheduler={**schedule, "decay_steps": 8})
    with pytest.raises(ValueError, match="Resume objective"):
        changed.on_load_checkpoint(saved)


def test_old_constant_lr_checkpoint_remains_valid_without_scheduler_metadata():
    model = build_training(tiny_release())
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    del checkpoint["gem_pose_training"]["scheduler"]
    model.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "kind", ["conditioning", "inference", "representation", "conflicting_modes"]
)
def test_factory_rejects_camera_mismatch_before_constructing_or_loading_models(monkeypatch, kind):
    from gem.pose.training import factory

    release = body_native_config()
    if kind in ("conditioning", "conflicting_modes"):
        release.conditioning["camera_mode"] = "perspective"
    if kind == "inference":
        release.inference["camera_mode"] = "perspective"
    if kind == "representation":
        release.inference["camera_representation"] = "crop_2d"
    if kind == "conflicting_modes":
        release.inference["camera_mode"] = "sam3db_perspective"

    def no_model(*args, **kwargs):
        raise AssertionError("Camera mismatch reached model construction or checkpoint loading")

    monkeypatch.setattr(factory, "build_pose_model", no_model)
    monkeypatch.setattr(factory, "load_pose_checkpoint", no_model)
    for initialization in (None, "unused-weights.ckpt"):
        with pytest.raises(ValueError, match="calibrated sam3db_perspective and legacy"):
            factory.build_body_training_model(
                soma_assets="unused-assets", release=release, initialization=initialization
            )


def test_validation_warmup_creates_normal_assets_that_can_be_reused_for_training(monkeypatch):
    from gem.pose.training import module as training_module

    class LazyGeometry(TinyGeometry):
        def __init__(self):
            super().__init__()
            self.register_buffer("runtime_asset", None, persistent=False)
            self.face_tensor = None

        @property
        def faces(self):
            if self.runtime_asset is None:
                self.runtime_asset = torch.ones(1)
                self.face_tensor = torch.tensor([[0, 1, 2]])
            return self.face_tensor

        def forward(self, parameters):
            return {
                name: value * self.runtime_asset
                for name, value in super().forward(parameters).items()
            }

    monkeypatch.setattr(training_module, "BodyGeometry", LazyGeometry)
    model = build_training(tiny_release())
    geometry = LazyGeometry()
    model.objective.geometry = geometry
    assert geometry.runtime_asset is None
    with torch.inference_mode():
        model.on_validation_start()
        assert torch.is_inference_mode_enabled()
    assert not torch.is_inference(geometry.runtime_asset)
    assert not torch.is_inference(geometry.face_tensor)
    asset = geometry.runtime_asset
    model.on_fit_start()
    assert geometry.runtime_asset is asset
    batch = default_collate([PreparedPoseDataset(prepared_samples())[0]])
    loss = model.objective(model(batch), batch)["loss"]
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.pose_model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)


@pytest.mark.parametrize("field,value", [("pelvis_ids", (1,)), ("finger_weight", 0.5)])
def test_resume_rejects_changed_pelvis_or_finger_objective(field, value):
    model = build_training(tiny_release())
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    altered = deepcopy(checkpoint)
    altered["gem_pose_training"][field] = value
    with pytest.raises(ValueError, match="Resume objective differs"):
        model.on_load_checkpoint(altered)
