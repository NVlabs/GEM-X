# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Callback sampling, camera placement and training-state isolation."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gem.pose.api import body_native_config
from gem.pose.evaluation import visualization as vis
from gem.pose.heads.output import PoseHeadOutput
from gem.pose.training.recovery import _capture_rng


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(0.5)
        self.task = SimpleNamespace(name="body")
        self.pose_release_id = "vis-test"

    def forward(self, batch):
        count = len(batch["img"])
        # Exercise RNG isolation even for models/providers with incidental draws.
        torch.rand(2)
        return PoseHeadOutput(
            {
                "body_pose": torch.eye(3).expand(count, 76, 3, 3),
                "global_orient": torch.eye(3).expand(count, 1, 3, 3),
                "identity_coeffs": torch.zeros(count, 128),
                "scale_params": torch.ones(count, 56),
            },
            torch.tensor([2.0, 0.1, -0.05]).expand(count, -1),
            {},
            torch.zeros(count, 4),
        )


class TinyGeometry(nn.Module):
    faces = torch.tensor([[0, 2, 1]])

    def forward(self, parameters, *, transl=None):
        count = len(parameters["body_pose"])
        points = torch.tensor([[0.0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]]).expand(count, -1, -1)
        if transl is not None:
            points = points + transl[:, None]
        return {"vertices": points, "joints": points}


class Samples:
    def __len__(self):
        return 30

    def __getitem__(self, index):
        random.random()
        np.random.rand()
        return {
            "img": torch.zeros(3, 32, 32),
            "bbx_xys": torch.tensor([100.0, 110.0, 200.0]),
            "K_fullimg": torch.tensor([[500.0, 0, 140], [0, 530, 120], [0, 0, 1]]),
            "img_size": torch.tensor([280.0, 240.0]),
            "keypoints_2d": torch.tensor([[0.0, 0, 1], [float("nan"), 0, 0], [0.1, 0, 1]]),
            "source_record_id": str(index),
        }


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_model = TinyModel()
        self.objective = nn.Module()
        self.objective.geometry = TinyGeometry()
        self.objective.landmark_projector = None
        self.release = body_native_config()

    @property
    def device(self):
        return torch.device("cpu")


class RecorderRenderer:
    calls = []

    def __init__(self, width, height, *, backend):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def render(self, vertices, faces, intrinsics, extrinsic, *, background):
        self.calls.append((vertices.copy(), intrinsics.copy(), extrinsic.copy()))
        return background.copy()


def test_callback_seeded_sampling_and_training_state(monkeypatch, tmp_path):
    monkeypatch.setattr(vis, "MotionMeshRenderer", RecorderRenderer)
    RecorderRenderer.calls = []
    model = TinyModule().train()
    model.pose_model.dropout.eval()  # Preserve child modes, not just the root flag.
    model.pose_model.weight.grad = torch.tensor(7.0)
    trainer = SimpleNamespace(
        global_step=1,
        is_global_zero=True,
        loggers=[],
        default_root_dir=str(tmp_path),
        max_steps=240000,
    )
    callback = vis.PoseVisualizationCallback(
        Samples(),
        name="source",
        seed=23456,
        num_samples=4,
        seed_by_step=True,
        forward_batch_size=8,
        precision="32-true",
        output_dir=tmp_path,
    )
    modes = [module.training for module in model.modules()]
    before = _capture_rng()
    callback.on_train_batch_end(trainer, model, None, None, 0)
    after = _capture_rng()
    assert before["python"] == after["python"]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch"], after["torch"])
    assert [module.training for module in model.modules()] == modes
    assert model.pose_model.weight.grad.item() == 7
    record = json.loads((tmp_path / "source/step-000001/samples.json").read_text())
    assert record["indices"] == np.random.default_rng(23457).choice(30, 4, replace=False).tolist()
    assert len(RecorderRenderer.calls) == 4
    vertices, _, extrinsic = RecorderRenderer.calls[0]
    # 2*fx/(s*bbox_size)=2.5m. Renderer receives camera-space vertices once.
    np.testing.assert_allclose(vertices[:, 2], 2.5)
    np.testing.assert_array_equal(extrinsic, np.eye(4))
    assert (tmp_path / "source/step-000001/mesh_predictions.png").is_file()
    callback.on_train_batch_end(trainer, model, None, None, 0)
    assert len(RecorderRenderer.calls) == 4  # No duplicate callback at one step.
    trainer.global_step = 10000
    callback.on_train_batch_end(trainer, model, None, None, 0)
    second = json.loads((tmp_path / "source/step-010000/samples.json").read_text())
    assert second["indices"] == np.random.default_rng(33456).choice(30, 4, replace=False).tolist()
    restored = vis.PoseVisualizationCallback(Samples(), name="source")
    restored.load_state_dict(callback.state_dict())
    assert restored.last_step == 10000


def test_benchmark_cadence_and_natural_end(monkeypatch):
    callback = vis.PoseVisualizationCallback(
        Samples(),
        name="benchmark",
        first_step=10001,
        first_step_is_additional=False,
        visualize_on_train_end=True,
    )
    calls = []
    monkeypatch.setattr(
        callback, "_visualize", lambda trainer, module: calls.append(trainer.global_step)
    )
    trainer = SimpleNamespace(global_step=0, max_steps=240000)
    for step in [1, 10000, 10001, 20000, 20001]:
        trainer.global_step = step
        callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_end(trainer, None)
    trainer.global_step = 240000
    callback.on_train_end(trainer, None)
    assert calls == [10001, 20001, 240000]


def test_render_failure_restores_modes_and_rng(monkeypatch, tmp_path):
    callback = vis.PoseVisualizationCallback(Samples(), name="broken")
    model = TinyModule().train()
    state = torch.get_rng_state()

    def fail(*args):
        model.eval()
        torch.rand(3)
        raise ValueError("renderer failed")

    monkeypatch.setattr(callback, "_run", fail)
    trainer = SimpleNamespace(global_step=1, is_global_zero=True)
    with pytest.raises(RuntimeError, match="renderer failed"):
        callback._visualize(trainer, model)
    assert model.training and model.pose_model.training
    assert torch.equal(state, torch.get_rng_state())
    assert callback.last_step is None


def test_crop_camera_preserves_full_image_projection():
    camera = np.array([[700.0, 3, 321], [0, 650, 209], [0, 0, 1]])
    box = np.array([300.0, 200, 400])
    points = np.array([[0.2, 0.3, 2], [-0.1, 0, 1]])
    full = points @ camera.T
    expected = (full[:, :2] / full[:, 2:] - box[:2]) / box[2] + 0.5
    expected *= np.array([320, 256])
    crop = points @ vis.crop_intrinsics(camera, box, 320, 256).T
    np.testing.assert_allclose(crop[:, :2] / crop[:, 2:], expected, atol=1e-12)


def test_native_gt_and_dense_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(vis, "MotionMeshRenderer", RecorderRenderer)
    RecorderRenderer.calls = []
    model = TinyModule()
    # Non-unit rows ensure translation is added after landmark projection.
    model.objective.landmark_projector = torch.eye(3) * 0.5
    row = Samples()[0]
    row.update(
        soma_body_pose=torch.zeros(76, 3),
        soma_global_orient=torch.zeros(3),
        soma_identity_coeffs=torch.zeros(128),
        soma_scale_params=torch.ones(56),
        pred_cam_gt=torch.tensor([2.0, 0.1, -0.05]),
        landmarks_2d=torch.tensor([[0.0, 0, 1], [0.02, 0, 1], [0, 0.02, 1]]),
    )
    callback = vis.PoseVisualizationCallback(
        [row],
        name="gt",
        num_samples=1,
        forward_batch_size=1,
        precision="32-true",
        output_dir=tmp_path,
    )
    trainer = SimpleNamespace(
        global_step=1, is_global_zero=True, loggers=[], default_root_dir=str(tmp_path)
    )
    callback._visualize(trainer, model)
    assert len(RecorderRenderer.calls) == 2
    np.testing.assert_allclose(RecorderRenderer.calls[0][0], RecorderRenderer.calls[1][0])
    assert (tmp_path / "gt/step-000001/gt_mesh.png").is_file()
    assert (tmp_path / "gt/step-000001/sapiens_predictions.png").is_file()


def test_explicit_steps_and_resume_preview(monkeypatch):
    callback = vis.PoseVisualizationCallback(
        Samples(),
        name="benchmark",
        steps=[10001, 200000, 210000],
        visualize_on_resume=True,
    )
    calls = []
    monkeypatch.setattr(
        callback, "_visualize", lambda trainer, module: calls.append(trainer.global_step)
    )
    trainer = SimpleNamespace(global_step=0, ckpt_path=None)
    callback.on_train_start(trainer, None)
    trainer.global_step, trainer.ckpt_path = 5000, "saved.ckpt"
    callback.on_train_start(trainer, None)
    for step in [10000, 10001, 200000, 200001, 210000]:
        trainer.global_step = step
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert calls == [5000, 10001, 200000, 210000]
