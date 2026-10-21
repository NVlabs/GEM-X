# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Hand optimization, consumed batches and real Lightning CPU resume."""

import importlib.util
import random
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.precision import MixedPrecision
from torch import nn
from torch.utils.data import default_collate
from transformers import DINOv3ViTConfig

from gem.pose.api import build_pose_model, hand_training_config, load_pose_checkpoint
from gem.pose.data.hand_training import PreparedHandPoseDataset
from gem.pose.data.training import PoseTrainingDataModule
from gem.pose.training.hand_losses import HandPoseObjective
from gem.pose.training.hand_module import PoseHandTrainingModule
from gem.pose.training.recovery import PoseTrainingState, _capture_rng


class TinyHandGeometry(nn.Module):
    """Fixed differentiable fixture; this does not substitute for real SOMA tests."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(517)
        self.register_buffer(
            "template", torch.randn(25, 3, generator=generator) * 0.03, persistent=False
        )
        self.register_buffer(
            "identity_basis", torch.randn(5, 75, generator=generator) * 0.002, persistent=False
        )
        self.register_buffer(
            "scale_basis", torch.randn(26, 75, generator=generator) * 0.002, persistent=False
        )

    def forward(self, parameters):
        poses = torch.cat((parameters["wrist_orient"], parameters["hand_pose"]), dim=1).float()
        scales = parameters["scale_params"].float()
        local = self.template + (
            parameters["identity_coeffs"].float() @ self.identity_basis
            + scales[:, 1:] @ self.scale_basis
        ).reshape(-1, 25, 3)
        joints = torch.einsum("bnij,bnj->bni", poses, local) * scales[:, :1, None]
        return {"joints": joints, "vertices": torch.cat((joints, joints + 0.001), dim=1)}


class InfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("inf")


class ControlledObjective(HandPoseObjective):
    """Inject only the two real skip conditions, after a normal objective."""

    def __init__(self):
        super().__init__(TinyHandGeometry(), kp2d_translation=tiny_release().kp2d_translation)
        self.backward_calls = 0

    def forward(self, output, batch):
        result = super().forward(output, batch)
        if batch.get("bad_forward", torch.tensor(False)).any():
            result["loss"] = result["loss"] * float("nan")
        if batch.get("overflow_backward", torch.tensor(False)).any():
            result["loss"] = InfiniteBackward.apply(result["loss"])
        if result["loss"].requires_grad:
            result["loss"].register_hook(self._count_backward)
        return result

    def _count_backward(self, gradient):
        self.backward_calls += 1
        return gradient


def tiny_release():
    release = hand_training_config()
    release.pose_release_id = "tiny-hand-training-test"
    config = DINOv3ViTConfig(
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
    release.backbone = {"config": config.to_dict()}
    release.head.update(
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=2,
        transformer_dim_head=8,
        transformer_mlp_dim=48,
        transformer_context_dim=16,
        transformer_dropout=0.15,
    )
    release.kp2d_translation.update(d_model=16, decoder_depth=1, num_heads=2, feat_hw=[8, 8])
    return release


def samples(count=6):
    path = Path(__file__).resolve().parents[2] / "scripts/make_hand_pose_example.py"
    spec = importlib.util.spec_from_file_location("hand_training_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.make_samples(count, seed=37)
    for row in rows:
        row.update(bad_forward=False, overflow_backward=False)
    return rows


def build_training():
    release = tiny_release()
    return PoseHandTrainingModule(
        build_pose_model(release).eval(),
        ControlledObjective(),
        release=release,
        optimizer={"lr": 1e-4, "backbone_lr": 5e-5},
    )


class Trace(pl.Callback):
    def __init__(self):
        self.rows = []
        self.first_after_state = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        cursor = batch["data_cursor"]
        self.rows.append(
            {
                "ids": list(batch["sample_id"]),
                "epoch": cursor["epoch"].tolist(),
                "batch": cursor["batch_index"].tolist(),
                "indices": cursor["sample_index"].tolist(),
                "python": random.random(),
                "numpy": float(np.random.random()),
                "torch": torch.rand(3),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.rows[-1]["global_step"] = trainer.global_step
        self.rows[-1]["data"] = deepcopy(trainer.datamodule.state_dict())
        if self.first_after_state is None:
            self.first_after_state = deepcopy(pl_module.state_dict())


def train(tmp_path, rows, *, workers=0, seed=41, resume=None, steps=7, batch_size=2, amp=False):
    torch.set_num_threads(2)
    pl.seed_everything(seed, workers=True)
    model = build_training()
    initial = deepcopy(model.state_dict())
    data = PoseTrainingDataModule(
        PreparedHandPoseDataset(rows),
        batch_size=batch_size,
        num_workers=workers,
        seed=71,
        shuffle=not amp and not any(row["bad_forward"] for row in rows),
    )
    trace = Trace()
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path,
        filename="step-{step}",
        auto_insert_metric_name=False,
        every_n_train_steps=1,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    options = {}
    if amp:
        # Real Lightning mixed-precision plugin and real CPU GradScaler exercise
        # unscale/hook/skip ordering without a GPU or a pretend scaler object.
        options["plugins"] = MixedPrecision(
            "16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu", init_scale=8)
        )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=steps,
        max_epochs=5,
        callbacks=[PoseTrainingState(), trace, checkpoint],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        deterministic=True,
        **options,
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume, weights_only=False)
    return {
        "model": model,
        "trainer": trainer,
        "data": data,
        "trace": trace,
        "initial": initial,
        "state": deepcopy(model.state_dict()),
        "optimizer": deepcopy(trainer.optimizers[0].state_dict()),
        "rng": _capture_rng(),
        "checkpoint": checkpoint,
    }


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_tree_equal(first[name], second[name])
    elif isinstance(first, tuple | list):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("workers", [0, 2])
def test_hand_full_training_resume_matches_epochs_data_optimizer_rng_and_public_loader(
    tmp_path, workers
):
    rows = samples()
    full = train(tmp_path / "full", rows, workers=workers)
    assert full["trainer"].global_step == 7
    assert all(module.training for module in full["model"].pose_model.modules())
    assert full["optimizer"]["param_groups"][0]["fused"] is None
    assert {int(state["step"]) for state in full["optimizer"]["state"].values()} == {7}
    assert any(
        not torch.equal(value, full["initial"][name]) for name, value in full["state"].items()
    )
    for cut, cursor in ((2, (0, 2)), (3, (1, 0))):
        path = tmp_path / "full" / f"step-{cut}.ckpt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        data = saved["PoseTrainingDataModule"]
        assert (data["epoch"], data["consumed_batches"]) == cursor
        assert all(
            name.startswith(("pose_model.", "objective.kp2d_outcoord_loss."))
            for name in saved["state_dict"]
        )
        assert saved["gem_pose_training"]["optimization"] == "manual_hand"
        restored = train(tmp_path / f"resume{cut}", rows, workers=workers, seed=900, resume=path)
        for key in ("state", "optimizer", "rng"):
            assert_tree_equal(restored[key], full[key])
        assert_tree_equal(restored["trace"].rows, full["trace"].rows[cut:])
        assert_tree_equal(restored["data"].state_dict(), full["data"].state_dict())

    loaded = load_pose_checkpoint(full["checkpoint"].last_model_path, release=None)
    batch = default_collate([PreparedHandPoseDataset(rows)[0]])
    full["model"].eval()
    with torch.inference_mode():
        actual, expected = loaded(batch), full["model"](batch)
    torch.testing.assert_close(actual.token, expected.token, rtol=0, atol=0)
    torch.testing.assert_close(actual.pred_cam, expected.pred_cam, rtol=0, atol=0)
    assert_tree_equal(actual.parameters, expected.parameters)


def test_nonfinite_forward_still_backpropagates_consumes_data_and_does_not_attempt_update(tmp_path):
    rows = samples(3)
    rows[0]["bad_forward"] = True
    run = train(tmp_path, rows, steps=2, batch_size=1)
    assert run["model"].objective.backward_calls == 3
    assert [row["global_step"] for row in run["trace"].rows] == [0, 1, 2]
    assert_tree_equal(run["trace"].first_after_state, run["initial"])
    assert {int(state["step"]) for state in run["optimizer"]["state"].values()} == {2}
    assert run["data"].data_epoch == 1 and run["data"].consumed_batches == 0


def test_real_cpu_amp_overflow_skips_update_and_restores_scaler_with_native_checkpoint(tmp_path):
    rows = samples(3)
    rows[0]["overflow_backward"] = True
    full = train(tmp_path / "full", rows, steps=3, batch_size=1, amp=True)
    assert [row["global_step"] for row in full["trace"].rows] == [1, 2, 3]
    assert_tree_equal(full["trace"].first_after_state, full["initial"])
    assert {int(state["step"]) for state in full["optimizer"]["state"].values()} == {2}
    checkpoint_path = tmp_path / "full/step-1.ckpt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["MixedPrecision"]["scale"] == 4
    assert saved["PoseTrainingDataModule"]["consumed_batches"] == 1
    assert not saved["optimizer_states"][0]["state"]
    resumed = train(
        tmp_path / "resume", rows, steps=3, batch_size=1, amp=True, seed=900, resume=checkpoint_path
    )
    for key in ("state", "optimizer", "rng"):
        assert_tree_equal(resumed[key], full[key])
    assert_tree_equal(resumed["trace"].rows, full["trace"].rows[1:])
    assert_tree_equal(
        resumed["trainer"].precision_plugin.scaler.state_dict(),
        full["trainer"].precision_plugin.scaler.state_dict(),
    )


def test_clip_uses_unscaled_global_norm_and_non_amp_nonfinite_gradient_raises(monkeypatch):
    model = build_training()
    optimizer = model.configure_optimizers()
    monkeypatch.setattr(model, "_amp_found_inf", lambda optimizer: None)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    for parameter in model.pose_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.on_before_optimizer_step(optimizer)
    norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.double().norm() for parameter in model.pose_model.parameters()])
    )
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=2e-6, rtol=0)
    next(model.pose_model.parameters()).grad.fill_(float("inf"))
    with pytest.raises(ValueError, match="Gradient norm is non-finite"):
        model.on_before_optimizer_step(optimizer)


def test_rank_amp_overflow_mismatch_raises_before_clipping(monkeypatch):
    from gem.pose.training import hand_module

    model = build_training()
    monkeypatch.setattr(model, "_amp_found_inf", lambda optimizer: torch.tensor(0))
    monkeypatch.setattr(hand_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(hand_module.dist, "is_initialized", lambda: True)

    def all_reduce(value, op):
        value.fill_(int(op == hand_module.dist.ReduceOp.MAX))

    monkeypatch.setattr(hand_module.dist, "all_reduce", all_reduce)
    with pytest.raises(ValueError, match="AMP overflow state differs across ranks"):
        model.on_before_optimizer_step(model.configure_optimizers())


def test_other_rank_nonfinite_forward_still_backpropagates_and_discards_local_gradients(
    monkeypatch,
):
    from gem.pose.training import hand_module

    model = build_training()
    optimizer = model.configure_optimizers()
    before = deepcopy(model.state_dict())
    monkeypatch.setattr(model, "optimizers", lambda: optimizer)
    monkeypatch.setattr(model, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(hand_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(hand_module.dist, "is_initialized", lambda: True)

    def peer_nonfinite(value, op):
        assert op == hand_module.dist.ReduceOp.MAX
        assert int(value) == 0  # This rank's ordinary objective is finite.
        value.fill_(1)

    monkeypatch.setattr(hand_module.dist, "all_reduce", peer_nonfinite)
    batch = default_collate([PreparedHandPoseDataset(samples())[0]])
    model.training_step(batch, 0)
    assert model.objective.backward_calls == 1
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.pose_model.parameters())
    assert_tree_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    "field,value", [("grad_clip_val", 0), ("root_joint_ids", [1]), ("mask_predicted_depth", False)]
)
def test_resume_rejects_changed_hand_optimization_or_loss_contract(field, value):
    model = build_training()
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    checkpoint["gem_pose_training"][field] = value
    with pytest.raises(ValueError, match="Resume objective differs"):
        model.on_load_checkpoint(checkpoint)


def test_validation_initializes_only_left_hand_without_inference_tensors(monkeypatch):
    from gem.pose.training import hand_module

    class LazyGeometry(TinyHandGeometry):
        def __init__(self):
            super().__init__()
            self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
            self.register_buffer("runtime_asset", None, persistent=False)
            self.sides = []

        def _layer(self, side, device):
            self.sides.append(side)
            if self.runtime_asset is None:
                self.runtime_asset = torch.ones(1, device=device)

        def forward(self, parameters):
            return {
                key: value * self.runtime_asset
                for key, value in super().forward(parameters).items()
            }

    monkeypatch.setattr(hand_module, "HandGeometry", LazyGeometry)
    model = build_training()
    geometry = LazyGeometry()
    model.objective.geometry = geometry
    with torch.inference_mode():
        model.on_validation_start()
    assert not torch.is_inference(geometry.runtime_asset)
    model.on_fit_start()
    assert geometry.sides == ["left", "left"]
    batch = default_collate([PreparedHandPoseDataset(samples())[0]])
    model.objective(model(batch), batch)["loss"].backward()
    grads = [p.grad for p in model.pose_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
