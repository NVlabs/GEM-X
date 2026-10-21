# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Pose training state and the native body optimization path."""

from dataclasses import asdict

import lightning.pytorch as pl
import torch

from gem.pose.api import PoseReleaseConfig
from gem.pose.geometry.body import BodyGeometry

from .optimization import ConstantThenCosineLR, build_pose_optimizer


class PoseTrainingModule(pl.LightningModule):
    """Own optimization while keeping the parameter model usable for inference.

    Geometry is frozen but differentiable in its inputs. Its external assets
    and the landmark projector are not stored as learned checkpoint tensors.
    Task subclasses own their objective contract and optimization behavior.
    """

    def __init__(self, pose_model, objective, *, release, optimizer=None):
        super().__init__()
        self.release = (
            release if isinstance(release, PoseReleaseConfig) else PoseReleaseConfig(**release)
        )
        if self.release.task != self.task_name:
            raise ValueError(f"{type(self).__name__} requires a {self.task_name} release")
        # Checkpoint loading returns an inference model. Training must restore
        # dropout/positional augmentation as well as keeping parameters trainable.
        self.pose_model = pose_model.train()
        self.objective = objective
        self.optimizer_options = dict(optimizer or {})
        self.save_hyperparameters(
            {
                "release": asdict(self.release),
                "optimizer": self.optimizer_options,
                "loss_weights": dict(self.objective.weights),
            },
            logger=False,
        )

    def forward(self, batch):
        return self.pose_model(batch)

    def _step(self, batch, stage):
        output = self.objective(self(batch), batch)
        if not bool(torch.isfinite(output["loss"]).all()):
            raise FloatingPointError(f"Pose-{self.task_name} objective is not finite")
        self.log_dict(
            {f"{stage}/{name}": value for name, value in output["losses"].items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=len(batch["img"]),
            sync_dist=stage == "val",
        )
        return output

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")["loss"]

    def _initialize_geometry(self):
        if isinstance(self.objective.geometry, BodyGeometry):
            # Frozen tensors can participate in later input differentiation.
            # Standalone validation must not construct them as inference tensors.
            with torch.inference_mode(False):
                _ = self.objective.geometry.faces

    def on_fit_start(self):
        # Materialize before first-batch RNG restoration on resume.
        self._initialize_geometry()

    def on_validation_start(self):
        self._initialize_geometry()

    def configure_optimizers(self):
        return build_pose_optimizer(self.pose_model, **self.optimizer_options)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["gem_pose_training"] = {
            "version": 1,
            "release": asdict(self.release),
            **self._objective_metadata(),
        }

    def on_load_checkpoint(self, checkpoint):
        metadata = checkpoint.get("gem_pose_training")
        if metadata is None:
            raise ValueError(
                "Resume needs a native Pose training checkpoint; use initialization for legacy weights"
            )
        if asdict(PoseReleaseConfig(**metadata["release"])) != asdict(self.release):
            raise ValueError("Resume release configuration differs from the saved Pose model")
        saved = {name: metadata.get(name) for name in self._objective_metadata()}
        if "pelvis_ids" in saved:
            saved["pelvis_ids"] = tuple(saved["pelvis_ids"])
        if saved != self._objective_metadata():
            raise ValueError("Resume objective differs from the saved Pose training configuration")


class PoseBodyTrainingModule(PoseTrainingModule):
    """Body recipe: automatic BF16 optimization and an optional cosine tail."""

    task_name = "body"

    def __init__(self, pose_model, objective, *, release, optimizer=None, scheduler=None):
        super().__init__(pose_model, objective, release=release, optimizer=optimizer)
        self.scheduler_options = None if scheduler is None else dict(scheduler)
        self.save_hyperparameters({"scheduler": self.scheduler_options}, logger=False)

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        if self.scheduler_options is None:
            return optimizer
        scheduler = ConstantThenCosineLR(optimizer, **self.scheduler_options)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _objective_metadata(self):
        return {
            "loss_weights": dict(self.objective.weights),
            "loss_reduction": self.objective.reduction,
            "minimum_depth": self.objective.minimum_depth,
            "pelvis_ids": tuple(self.objective.pelvis_ids),
            "finger_weight": self.objective.finger_weight,
            "scheduler": self.scheduler_options,
        }
