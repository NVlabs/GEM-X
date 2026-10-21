# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EgoHand manual optimization with post-unscale gradient clipping."""

import torch
import torch.distributed as dist

from gem.pose.geometry.hand import HandGeometry

from .module import PoseTrainingModule
from .optimization import build_pose_optimizer


class PoseHandTrainingModule(PoseTrainingModule):
    """Train canonical-left MHR hands with the selected FP16 recipe.

    Forward failures still run backward to complete DDP's reduction, then
    discard gradients. AMP overflow instead reaches GradScaler's normal skip
    path. The data callback consumes both kinds of batch; optimizer attempts
    and actually applied Adam updates can therefore have different counts.
    """

    task_name = "hand"

    def __init__(self, pose_model, objective, *, release, optimizer=None, grad_clip_val=1.0):
        options = {"fused": None, **dict(optimizer or {})}
        super().__init__(pose_model, objective, release=release, optimizer=options)
        self.automatic_optimization = False
        self.grad_clip_val = float(grad_clip_val)
        self.save_hyperparameters({"grad_clip_val": self.grad_clip_val}, logger=False)

    def _objective_metadata(self):
        metadata = {
            "loss_weights": dict(self.objective.weights),
            "loss_reduction": self.objective.reduction,
            "root_joint_ids": list(self.objective.root_joint_ids),
            "camera_mode": self.objective.camera_mode,
            "camera_depth_mode": self.objective.camera_depth_mode,
            "camera_scalar_loss_type": self.objective.camera_scalar_loss_type,
            "projection_denominator_floor": self.objective.projection_denominator_floor,
            "mask_predicted_depth": self.objective.mask_predicted_depth,
            "mask_nonfinite_gt_projection": self.objective.mask_nonfinite_gt_projection,
            "grad_clip_val": self.grad_clip_val,
            "optimization": "manual_hand",
            "camera_scalar_loss_std": self.objective.camera_scalar_loss_std,
            "camera_log_value_bounds": list(self.objective.camera_log_value_bounds),
            "kp2d_translation": self.objective.kp2d_translation,
        }
        return metadata

    def configure_optimizers(self):
        return build_pose_optimizer(
            self.pose_model,
            additional_head_parameters=self.objective.kp2d_outcoord_loss.parameters(),
            **self.optimizer_options,
        )

    def _initialize_geometry(self):
        geometry = self.objective.geometry
        if isinstance(geometry, HandGeometry):
            with torch.inference_mode(False):
                geometry._layer("left", geometry._device_anchor.device)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        output = self.objective(self(batch), batch)
        loss = output["loss"]
        nonfinite = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
        skip_forward = bool(nonfinite.item())

        optimizer.zero_grad()
        with torch.autocast(device_type=loss.device.type, enabled=False):
            self.manual_backward(loss)
        self.log(
            "train/nonfinite_forward_skip",
            float(skip_forward),
            on_step=True,
            on_epoch=False,
        )
        if skip_forward:
            optimizer.zero_grad()
            return

        optimizer.step()
        self.log_dict(
            {f"train/{name}": value.detach() for name, value in output["losses"].items()},
            on_step=True,
            on_epoch=True,
            batch_size=len(batch["img"]),
        )

    def _amp_found_inf(self, optimizer):
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is None or not scaler.is_enabled():
            return None
        # GradScaler exposes no public post-unscale overflow flag. This is the
        # same state consumed by its immediately following step(optimizer).
        values = list(scaler._found_inf_per_device(optimizer).values())
        if not values:
            raise RuntimeError("Hand gradient handling requires GradScaler post-unscale state")
        return (torch.stack([value.detach().float() for value in values]).sum() > 0).to(
            dtype=torch.int32
        )

    def on_before_optimizer_step(self, optimizer):
        overflow = self._amp_found_inf(optimizer)
        if overflow is not None:
            any_overflow, every_overflow = overflow.clone(), overflow.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(any_overflow, op=dist.ReduceOp.MAX)
                dist.all_reduce(every_overflow, op=dist.ReduceOp.MIN)
            if bool(any_overflow.item()) != bool(every_overflow.item()):
                raise ValueError("AMP overflow state differs across ranks")
            if bool(any_overflow.item()):
                self.log("train/grad_norm_nonfinite", 1.0, on_step=True, on_epoch=False)
                self.log("train/amp_overflow_skip", 1.0, on_step=True, on_epoch=False)
                # Do not clip/mutate gradients; GradScaler owns this skip.
                return

        self.log("train/grad_norm_nonfinite", 0.0, on_step=True, on_epoch=False)
        self.log("train/amp_overflow_skip", 0.0, on_step=True, on_epoch=False)
        if self.grad_clip_val <= 0:
            return
        parameters = list(self.pose_model.task.head.parameters()) + list(
            self.pose_model.backbone.parameters()
        )
        parameters.extend(self.pose_model.kp2d_head.parameters())
        parameters.extend(self.objective.kp2d_outcoord_loss.parameters())
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.grad_clip_val, error_if_nonfinite=False
        )
        nonfinite = (~torch.isfinite(norm.detach())).to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
        if bool(nonfinite.item()):
            self.log("train/grad_norm_nonfinite", 1.0, on_step=True, on_epoch=False)
            raise ValueError("Gradient norm is non-finite")
        self.log("train/grad_norm", norm, on_step=True, on_epoch=True)
