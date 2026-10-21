# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Lightning training for SOMA image/video motion estimation."""

import lightning.pytorch as pl
import torch

from gem.motion.frontend.joint_sets import SOMA29_NOFINGER, get_soma_observation_schema
from gem.motion.frontend.observation import normalize_keypoints
from gem.motion.frontend.time import TIME_INPUT_KEYS, ReferenceMotionTime
from gem.motion.pipeline import MotionPipeline
from gem.motion.training.losses import ImageT1Loss
from gem.motion.training.preparation import TrainingPreparation
from gem.motion.training.video_losses import VideoLoss
from gem.pose.release_ids import canonical_pose_release_id


class MotionModel(pl.LightningModule):
    """Train one motion network on native video and B2/native-GT T1 batches.

    Fixed XLO/MID body models and image landmark assets are injected by the
    caller. They remain differentiable with respect to pose parameters but
    their own weights are frozen and excluded from training checkpoints.
    """

    def __init__(
        self,
        pipeline: MotionPipeline,
        image_loss: ImageT1Loss,
        video_loss: VideoLoss | None = None,
        *,
        observation_schema=SOMA29_NOFINGER,
        observation_augmentation="common2d_v1",
        video_loss_options=None,
        preparation_options=None,
        time_options=None,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_milestones=(188500, 259188),
        lr_gamma=0.5,
        pose_release_id=None,
    ):
        super().__init__()
        self.schema = get_soma_observation_schema(observation_schema)
        if pipeline.denoiser3d.obs_num_joints != self.schema.num_joints:
            raise ValueError("Training observation schema does not match the network width")
        if pipeline.endecoder.body_model is None:
            raise ValueError("Motion training requires the codec's fixed SOMA body model")
        self.pipeline = pipeline
        self.image_loss = image_loss
        self.video_loss = (
            video_loss if video_loss is not None else VideoLoss(**(video_loss_options or {}))
        )
        video_loss_options = {
            "weights": dict(self.video_loss.weights),
            "motion_field_weights": self.video_loss.motion_field_weights,
            "static_velocity_threshold": self.video_loss.static_velocity_threshold,
            "static_fps_by_source": self.video_loss.static_fps_by_source,
            "root_motion_supervision": self.video_loss.root_motion_supervision,
        }
        self.pipeline.endecoder.body_model.requires_grad_(False)
        self.image_loss.mid_body.requires_grad_(False)
        self.preparation = TrainingPreparation(
            pipeline.endecoder,
            self.schema,
            observation_augmentation=observation_augmentation,
            **(preparation_options or {}),
        )
        self.motion_time = ReferenceMotionTime(**(time_options or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_milestones = tuple(lr_milestones)
        self.lr_gamma = lr_gamma
        self.pose_release_id = pose_release_id
        self.attempts_completed = 0
        self.branch_counts = {}
        self.save_hyperparameters(
            {
                "observation_schema": observation_schema,
                "observation_augmentation": observation_augmentation,
                "video_loss_options": video_loss_options,
                "preparation_options": preparation_options,
                "time_options": time_options,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "lr_milestones": tuple(lr_milestones),
                "lr_gamma": lr_gamma,
                "pose_release_id": pose_release_id,
            }
        )

    def prepare_timing(self, inputs, *, train):
        if "positions" in inputs and inputs["positions"] is not None:
            inputs = {**inputs, "time_positions": inputs["positions"]}
        prepared = self.motion_time.prepare_pipeline(inputs, train=train, postproc=False)
        return {**prepared, "positions": self.motion_time.clock(prepared)}

    def forward(self, inputs):
        """Raw inference from prepared detector observations and timing."""
        return self.pipeline(self.prepare_timing(inputs, train=False))

    def forward_video(self, batch, *, attempt_step=None):
        inputs = self.preparation.prepare_video(batch, **self._draw_context(attempt_step))
        inputs = self.prepare_timing(inputs, train=True)
        output = self.pipeline.forward_train(inputs)
        return self.video_loss(output, inputs, codec=self.pipeline.endecoder)

    def forward_image(self, batch, *, attempt_step=None):
        conditions = self.preparation.prepare_image_conditions(
            batch, **self._draw_context(attempt_step)
        )
        features = batch["f_imgseq"]
        positions = torch.zeros(features.shape[:2], device=features.device, dtype=features.dtype)
        output = self.pipeline.denoiser3d(length=batch["length"], positions=positions, **conditions)
        return self.image_loss(output, batch["image_labels"], endecoder=self.pipeline.endecoder)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Raw full-sequence inference with the original validation frame mask.

        Data adapters own detector selection and benchmark GT. Evaluation has
        no training augmentation, modality dropout or contact postprocessing.
        Geometry and metrics are delegated to the configured evaluation callback.
        """
        if len(batch["length"]) != 1:
            raise ValueError("Motion validation expects one full sequence per batch")
        if batch["kp2d"].shape[-2] != self.schema.num_joints:
            raise ValueError(
                "Validation detector joints do not match the Motion observation schema"
            )
        obs = normalize_keypoints(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs = obs.masked_fill(~batch["mask"].bool()[..., None, None], 0)
        inputs = {
            name: batch[name]
            for name in ("length", "bbx_xys", "K_fullimg", "cam_angvel", "f_imgseq")
        }
        inputs["obs"] = obs
        inputs.update({name: batch[name] for name in (*TIME_INPUT_KEYS, "meta") if name in batch})
        outputs = self(inputs)
        # The retained evaluation callbacks consume unbatched sequence params.
        for coordinates in ("incam", "global"):
            params = {
                name: value[0] for name, value in outputs[f"pred_body_params_{coordinates}"].items()
            }
            outputs[f"pred_body_params_{coordinates}"] = params
            outputs[f"pred_soma_params_{coordinates}"] = params
        return outputs

    def _draw_context(self, attempt_step):
        return {
            "attempt_step": self.attempts_completed if attempt_step is None else attempt_step,
            "global_rank": self.global_rank,
            "world_size": self._trainer.world_size if self._trainer is not None else 1,
        }

    def training_step(self, batch, batch_idx):
        branch = batch.get("training_branch", "video")
        if branch == "image":
            outputs = self.forward_image(batch)
            source = batch.get("image_source", "image")
        elif branch == "video":
            outputs = self.forward_video(batch)
            source = "video"
        else:
            raise ValueError(f"Unknown training branch {branch!r}")
        if self._trainer is not None:
            options = dict(
                on_step=True, on_epoch=True, batch_size=len(batch["length"]), sync_dist=True
            )
            self.log("train/loss", outputs["loss"], **options)
            for name, value in outputs.items():
                if name.endswith("_loss"):
                    self.log(f"train/{branch}_{name}", value, **options)
        # Batch-end checkpoint callbacks run before the module's corresponding
        # hook. Publish consumption here, before returning to automatic backward,
        # so ordinary batch-end saves see this attempt (including AMP skips).
        self.attempts_completed += 1
        self.branch_counts[source] = self.branch_counts.get(source, 0) + 1
        if self._trainer is not None and self.trainer.datamodule is not None:
            self.trainer.datamodule.mark_consumed(self.attempts_completed)
        # Image losses do not use contact/world heads. Returning their unused
        # prediction tensors makes DDP's unused-parameter discovery incorrect.
        return {"loss": outputs["loss"]}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.pipeline.denoiser3d.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def _fixed_asset_key(name):
        return name.startswith(("pipeline.endecoder.body_model.", "image_loss.mid_body."))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["gem_motion_training"] = {
            "attempts_completed": self.attempts_completed,
            "branch_counts": dict(self.branch_counts),
            "observation_schema": self.schema.name,
            "pose_release_id": self.pose_release_id,
        }
        state = checkpoint["state_dict"]
        for name in list(state):
            if self._fixed_asset_key(name):
                del state[name]

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["gem_motion_training"]
        if state["observation_schema"] != self.schema.name:
            raise ValueError("Resume checkpoint uses a different observation schema")
        if canonical_pose_release_id(state["pose_release_id"]) != canonical_pose_release_id(
            self.pose_release_id
        ):
            raise ValueError("Resume checkpoint uses a different pose feature release")
        self.attempts_completed = state["attempts_completed"]
        self.branch_counts = dict(state["branch_counts"])
        for name, value in self.state_dict().items():
            if self._fixed_asset_key(name):
                checkpoint["state_dict"][name] = value
