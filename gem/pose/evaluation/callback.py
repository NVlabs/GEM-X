# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic benchmark evaluation without changing the training RNG or modes."""

import json
from pathlib import Path

import lightning.pytorch as pl

from gem.pose.predictor import PoseInferenceConfig, PosePredictor
from gem.pose.training.recovery import _capture_rng, _restore_rng

from .runner import evaluate_dataset


class PoseEvaluationCallback(pl.Callback):
    """Evaluate an independent, prepared benchmark dataset on training ranks.

    Every rank evaluates its DistributedSampler shard; padding is deduplicated
    by sample_id. GT follows the evaluator's protocol, never the training loss.
    """

    def __init__(
        self,
        dataset,
        evaluator,
        *,
        name,
        every_n_steps=1000,
        steps=None,
        batch_size=16,
        num_workers=0,
        precision="32-true",
        evaluate_on_train_end=True,
        evaluate_on_resume=False,
        output_dir=None,
    ):
        super().__init__()
        self.dataset, self.evaluator, self.name = dataset, evaluator, name
        self.every_n_steps = every_n_steps
        self.steps = None if steps is None else set(steps)
        self.options = dict(batch_size=batch_size, num_workers=num_workers, precision=precision)
        self.evaluate_on_train_end = evaluate_on_train_end
        self.evaluate_on_resume = evaluate_on_resume
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.last_step = None
        self.last_result = None

    @property
    def state_key(self):
        return f"PoseEvaluationCallback/{self.name}"

    def state_dict(self):
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict):
        self.last_step = state_dict["last_step"]

    def _evaluate(self, trainer, pl_module):
        step = trainer.global_step
        if self.last_step == step:
            return
        random_state = _capture_rng()
        modes = [(module, module.training) for module in pl_module.modules()]
        release = pl_module.release
        camera = {
            name: value
            for name, value in (release.conditioning or {}).items()
            if name in {"camera_mode", "focal_length", "focal_length_mode"}
        }
        try:
            predictor = PosePredictor(
                pl_module.pose_model,
                pl_module.objective.geometry,
                PoseInferenceConfig(**{**camera, **release.inference}),
            )
            self.last_result = evaluate_dataset(
                predictor, self.evaluator, self.dataset, **self.options
            )
            evaluated = self.last_result["sample_count"]
            total = len(self.dataset)
            values = self.last_result["metrics"]
            values.update(
                eval_samples=evaluated,
                eval_total_samples=total,
                eval_sample_coverage=evaluated / total,
            )
            if values.get("camera_root_translation_requested"):
                full_dataset = evaluated == total
                values.update(
                    camera_root_translation_full_dataset=float(full_dataset),
                    camera_root_translation_reportable=float(
                        full_dataset
                        and values["camera_root_translation_valid_samples"] == evaluated
                    ),
                )
            metrics = {
                f"eval/{self.name}/{name}": value
                for name, value in self.last_result["metrics"].items()
                if value is not None
            }
            if trainer.is_global_zero and metrics:
                # Resume and natural-end hooks cannot use Lightning's step
                # logging API. Direct logger calls also retain the actual
                # evaluation step when it is off the train logging cadence.
                for logger in trainer.loggers:
                    logger.log_metrics(metrics, step=step)
                    logger.save()
            if trainer.is_global_zero and self.output_dir is not None:
                directory = self.output_dir / self.name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"step-{step:06d}.json").write_text(
                    json.dumps({"step": step, **self.last_result}, indent=2, allow_nan=False) + "\n"
                )
            self.last_step = step
        finally:
            for module, training in modes:
                module.training = training
            _restore_rng(random_state)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        due = (
            self.every_n_steps and step % self.every_n_steps == 0
            if self.steps is None
            else step in self.steps
        )
        if step > 0 and due:
            self._evaluate(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        # on_fit_start materializes normal geometry buffers and configures
        # compiled modules before an inference-mode benchmark can use them.
        if self.evaluate_on_resume and trainer.ckpt_path and trainer.global_step > 0:
            self._evaluate(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        completed = (trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps) or (
            trainer.max_epochs > 0 and trainer.current_epoch >= trainer.max_epochs
        )
        if self.evaluate_on_train_end and completed and trainer.global_step > 0:
            self._evaluate(trainer, pl_module)
