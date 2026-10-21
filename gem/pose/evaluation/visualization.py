# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic crop/mesh previews on caller-owned unaugmented samples."""

import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import default_collate

from gem.common.camera import compute_transl_full_cam
from gem.common.geometry.rotations import aa_to_rotmat
from gem.common.rendering import MotionMeshRenderer
from gem.pose.camera import project_pose
from gem.pose.evaluation.runner import _to_device
from gem.pose.predictor import PoseInferenceConfig, PosePredictor
from gem.pose.training.recovery import _capture_rng, _restore_rng


def crop_intrinsics(intrinsics, bbox, width, height):
    """Apply the square image crop to a full-image pinhole camera."""
    cx, cy, size = np.asarray(bbox)
    affine = np.array(
        [
            [width / size, 0, width * (0.5 - cx / size)],
            [0, height / size, height * (0.5 - cy / size)],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return affine @ np.asarray(intrinsics)


def _rgb(image):
    image = image.detach().float().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return (image * 255).clip(0, 255).astype(np.uint8)


def _keypoints(image, points, color, radius=3):
    image = image.copy()
    points = np.asarray(points)
    valid = np.isfinite(points).all(-1)
    if points.shape[-1] == 3:
        valid &= points[:, 2] >= 0.3
    height, width = image.shape[:2]
    for x, y in points[valid, :2]:
        x, y = (x + 0.5) * width, (y + 0.5) * height
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image


def _grid(panels, columns):
    panels = list(panels)
    columns = min(columns, len(panels))
    while len(panels) % columns:
        panels.append(np.zeros_like(panels[0]))
    return np.concatenate(
        [np.concatenate(panels[i : i + columns], axis=1) for i in range(0, len(panels), columns)],
        axis=0,
    )


class PoseVisualizationCallback(pl.Callback):
    """Render independently sampled images without changing training state.

    The dataset supplies normal Pose prediction crops. Native body training GT
    enables an additional GT mesh; ``landmarks_2d`` enables dense GT/prediction
    overlays using the training objective's landmark projector. The samples must
    be unaugmented. Benchmark datasets need no SOMA target parameters.

    All ranks enter the callback, but only rank zero loads images and renders.
    PNGs are always saved. W&B loggers receive the same grids when configured.
    Failed rendering is surfaced to every rank rather than silently dropping
    required training previews. No optimizer or training data cursor is touched.
    """

    def __init__(
        self,
        dataset,
        *,
        name,
        num_samples=4,
        seed=12345,
        every_n_steps=10000,
        steps=None,
        first_step=1,
        first_step_is_additional=True,
        seed_by_step=False,
        sort_indices=False,
        visualize_on_train_start=False,
        visualize_on_resume=False,
        visualize_on_train_end=False,
        forward_batch_size=64,
        precision="32-true",
        output_dir=None,
        render_backend="software",
        columns=None,
        evaluator=None,
        full_image=False,
        require_landmarks=False,
    ):
        super().__init__()
        if num_samples <= 0 or forward_batch_size < num_samples:
            raise ValueError("Visualization requires 0 < num_samples <= forward_batch_size")
        if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("Unsupported visualization precision")
        self.dataset, self.name = dataset, name
        self.num_samples, self.seed = num_samples, seed
        self.every_n_steps, self.first_step = every_n_steps, first_step
        self.steps = None if steps is None else set(steps)
        self.first_step_is_additional, self.seed_by_step = first_step_is_additional, seed_by_step
        self.sort_indices = sort_indices
        self.visualize_on_train_start = visualize_on_train_start
        self.visualize_on_resume = visualize_on_resume
        self.visualize_on_train_end = visualize_on_train_end
        self.forward_batch_size, self.precision = forward_batch_size, precision
        self.output_dir, self.render_backend = output_dir, render_backend
        self.columns = columns or num_samples
        self.evaluator, self.full_image = evaluator, full_image
        self.require_landmarks = require_landmarks
        self.last_step = None
        self._samples = None
        self._indices = None

    @property
    def state_key(self):
        return f"PoseVisualizationCallback/{self.name}"

    def state_dict(self):
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict):
        self.last_step = state_dict["last_step"]

    def on_train_start(self, trainer, pl_module):
        if self.visualize_on_train_start or (
            self.visualize_on_resume and trainer.ckpt_path and trainer.global_step > 0
        ):
            self._visualize(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        offset = 0 if self.first_step_is_additional else self.first_step or 0
        periodic = (
            self.every_n_steps and step >= offset and (step - offset) % self.every_n_steps == 0
        )
        due = (step == self.first_step or periodic) if self.steps is None else step in self.steps
        if step > 0 and due:
            self._visualize(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if (
            self.visualize_on_train_end
            and trainer.global_step > 0
            and trainer.global_step >= trainer.max_steps > 0
        ):
            self._visualize(trainer, pl_module)

    def _visualize(self, trainer, pl_module):
        step = trainer.global_step
        if step == self.last_step:
            return
        rng = _capture_rng()
        modes = [(module, module.training) for module in pl_module.modules()]
        error = [None]
        try:
            if trainer.is_global_zero:
                try:
                    self._run(trainer, pl_module, step)
                except Exception as exc:
                    # Rendering is rank-zero work; propagate failures before the
                    # next DDP collective rather than leaving other ranks waiting.
                    error[0] = f"{type(exc).__name__}: {exc}"
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError(f"Pose visualization {self.name}: {error[0]}")
            self.last_step = step
        finally:
            for module, training in modes:
                module.training = training
            _restore_rng(rng)

    @torch.no_grad()
    def _run(self, trainer, pl_module, step):
        if self._samples is None or self.seed_by_step:
            count = min(self.num_samples, len(self.dataset))
            seed = self.seed + step if self.seed_by_step else self.seed
            self._indices = (
                np.random.default_rng(seed).choice(len(self.dataset), count, replace=False).tolist()
            )
            if self.sort_indices:
                self._indices.sort()
            self._samples = [self.dataset[index] for index in self._indices]
        count = len(self._samples)
        # Preserve the source recipe's fixed inference batch, independent of
        # the number of displayed crops. Padding is never shown or evaluated.
        samples = self._samples + [self._samples[-1]] * (self.forward_batch_size - count)
        batch = _to_device(default_collate(samples), pl_module.device)
        release = pl_module.release
        camera = {
            name: value
            for name, value in (release.conditioning or {}).items()
            if name in {"camera_mode", "focal_length", "focal_length_mode"}
        }
        predictor = PosePredictor(
            pl_module.pose_model,
            pl_module.objective.geometry,
            PoseInferenceConfig(**{**camera, **release.inference}),
        )
        context = (
            nullcontext()
            if self.precision == "32-true"
            else torch.autocast(
                device_type=pl_module.device.type,
                dtype=torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16,
            )
        )
        with context:
            prediction = predictor.predict_batch(batch)
        panels = self._render_panels(pl_module, batch, prediction, count)
        directory = Path(self.output_dir or Path(trainer.default_root_dir) / "visualizations")
        directory = directory / self.name / f"step-{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        grids = {name: _grid(items, self.columns) for name, items in panels.items() if items}
        if self.require_landmarks and "sapiens_predictions" not in grids:
            raise ValueError("This visualization source requires visible landmark targets")
        for name, grid in grids.items():
            Image.fromarray(grid).save(directory / f"{name}.png")
        (directory / "samples.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "seed": self.seed + step if self.seed_by_step else self.seed,
                    "indices": self._indices,
                    "samples": [
                        str(row.get("source_record_id", row.get("sample_id", "")))
                        for row in self._samples
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                for name, grid in grids.items():
                    logger.log_image(
                        key=f"eval_vis/{self.name}/{name}",
                        images=[grid],
                        step=step,
                        caption=[f"Step {step} | {self.name} | green=GT, red=prediction"],
                    )

    def _render_panels(self, pl_module, batch, prediction, count):
        def cpu(value):
            return value.detach().float().cpu().numpy()

        panels = {
            name: []
            for name in (
                "input_crops",
                "mesh_predictions",
                "gt_mesh",
                "keypoint_predictions",
                "sapiens_predictions",
                "fullimg_mesh",
            )
        }
        geometry = pl_module.objective.geometry
        gt = None
        if "soma_body_pose" in batch:
            parameters = {
                "body_pose": aa_to_rotmat(batch["soma_body_pose"].float().reshape(-1, 3)).reshape(
                    -1, 76, 3, 3
                ),
                "global_orient": aa_to_rotmat(batch["soma_global_orient"].float())[:, None],
                "identity_coeffs": batch["soma_identity_coeffs"].float(),
                "scale_params": batch["soma_scale_params"].float(),
            }
            translation = compute_transl_full_cam(
                batch["pred_cam_gt"].float(), batch["bbx_xys"].float(), batch["K_fullimg"].float()
            )
            gt = geometry(parameters, transl=translation)
        landmarks = None
        if "landmarks_2d" in batch and pl_module.objective.landmark_projector is not None:
            points = torch.einsum(
                "kv,bvd->bkd",
                pl_module.objective.landmark_projector.float(),
                prediction.vertices_camera.float() - prediction.camera_translation[:, None],
            )
            points = points + prediction.camera_translation[:, None]
            _, landmarks, _ = project_pose(points, prediction.intrinsics, batch["bbx_xys"])
        pred_keypoints = prediction.keypoints_crop
        gt_keypoints = batch.get("keypoints_2d")
        if self.evaluator is not None:
            self.evaluator.to(pl_module.device)
            points, _, translation, intrinsics = self.evaluator._joint_pair(prediction, batch)
            pred_keypoints = self.evaluator._project_crop(
                points + translation[:, None], intrinsics, batch["bbx_xys"]
            )
            gt_keypoints = batch.get("benchmark_keypoints_2d")
        height, width = batch["img"].shape[-2:]
        with MotionMeshRenderer(width, height, backend=self.render_backend) as renderer:
            for index in range(count):
                image = _rgb(batch["img"][index])
                panels["input_crops"].append(image)
                camera = crop_intrinsics(
                    cpu(prediction.intrinsics[index]), cpu(batch["bbx_xys"][index]), width, height
                )
                faces = prediction.faces[index].detach().cpu().numpy()
                panels["mesh_predictions"].append(
                    renderer.render(
                        cpu(prediction.vertices_camera[index]),
                        faces,
                        camera,
                        np.eye(4),
                        background=image,
                    )
                )
                keypoints = image
                if gt_keypoints is not None:
                    keypoints = _keypoints(keypoints, cpu(gt_keypoints[index]), (0, 255, 0))
                panels["keypoint_predictions"].append(
                    _keypoints(keypoints, cpu(pred_keypoints[index]), (255, 0, 0))
                )
                if gt is not None:
                    camera = crop_intrinsics(
                        cpu(batch["K_fullimg"][index]), cpu(batch["bbx_xys"][index]), width, height
                    )
                    panels["gt_mesh"].append(
                        renderer.render(
                            cpu(gt["vertices"][index]), faces, camera, np.eye(4), background=image
                        )
                    )
                if landmarks is not None and bool((batch["landmarks_2d"][index, :, 2] > 0).any()):
                    image = _keypoints(image, cpu(batch["landmarks_2d"][index]), (0, 255, 0), 2)
                    panels["sapiens_predictions"].append(
                        _keypoints(image, cpu(landmarks[index]), (255, 0, 0), 2)
                    )
        if self.full_image:
            for index in range(count):
                with Image.open(batch["imgname"][index]) as stream:
                    image = np.asarray(stream.convert("RGB"))
                height, width = image.shape[:2]
                scale = min(1.0, 512 / height)
                width, height = round(width * scale), round(height * scale)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                camera = cpu(prediction.intrinsics[index]).copy()
                camera[:2] *= scale
                with MotionMeshRenderer(width, height, backend=self.render_backend) as renderer:
                    panels["fullimg_mesh"].append(
                        renderer.render(
                            cpu(prediction.vertices_camera[index]),
                            prediction.faces[index].detach().cpu().numpy(),
                            camera,
                            np.eye(4),
                            background=image,
                        )
                    )
            max_height = max(image.shape[0] for image in panels["fullimg_mesh"])
            max_width = max(image.shape[1] for image in panels["fullimg_mesh"])
            panels["fullimg_mesh"] = [
                np.pad(
                    image,
                    ((0, max_height - image.shape[0]), (0, max_width - image.shape[1]), (0, 0)),
                )
                for image in panels["fullimg_mesh"]
            ]
        return panels
