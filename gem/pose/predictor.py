# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crop-to-mesh and RGB-to-mesh prediction with explicit camera/side semantics."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .camera import decode_camera, project_pose
from .conditioning.camera_depth import depth_parameter_to_legacy_camera
from .conditioning.fixed_depth_translation import (
    camera_scale_to_translation_z,
    crop_keypoints_to_full_image,
    solve_lateral_translation_full_perspective,
)
from .data.crop import prepare_pose_batch
from .heads.output import PoseHeadOutput


@dataclass
class PoseInferenceConfig:
    image_size: int = 256
    camera_mode: str = "sam3db_perspective"
    focal_length: float = 5000.0
    focal_length_mode: str = "fixed"
    camera_representation: str = "legacy"
    hand_mirror: str = "full_image"
    minimum_depth: float = 1e-4
    camera_log_value_bounds: tuple[float, float] = (-10.0, 10.0)

    def __post_init__(self):
        if self.camera_mode not in {"sam3db_perspective", "perspective", "weak_perspective"}:
            raise ValueError(f"Unknown camera mode {self.camera_mode!r}")
        if self.camera_representation not in {"legacy", "crop_2d", "log_lambda"}:
            raise ValueError(f"Unknown camera representation {self.camera_representation!r}")
        if self.hand_mirror not in {"full_image", "legacy_crop"}:
            raise ValueError(f"Unknown hand crop convention {self.hand_mirror!r}")
        if self.focal_length_mode not in {"fixed", "diagonal"}:
            raise ValueError(f"Unknown focal length mode {self.focal_length_mode!r}")


@dataclass
class PosePrediction:
    """Geometry in original camera axes, meters; 2D points in original pixels.

    Parameters describe the physical side. ``head`` retains canonical-left
    hand predictions and the original final query for downstream feature use.
    Camera translation is added exactly once to ``joints_camera``/vertices.
    Faces are B×F×3, including each hand side's correct vertex winding.
    """

    task: str
    parameters: dict[str, Tensor]
    camera_translation: Tensor
    joints_camera: Tensor
    vertices_camera: Tensor
    keypoints_2d: Tensor
    keypoints_crop: Tensor
    projection_valid: Tensor
    faces: Tensor
    right_hand: Tensor | None
    head: PoseHeadOutput
    pose_release_id: str | None
    intrinsics: Tensor | None


class PosePredictor(nn.Module):
    """Combine a parameter-only PoseModel with a task-specific geometry module.

    ``predict`` accepts RGB uint8 images plus caller-owned square boxes and
    pinhole intrinsics. ``predict_batch`` accepts normalized canonical crops;
    right-hand rows also carry their original bbox/K and ``right_hand`` mask.
    Detection, tracking and lens undistortion stay with the image input stages.
    """

    def __init__(self, model, geometry, config=None):
        super().__init__()
        self.model = model
        self.geometry = geometry
        if config is None:
            conditioning = model.conditioning
            inference = {}
            if model.task.name == "hand":
                from .api import hand_config

                inference = hand_config().inference
            config = PoseInferenceConfig(
                **inference,
                **(
                    {}
                    if conditioning is None
                    else {
                        "camera_mode": conditioning.camera_mode,
                        "focal_length": conditioning.focal_length,
                        "focal_length_mode": conditioning.focal_length_mode,
                    }
                ),
            )
        elif isinstance(config, dict):
            config = PoseInferenceConfig(**config)
        self.config = config
        if model.task.name == "hand":
            mode = getattr(model.task.head, "camera_depth_mode", "s")
            if mode != "log_lambda" or config.camera_representation != "log_lambda":
                raise ValueError("Hand prediction requires the current log_lambda architecture")
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _camera(self, batch):
        if self.config.camera_mode != "perspective":
            # Weak-perspective checkpoints also need a full-image camera in
            # the deployment predictor; their raw s/tx/ty is converted once.
            return batch["K_fullimg"].float()
        wh = batch["img_size"].float()
        focal = (
            wh.square().sum(-1).sqrt()
            if self.config.focal_length_mode == "diagonal"
            else wh.new_full((len(wh),), self.config.focal_length)
        )
        camera = torch.zeros(len(wh), 3, 3, device=wh.device)
        camera[:, 0, 0] = camera[:, 1, 1] = focal
        camera[:, :2, 2] = wh / 2
        camera[:, 2, 2] = 1
        return camera

    @torch.inference_mode()
    def predict_batch(self, batch):
        if self.training or self.model.training:
            raise RuntimeError("Pose prediction requires eval mode")
        head = self.model(batch)
        # Keep camera and geometry arithmetic in FP32 even when the neural
        # backbone/head are called under autocast by an inference service.
        with torch.autocast(device_type=head.pred_cam.device.type, enabled=False):
            return self._decode_prediction(head, batch)

    def _decode_prediction(self, head, batch):
        camera = self._camera(batch)
        raw_camera = head.pred_cam
        representation = self.config.camera_representation
        if self.model.task.name == "hand":
            raw_camera = depth_parameter_to_legacy_camera(
                raw_camera,
                batch["bbx_xys"],
                camera,
                self.model.task.head.camera_depth_mode,
                log_value_bounds=self.config.camera_log_value_bounds,
            )
            representation = "legacy"
        translation = decode_camera(
            raw_camera,
            batch["bbx_xys"],
            camera,
            representation=representation,
        )
        right = None
        if self.model.task.name == "hand":
            if self.model.kp2d_head is not None:
                translation = self._keypoint_translation(
                    head, batch, camera, raw_camera, translation
                )
            right = batch.get(
                "right_hand",
                torch.zeros(len(translation), device=translation.device, dtype=torch.bool),
            )
            geometry = self.geometry(head.parameters, right=right, transl=translation)
            translation = geometry["transl"]
            parameters, faces = geometry["parameters"], geometry["faces"]
        else:
            geometry = self.geometry(head.parameters, transl=translation)
            parameters = head.parameters
            faces = self.geometry.faces.expand(len(translation), -1, -1)
        bbox = batch.get("original_bbx_xys", batch["bbx_xys"])
        if right is not None and bool(right.any()):
            if "original_K_fullimg" not in batch or "original_bbx_xys" not in batch:
                raise ValueError(
                    "Canonical right-hand crops must include original_K_fullimg and original_bbx_xys"
                )
            if self.config.camera_mode == "perspective":
                # Undo the canonical camera reflection for synthetic cameras.
                from .tasks.hand import mirror_hand_camera

                _, unmirrored = mirror_hand_camera(
                    batch["bbx_xys"][right],
                    camera[right],
                    batch["img_size"][right, 0],
                    calibrated=self.config.hand_mirror == "full_image",
                )
                camera = camera.clone()
                camera[right] = unmirrored
            else:
                camera = batch["original_K_fullimg"].float()
        pixels, crop, valid = project_pose(
            geometry["joints"], camera, bbox, minimum_depth=self.config.minimum_depth
        )
        return PosePrediction(
            self.model.task.name,
            parameters,
            translation,
            geometry["joints"],
            geometry["vertices"],
            pixels,
            crop,
            valid,
            faces,
            right,
            head,
            self.model.pose_release_id,
            camera,
        )

    def _keypoint_translation(self, head, batch, camera, legacy_camera, fallback):
        """Use GA direct 2D support to solve XY at the predicted metric depth."""
        options = self.model.kp2d_translation
        direct = head.auxiliary["kp2d"]
        crop = direct["mu"].float()
        full = crop_keypoints_to_full_image(crop, batch["bbx_xys"].float())
        image_wh = batch["img_size"].float()
        point_mask = (
            (direct["presence"].float() >= options["presence_threshold"])
            & ((crop >= -0.5) & (crop < 0.5)).all(-1)
            & ((full >= 0) & (full < image_wh[:, None])).all(-1)
        )
        # Solve in the canonical-left frame, before mirroring a physical right hand.
        canonical = self.geometry(head.parameters)
        depth = camera_scale_to_translation_z(legacy_camera[:, 0], batch["bbx_xys"].float(), camera)
        solved = solve_lateral_translation_full_perspective(
            canonical["joints"],
            full,
            depth,
            camera,
            point_mask=point_mask,
            min_points=options["min_points"],
            min_depth=options["min_depth"],
            damping=options["damping"],
            max_condition_number=options["max_condition_number"],
        )
        head.auxiliary["translation_solve"] = {
            "valid": solved.valid,
            "num_points": solved.num_points,
            "condition_number": solved.condition_number,
        }
        return torch.where(
            solved.valid[:, None],
            torch.cat((solved.translation_xy, depth[:, None]), dim=-1),
            fallback,
        )

    def prepare(self, images, bboxes, intrinsics=None, *, right_hand=None):
        """Prepare crops and canonical/original camera metadata for this release."""
        if self.model.task.name != "hand" and right_hand is not None:
            raise ValueError("right_hand is only meaningful for the hand task")
        batch = prepare_pose_batch(
            images,
            bboxes,
            intrinsics,
            right_hand=right_hand,
            image_size=self.config.image_size,
            hand_mirror=self.config.hand_mirror,
            device=self.device,
        )
        batch["img"] = batch["img"].to(dtype=next(self.model.parameters()).dtype)
        return batch

    @torch.inference_mode()
    def encode(self, images, bboxes, intrinsics=None, *, right_hand=None):
        """Extract final query/parameters without FK or physical-side mirroring."""
        batch = self.prepare(images, bboxes, intrinsics, right_hand=right_hand)
        return self.model.encode(batch)

    @torch.inference_mode()
    def predict(self, images, bboxes, intrinsics=None, *, right_hand=None):
        batch = self.prepare(images, bboxes, intrinsics, right_hand=right_hand)
        return self.predict_batch(batch)

    def forward(self, images, bboxes, intrinsics=None, *, right_hand=None):
        return self.predict(images, bboxes, intrinsics, right_hand=right_hand)
