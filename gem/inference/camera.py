# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-OpenMDW-1.1
# Apache-2.0 except for restore_source_camera_predictions, which retains
# its function-level OpenMDW-1.1 notice below.

"""Optional raster/camera adapters around the unchanged Pose predictor."""

from dataclasses import replace

import numpy as np
import torch

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.fisheye import perspective_crop
from gem.pose.data.crop import IMAGENET_MEAN, IMAGENET_STD
from gem.pose.tasks.hand import mirror_hand_camera


def predict_square_pixels(predictor, image, boxes, intrinsics, *, right_hand):
    """Infer with square pixels, then express 2D results in the source raster.

    Resizing changes neither the physical camera axes nor the predicted 3D
    units. The inverse affine also handles releases with synthetic intrinsics.
    """
    transform = SquarePixelTransform.from_intrinsics(
        intrinsics.numpy(), image.shape[1], image.shape[0]
    )
    if transform.is_identity:
        return predictor.predict(image, boxes, intrinsics, right_hand=right_hand)
    prediction = predictor.predict(
        transform.resize_image(image),
        torch.as_tensor(transform.transform_boxes(boxes.numpy()), dtype=boxes.dtype),
        torch.as_tensor(transform.intrinsics, dtype=intrinsics.dtype),
        right_hand=right_hand,
    )
    pixels = prediction.keypoints_2d
    scale = pixels.new_tensor([transform.scale_x, transform.scale_y])
    pixels = (pixels + 0.5) / scale - 0.5
    source_boxes = boxes.to(pixels.device)
    crop = (pixels - source_boxes[:, None, :2]) / source_boxes[:, None, 2:]
    inverse = prediction.intrinsics.new_tensor(
        [
            [1 / transform.scale_x, 0, 0.5 / transform.scale_x - 0.5],
            [0, 1 / transform.scale_y, 0.5 / transform.scale_y - 0.5],
            [0, 0, 1],
        ]
    )
    with torch.autocast(device_type=prediction.intrinsics.device.type, enabled=False):
        source_intrinsics = inverse @ prediction.intrinsics
    return replace(
        prediction,
        keypoints_2d=pixels,
        keypoints_crop=crop,
        intrinsics=source_intrinsics,
    )


def _perspective_batch(predictor, image, boxes, right_hand, camera):
    """Rectified crops are normalized directly, without a second affine crop."""
    size = predictor.config.image_size
    crops, cameras, virtual_boxes, rotations = [], [], [], []
    accepted, skipped = [], {}
    for index, box in enumerate(boxes.numpy()):
        try:
            crop, intrinsics, virtual_box, rotation = perspective_crop(image, box, camera, size)
        except (FloatingPointError, ValueError) as error:
            # Edge detections can lie outside the calibrated forward lens
            # domain. Preserve other hands and frames, as in the source pipeline.
            skipped[index] = str(error)
            continue
        accepted.append(index)
        crops.append(crop)
        cameras.append(intrinsics)
        virtual_boxes.append(virtual_box)
        rotations.append(rotation)
    if not accepted:
        return None, None, accepted, skipped
    right_hand = right_hand[accepted]
    original_boxes = torch.as_tensor(np.stack(virtual_boxes), dtype=torch.float32)
    original_cameras = torch.as_tensor(np.stack(cameras), dtype=torch.float32)
    model_boxes, model_cameras = original_boxes.clone(), original_cameras.clone()
    normalized = []
    for index, crop in enumerate(crops):
        if right_hand[index]:
            crop = crop[:, ::-1]
            model_boxes[index], model_cameras[index] = mirror_hand_camera(
                original_boxes[index], original_cameras[index], size, calibrated=True
            )
        crop = (crop.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        normalized.append(torch.from_numpy(crop.transpose(2, 0, 1).copy()))
    batch = {
        "img": torch.stack(normalized),
        "bbx_xys": model_boxes,
        "K_fullimg": model_cameras,
        "img_size": torch.full((len(accepted), 2), float(size)),
        "right_hand": right_hand,
        "original_bbx_xys": original_boxes,
        "original_K_fullimg": original_cameras,
    }
    batch = {name: value.to(predictor.device) for name, value in batch.items()}
    batch["img"] = batch["img"].to(dtype=next(predictor.model.parameters()).dtype)
    return batch, np.stack(rotations), accepted, skipped


@torch.inference_mode()
def predict_perspective(predictor, image, boxes, *, right_hand, camera):
    """Predict in virtual pinhole views and restore physical source-camera poses."""
    if (
        predictor.config.hand_mirror != "full_image"
        or predictor.config.camera_mode != "sam3db_perspective"
    ):
        raise ValueError(
            "Perspective hand inference requires calibrated full_image camera semantics"
        )
    batch, rotations, accepted, skipped = _perspective_batch(
        predictor, image, boxes, right_hand, camera
    )
    if batch is None:
        return None, [], accepted, skipped
    prediction = predictor.predict_batch(batch)
    rotation = torch.as_tensor(
        rotations, dtype=prediction.joints_camera.dtype, device=prediction.joints_camera.device
    )
    with torch.autocast(device_type=rotation.device.type, enabled=False):
        joints = prediction.joints_camera @ rotation.transpose(-1, -2)
        vertices = prediction.vertices_camera @ rotation.transpose(-1, -2)
        translation = (rotation @ prediction.camera_translation[..., None]).squeeze(-1)
        wrist = predictor.geometry.rotate_wrist_orientation(
            prediction.parameters["wrist_orient"], rotation, right=batch["right_hand"]
        )
    points = joints.double().cpu().numpy()
    valid = np.isfinite(points).all(-1) & (points[..., 2] > predictor.config.minimum_depth)
    pixels = np.full((*points.shape[:-1], 2), np.nan)
    pixels[valid] = camera.project(points[valid])
    valid &= np.isfinite(pixels).all(-1)
    pixels[~valid] = np.nan
    pixels = torch.as_tensor(pixels, dtype=joints.dtype, device=joints.device)
    source_boxes = boxes[accepted].to(joints.device)
    crop = (pixels - source_boxes[:, None, :2]) / source_boxes[:, None, 2:]
    result = replace(
        prediction,
        parameters={**prediction.parameters, "wrist_orient": wrist},
        joints_camera=joints,
        vertices_camera=vertices,
        camera_translation=translation,
        keypoints_2d=pixels,
        keypoints_crop=crop,
        projection_valid=torch.as_tensor(valid, device=joints.device),
        intrinsics=None,
    )
    crop_metadata = [
        {
            "intrinsics": batch["original_K_fullimg"][index],
            "bbox": batch["original_bbx_xys"][index],
            "rotation_to_source": rotation[index],
        }
        for index in range(len(accepted))
    ]
    return result, crop_metadata, accepted, skipped


# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# restore_source_camera_predictions adapts inference/pipeline/predictor.py from
# NVIDIA EgoHand under LicenseRef-OpenMDW-1.1. Other camera-adapter functions in
# this file retain the repository's Apache-2.0 license. See ATTRIBUTIONS.md and
# LICENSES/LicenseRef-OpenMDW-1.1.txt.
def restore_source_camera_predictions(result, camera):
    """Restore full-image-undistorted outputs to their calibrated source raster.

    Input boxes and frame intrinsics describe the base pinhole raster, after
    undoing any square-pixel resize. Raw and smoothed joints already contain
    translation in the unchanged source camera axes. Only pixel quantities are
    replaced; geometry and physical-side parameters retain their values.
    """
    from gem.common.undistortion import source_boxes

    def array(value):
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )

    def restore(value, template):
        if isinstance(template, torch.Tensor):
            return torch.as_tensor(value, dtype=template.dtype, device=template.device)
        return np.asarray(value, dtype=np.asarray(template).dtype)

    minimum_depth = (result.get("release") or {}).get("inference", {}).get("minimum_depth", 1e-4)
    calibration = camera.to_dict()
    frames = []
    for original_frame in result["frames"]:
        frame = dict(original_frame, camera=calibration, intrinsics=None, instances=[])
        instances = original_frame["instances"]
        input_boxes = np.asarray([array(row["bbox"]) for row in instances]).reshape(-1, 3)
        intrinsics = array(original_frame["intrinsics"])
        if result.get("camera_adapter", {}).get("square_pixels", False):
            resize = SquarePixelTransform.from_intrinsics(intrinsics, *original_frame["image_size"])
            # The saved base square is an invertible crop descriptor, not the
            # anisotropic footprint of the actual model square. Reconstruct
            # that square and its K before projecting its full source bounds.
            input_boxes = resize.transform_boxes(input_boxes)
            intrinsics = resize.intrinsics
        boxes, accepted = source_boxes(input_boxes, camera, intrinsics)
        for index in np.flatnonzero(accepted):
            original = instances[index]
            row = dict(original, bbox=restore(boxes[index], original["bbox"]), intrinsics=None)
            variants = [row]
            if "smoothed" in original:
                row["smoothed"] = dict(original["smoothed"])
                variants.append(row["smoothed"])
            for variant in variants:
                points = array(variant["joints_camera"]).astype(np.float64)
                valid = np.isfinite(points).all(axis=-1) & (points[:, 2] > minimum_depth)
                pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
                pixels[valid] = camera.project(points[valid])
                crop = (pixels - boxes[index, :2]) / boxes[index, 2]
                valid &= np.isfinite(crop).all(axis=-1)
                pixels[~valid] = np.nan
                crop[~valid] = np.nan
                variant["keypoints_2d"] = restore(pixels, variant["keypoints_2d"])
                variant["keypoints_crop"] = restore(crop, variant["keypoints_crop"])
                variant["projection_valid"] = restore(valid, variant["projection_valid"])
            frame["instances"].append(row)
        frames.append(frame)
    restored = {
        **result,
        "frames": frames,
        "camera_adapter": {
            **result.get("camera_adapter", {}),
            "full_image_undistortion": True,
            "output_raster": "source",
        },
    }
    if "box_provider" in result:
        restored["box_provider"] = {
            **result["box_provider"],
            "inference_annotation_coordinates": "undistorted_image",
            "annotation_coordinates": "source_image",
        }
    return restored
