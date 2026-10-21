# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framewise Pose prediction using the shared crop/camera/geometry API."""

from dataclasses import asdict
from pathlib import Path

import torch

from .inputs import load_mapping
from .serialization import tensor_tree


def load_predictor(checkpoint, *, task, release=None, soma_assets, device, hand_mirror=None):
    from gem.pose.api import (
        PoseReleaseConfig,
        body_config,
        body_sg6_config,
        checkpoint_pose_release,
        hand_config,
        load_pose_predictor,
    )

    presets = {
        "body": body_config,
        "body_sg6": body_sg6_config,
        "hand": hand_config,
    }
    if release is None:
        config = checkpoint_pose_release(checkpoint)
        if config is None:
            config = presets[task]()
    elif release in presets:
        config = presets[release]()
    else:
        config = PoseReleaseConfig(**load_mapping(release))
    if config.task != task:
        raise ValueError(f"Pose release task {config.task!r} does not match --task {task!r}")
    if hand_mirror is not None:
        if task != "hand":
            raise ValueError("--hand-mirror applies to hand releases")
        config.inference["hand_mirror"] = hand_mirror
    predictor = load_pose_predictor(checkpoint, config, soma_assets=soma_assets, device=device)
    return predictor, asdict(config)


def predict_frame(
    predictor, image, frame_id, record, *, task, topology, square_pixels=False, camera=None
):
    """Predict one decoded RGB frame, retaining caller instance order and IDs."""
    instances = record["instances"]
    result = {
        "frame_id": frame_id,
        "image_size": [image.shape[1], image.shape[0]],
        "intrinsics": (
            None
            if camera is not None
            else torch.as_tensor(record["intrinsics"], dtype=torch.float32)
        ),
        "instances": [],
    }
    if camera is not None:
        if [camera.width, camera.height] != result["image_size"]:
            raise ValueError("Camera calibration dimensions must match the decoded source raster")
        result["camera"] = camera.to_dict()
    if not instances:
        return result
    boxes = torch.tensor([instance["bbox"] for instance in instances], dtype=torch.float32)
    right = (
        torch.tensor([instance["side"] == "right" for instance in instances])
        if task == "hand"
        else None
    )
    crop_metadata = None
    if camera is not None:
        from .camera import predict_perspective

        prediction, crop_metadata, accepted, skipped = predict_perspective(
            predictor, image, boxes, right_hand=right, camera=camera
        )
        if skipped:
            result["skipped_instances"] = [
                {**instances[index], "reason": reason} for index, reason in skipped.items()
            ]
        if prediction is None:
            return result
        instances = [instances[index] for index in accepted]
        boxes = boxes[accepted]
    elif square_pixels:
        from .camera import predict_square_pixels

        prediction = predict_square_pixels(
            predictor, image, boxes, result["intrinsics"], right_hand=right
        )
    else:
        prediction = predictor.predict(image, boxes, result["intrinsics"], right_hand=right)
    fields = (
        "camera_translation",
        "joints_camera",
        "vertices_camera",
        "keypoints_2d",
        "keypoints_crop",
        "projection_valid",
    )
    for index, instance in enumerate(instances):
        topology_key = instance["side"] if task == "hand" else "body"
        if topology_key not in topology:
            topology[topology_key] = tensor_tree(prediction.faces[index])
        row = {
            "track_id": instance["track_id"],
            "bbox": boxes[index],
            "intrinsics": (
                None if prediction.intrinsics is None else tensor_tree(prediction.intrinsics[index])
            ),
            "topology": topology_key,
            "parameters": {key: value[index] for key, value in prediction.parameters.items()},
            "pose_token": prediction.head.token[index],
            **{key: getattr(prediction, key)[index] for key in fields},
        }
        if crop_metadata is not None:
            row["perspective_crop"] = crop_metadata[index]
        for name in ("bbox_observed", "detection_score"):
            if name in instance:
                row[name] = instance[name]
        if task == "hand":
            row["side"] = instance["side"]
        result["instances"].append(tensor_tree(row))
    return result


def run_pose(
    predictor,
    input_path,
    annotations,
    *,
    task,
    video=False,
    release=None,
    square_pixels=False,
    perspective_camera=None,
    undistort_camera=None,
):
    """Predict supplied frames, retaining gaps and the chosen camera raster.

    ``undistort_camera`` remaps the complete image before ordinary pinhole
    cropping and optional square-pixel correction. Supplied boxes must already
    be in that undistorted base-K raster. Raw results remain in that raster so
    temporal filtering can finish before source-camera projection is restored.
    """
    import cv2

    if perspective_camera is not None and undistort_camera is not None:
        raise ValueError("Choose perspective crops or full-image undistortion, not both")
    undistortion = None
    if undistort_camera is not None:
        from gem.common.camera_models import camera_from_dict
        from gem.common.undistortion import FullImageUndistortion

        if task != "hand":
            raise ValueError("Full-image undistortion requires hand inference")
        undistortion = FullImageUndistortion(camera_from_dict(undistort_camera))
    camera = None
    if perspective_camera is not None:
        from gem.common.camera_models import camera_from_dict

        if task != "hand" or square_pixels:
            raise ValueError(
                "Perspective camera cropping requires hand inference without square resizing"
            )
        camera = camera_from_dict(perspective_camera)
    topology, frames = {}, []

    def predict_image(bgr, frame_id):
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        record = annotations[frame_id]
        if undistortion is not None:
            image = undistortion.apply(image)
            record = {**record, "intrinsics": undistortion.K}
        return predict_frame(
            predictor,
            image,
            frame_id,
            record,
            task=task,
            topology=topology,
            square_pixels=square_pixels,
            camera=camera,
        )

    if video:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0
        try:
            while annotations and frame_id <= max(annotations):
                available, bgr = capture.read()
                if not available:
                    remaining = [key for key in annotations if key >= frame_id]
                    raise ValueError(f"Video ended before annotated frames {remaining}")
                if frame_id in annotations:
                    frames.append(predict_image(bgr, frame_id))
                frame_id += 1
        finally:
            capture.release()
    else:
        bgr = cv2.imread(str(input_path))
        if bgr is None:
            raise OSError(f"Cannot read image {input_path}")
        frames.append(predict_image(bgr, 0))
        fps = None
        source_frame_count = 1
    return {
        "format": "gem_pose_predictions_v1",
        "task": task,
        "mode": "framewise_video" if video else "image",
        "source": str(Path(input_path).resolve()),
        "source_fps": fps,
        "source_frame_count": source_frame_count,
        "release": release,
        "camera_adapter": {
            "square_pixels": square_pixels,
            "perspective_crops": camera is not None,
            "full_image_undistortion": undistortion is not None,
            "output_raster": "undistorted" if undistortion is not None else "source",
        },
        "coordinates": {
            "geometry": "camera_meters",
            "keypoints_2d": "full_image_pixels",
            "keypoints_crop": "centered_crop",
            "pose_token": "canonical_head",
        },
        "topology": topology,
        "frames": frames,
    }
