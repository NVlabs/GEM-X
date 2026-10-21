# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic Body Pose boxes using the existing person detection/tracking policy."""

from pathlib import Path

from gem.common.detectors import YOLOXONNX, detect_track

from .inputs import input_frames, pose_intrinsics


def detect_body_annotations(input_path, checkpoint, *, video=False, intrinsics=None, device="cpu"):
    """Select the primary person on the source timeline, using Pose intrinsics.

    Detection, association, gap filling and crop padding match Body Motion.
    Pose retains source frame IDs/FPS and its max-side focal estimate.
    """
    detector = YOLOXONNX(checkpoint, device=device)
    with input_frames(input_path, video) as frames:
        height, width = frames[0].shape[:2]
        camera = pose_intrinsics(width, height, intrinsics)
        tracks = detect_track(frames, detector)
    annotations = {
        int(frame_id): {
            "intrinsics": camera,
            "instances": [
                {
                    "track_id": int(tracks.track_ids[index]),
                    "bbox": tracks.boxes_xys[index].tolist(),
                    "bbox_observed": bool(tracks.observed_primary[index]),
                }
            ],
        }
        for index, frame_id in enumerate(tracks.frame_ids)
    }
    metadata = {
        "provider": "yolox",
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": device,
        "tracking": "primary",
        "camera_source": "provided" if intrinsics is not None else "pose_max_side_estimate",
        "image_size": [width, height],
        "annotation_coordinates": "source_image",
    }
    return annotations, metadata
