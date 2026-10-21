# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose input records, decoded RGB frames and camera estimates."""

import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def input_frames(path, video):
    """Decode RGB input on its original image/video timeline."""
    if video:
        from .video import VideoFrames

        with VideoFrames(path) as frames:
            yield frames
    else:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"Cannot read image {path}")
        yield [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]


def pose_intrinsics(width, height, provided):
    """Use supplied intrinsics or Pose's established max-side focal estimate."""
    if provided is not None:
        import numpy as np

        return np.asarray(provided, dtype=np.float32).tolist()
    focal = float(max(width, height))
    return [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]


def load_mapping(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            value = json.load(handle)
    else:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON/YAML object in {path}")
    return value


def frame_annotations(document, *, task, video, intrinsics=None):
    """Return sorted, exact frame IDs and instance records, with no gap filling.

    Boxes are square ``[center_x, center_y, size]`` in full-image pixels. IDs
    belong to the caller; hand IDs may repeat for opposite sides of a person.
    Intrinsics are a shared3x3 matrix or a per-frame ``intrinsics`` override.
    """
    if task not in {"body", "hand"}:
        raise ValueError(f"Unknown Pose task {task!r}")
    records = document["frames"] if video else {0: document}
    common_camera = document.get("intrinsics")
    frames = {}
    for key, record in records.items():
        if not str(key).isdecimal():
            raise ValueError("Frame keys must be non-negative integer frame IDs")
        frame_id = int(key)
        if frame_id in frames:
            raise ValueError(f"Duplicate frame ID {frame_id}")
        camera = intrinsics if intrinsics is not None else record.get("intrinsics", common_camera)
        if camera is None:
            raise ValueError(f"Frame {frame_id} needs supplied intrinsics")
        instances = record["instances"]
        identities = set()
        for instance in instances:
            track_id = instance["track_id"]
            if not isinstance(track_id, str | int) or isinstance(track_id, bool):
                raise ValueError("track_id must be a string or integer")
            side = instance.get("side")
            if task == "hand" and side not in {"left", "right"}:
                raise ValueError("Hand instances require side='left' or side='right'")
            identity = (track_id, side if task == "hand" else None)
            if identity in identities:
                raise ValueError(f"Duplicate track/side {identity} in frame {frame_id}")
            identities.add(identity)
            if len(instance["bbox"]) != 3:
                raise ValueError("bbox must contain [center_x, center_y, size]")
        frames[frame_id] = {"intrinsics": camera, "instances": instances}
    return dict(sorted(frames.items()))
