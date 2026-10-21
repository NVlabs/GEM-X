# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive SOMA parameter filtering and consistent physical-hand geometry."""

from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch

from .smoothing import AdaptiveSmoothingConfig, smooth_rotations, smooth_shape, smooth_translation

PARAMETER_FIELDS = ("wrist_orient", "hand_pose", "identity_coeffs", "scale_params")
SHAPE_FIELDS = ("identity_coeffs", "scale_params")
GEOMETRY_FIELDS = (
    "camera_translation",
    "joints_camera",
    "vertices_camera",
    "keypoints_2d",
    "keypoints_crop",
    "projection_valid",
)
FK_BATCH_SIZE = 64


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _restore(value, template):
    if isinstance(template, torch.Tensor):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device)
    if np.ndim(template) == 0:
        return float(value)
    return np.asarray(value, dtype=np.asarray(template).dtype)


def _project(local_joints, translation, intrinsics, bbox, minimum_depth, *, camera=None):
    """Retain the source float64 projection after filtering local geometry."""
    points = (
        np.asarray(local_joints, dtype=np.float64) + np.asarray(translation, dtype=np.float64)[None]
    )
    valid = np.isfinite(points).all(-1) & (points[:, 2] > minimum_depth)
    safe = np.zeros_like(points)
    safe[:, 2] = 1.0
    safe[valid] = points[valid]
    if camera is None:
        pixels = (safe / safe[:, 2:]) @ np.asarray(intrinsics, dtype=np.float64).T
    else:
        from gem.common.camera_models import camera_from_dict

        pixels = camera_from_dict(camera).project(safe)
    crop = (pixels[:, :2] - np.asarray(bbox, dtype=np.float64)[None, :2]) / float(bbox[2])
    valid &= np.isfinite(crop).all(-1)
    pixels, crop = pixels[:, :2].copy(), crop.copy()
    pixels[~valid] = np.nan
    crop[~valid] = np.nan
    return points, pixels, crop, valid


def _segments(entries, fps, config):
    """Split at missing decoded frames, nonfinite parameters and time gaps."""
    segment = []
    for entry in entries:
        frame_id, row, _, _ = entry
        values = [
            row["camera_translation"],
            *(row["parameters"][name] for name in PARAMETER_FIELDS),
        ]
        valid = all(np.isfinite(_numpy(value)).all() for value in values)
        if segment and (
            not valid
            or frame_id != segment[-1][0] + 1
            or (frame_id - segment[-1][0]) / fps > config.max_gap_seconds
        ):
            yield segment
            segment = []
        if valid:
            segment.append(entry)
    if segment:
        yield segment


def smooth_hand_predictions(result, *, geometry, config=None):
    """Add filtered parameters and matching geometry, preserving raw rows.

    The filter is offline and uses decoded ``frame_id / source_fps`` timestamps.
    Each ``(track_id, side)`` is independent; absent predictions and nonfinite
    parameters split tracks and remain unchanged. Geometry is decoded in FP32
    from the already physical-side rotations, with no second right-hand mirror.
    The supplied HandGeometry is reused serially after neural inference.
    """
    if result["task"] != "hand" or result["mode"] != "framewise_video":
        raise ValueError("Hand smoothing requires framewise hand video predictions")
    fps = float(result["source_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Hand smoothing requires a finite positive source_fps")
    config = AdaptiveSmoothingConfig() if config is None else config
    frames = [
        dict(frame, instances=[dict(row) for row in frame["instances"]])
        for frame in result["frames"]
    ]
    frame_count = result.get(
        "source_frame_count", max((frame["frame_id"] for frame in frames), default=-1) + 1
    )
    groups = defaultdict(list)
    for frame in frames:
        for row in frame["instances"]:
            row["smoothed"] = {
                **{name: row[name] for name in GEOMETRY_FIELDS},
                "parameters": dict(row["parameters"]),
                "shape_parameters": {name: row["parameters"][name] for name in SHAPE_FIELDS},
            }
            groups[(row["track_id"], row["side"])].append(
                (
                    frame["frame_id"],
                    row,
                    row.get("intrinsics", frame["intrinsics"]),
                    frame.get("camera"),
                )
            )
    release = result.get("release") or {}
    minimum_depth = release.get("inference", {}).get("minimum_depth", 1e-4)
    device = geometry._device_anchor.device
    for (_, side), entries in groups.items():
        entries.sort(key=lambda item: item[0])
        for segment in _segments(entries, fps, config):
            times = np.array([entry[0] / fps for entry in segment], dtype=np.float64)
            rows = [entry[1] for entry in segment]
            translations = smooth_translation(
                np.stack([_numpy(row["camera_translation"]) for row in rows]), times, config
            )
            parameters = {
                name: np.stack([_numpy(row["parameters"][name]) for row in rows])
                for name in PARAMETER_FIELDS
            }
            rotations = smooth_rotations(
                np.concatenate((parameters["wrist_orient"], parameters["hand_pose"]), axis=1),
                times,
                config,
            )
            parameters["wrist_orient"], parameters["hand_pose"] = rotations[:, :1], rotations[:, 1:]
            for name in SHAPE_FIELDS:
                parameters[name] = smooth_shape(parameters[name], times, config.shape_sigma_seconds)
            # Store in the original parameter dtype before FK: consumers must
            # recover the same geometry from exactly these saved parameters.
            for index, row in enumerate(rows):
                smoothed = row["smoothed"]
                smoothed["parameters"] = {
                    name: _restore(parameters[name][index], row["parameters"][name])
                    for name in PARAMETER_FIELDS
                }
                smoothed["shape_parameters"] = {
                    name: smoothed["parameters"][name] for name in SHAPE_FIELDS
                }
                smoothed["camera_translation"] = _restore(
                    translations[index], row["camera_translation"]
                )

            for start in range(0, len(rows), FK_BATCH_SIZE):
                batch = rows[start : start + FK_BATCH_SIZE]
                physical = {
                    name: torch.as_tensor(
                        np.stack([_numpy(row["smoothed"]["parameters"][name]) for row in batch]),
                        dtype=torch.float32,
                        device=device,
                    )
                    for name in PARAMETER_FIELDS
                }
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                    decoded = geometry.forward_physical(
                        physical,
                        right=torch.full(
                            (len(batch),), side == "right", dtype=torch.bool, device=device
                        ),
                    )
                joints = _numpy(decoded["joints"])
                vertices = _numpy(decoded["vertices"])
                for offset, row in enumerate(batch):
                    _, _, intrinsics, camera = segment[start + offset]
                    smoothed = row["smoothed"]
                    translation = _numpy(smoothed["camera_translation"])
                    points, pixels, crop, valid = _project(
                        joints[offset],
                        translation,
                        _numpy(intrinsics) if intrinsics is not None else None,
                        _numpy(row["bbox"]),
                        minimum_depth,
                        camera=camera,
                    )
                    smoothed.update(
                        joints_camera=_restore(points, row["joints_camera"]),
                        vertices_camera=_restore(
                            vertices[offset].astype(np.float64)
                            + translation.astype(np.float64)[None],
                            row["vertices_camera"],
                        ),
                        keypoints_2d=_restore(pixels, row["keypoints_2d"]),
                        keypoints_crop=_restore(crop, row["keypoints_crop"]),
                        projection_valid=_restore(valid, row["projection_valid"]),
                    )
    return {
        **result,
        "frames": frames,
        "hand_smoothing": {
            "method": "adaptive_soma_parameters",
            "config": asdict(config),
            "timeline_frame_count": int(frame_count),
            "frame_grid": "decoded_frame_id",
            "time_unit": "seconds",
            "missing_predictions": "preserved_track_boundaries",
            "rotations_filtered": True,
            "fk_parameters": "smoothed.parameters",
        },
    }
