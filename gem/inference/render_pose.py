# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render saved framewise Pose geometry, including physical left/right hands."""

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from gem.common.rendering import MotionMeshRenderer

from .render import RGBVideoWriter, _label, resized_intrinsics
from .video import VideoFrames

# SOMAHandLayer's 25-joint contract: wrist, thumb, then four five-joint fingers.
_HAND_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8, 9),
    (0, 10, 11, 12, 13, 14),
    (0, 15, 16, 17, 18, 19),
    (0, 20, 21, 22, 23, 24),
)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _camera_keypoints(rgb, record, variant):
    """Draw stored lens-projected joints in the original source raster."""
    scale = np.array([rgb.shape[1], rgb.shape[0]]) / record["image_size"]
    for row in record["instances"]:
        geometry = row["smoothed"] if variant == "smoothed" else row
        pixels = (_array(geometry["keypoints_2d"]) + 0.5) * scale - 0.5
        valid = _array(geometry["projection_valid"]) & np.isfinite(pixels).all(axis=-1)
        color = (80, 225, 120) if row["side"] == "left" else (80, 175, 255)
        for chain in _HAND_CHAINS:
            for parent, child in zip(chain, chain[1:]):
                if valid[parent] and valid[child]:
                    cv2.line(
                        rgb,
                        tuple(np.rint(pixels[parent]).astype(int)),
                        tuple(np.rint(pixels[child]).astype(int)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        for point in pixels[valid]:
            cv2.circle(rgb, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
    return rgb


def _composite(renderer, image, record, topology, variant, *, size=None):
    """One combined scene keeps depth ordering correct between hand instances."""
    size = size or (renderer.width, renderer.height)
    rgb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if record is None or not record["instances"]:
        return rgb
    if record["image_size"] != [image.shape[1], image.shape[0]]:
        raise ValueError("Source raster dimensions differ from the saved Pose result")
    if record.get("camera") is not None:
        return _camera_keypoints(rgb, record, variant)
    reference_camera = _array(record["instances"][0].get("intrinsics", record["intrinsics"]))
    vertices, faces, offset = [], [], 0
    for row in record["instances"]:
        geometry = row["smoothed"] if variant == "smoothed" else row
        points = _array(geometry["vertices_camera"])
        camera = _array(row.get("intrinsics", record["intrinsics"]))
        if not np.array_equal(camera, reference_camera):
            # Custom releases may synthesize a different K per hand. Express
            # their rays under one render camera; pinhole Z/depth is unchanged.
            points = np.linalg.solve(reference_camera, camera @ points.T).T
        vertices.append(points)
        faces.append(_array(topology[row["topology"]]) + offset)
        offset += len(points)
    camera = resized_intrinsics(
        reference_camera,
        (image.shape[1], image.shape[0]),
        (renderer.width, renderer.height),
    )
    # Saved vertices already contain camera translation and physical-side FK.
    return renderer.render(
        np.concatenate(vertices), np.concatenate(faces), camera, np.eye(4), background=rgb
    )


def render_pose(
    result, output_dir, *, source=None, width=None, backend="software", variant=None, progress=None
):
    """Export image/video overlays directly from stored Pose geometry.

    No model or SOMA assets are loaded. Absent predictions stay absent; a video
    retains all source frames, including unannotated gaps. When smoothing was
    explicitly requested during hand inference, the default display uses it.
    Calibrated perspective-crop bundles draw source-camera keypoints; a pinhole mesh renderer cannot
    represent lens distortion. They require no Open3D rendering context.
    """
    variant = variant or ("smoothed" if result.get("hand_smoothing") else "raw")
    if variant not in ("raw", "smoothed"):
        raise ValueError("Pose rendering supports raw or smoothed geometry")
    if variant == "smoothed" and not result.get("hand_smoothing"):
        raise ValueError("This Pose result has no smoothed hand geometry")
    if width is not None and (width < 64 or width % 2):
        raise ValueError("Render width must be an even integer of at least 64 pixels")
    path = source or result["source"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {row["frame_id"]: row for row in result["frames"]}
    topology = result["topology"]
    camera_projected = any(frame.get("camera") is not None for frame in result["frames"])
    metadata = {
        "format": "gem_pose_render_v1",
        "variant": variant,
        "overlay_type": "source_camera_keypoints" if camera_projected else "mesh",
        "videos": {},
        "images": {},
    }
    title = f"{result['task'].capitalize()} Pose" + (" - smoothed" if variant == "smoothed" else "")
    if camera_projected:
        title += " - source-camera keypoints"
    if result["mode"] == "image":
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"Cannot read source image {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        render_width = width or image.shape[1]
        height = round(image.shape[0] * render_width / image.shape[1])
        context = (
            nullcontext()
            if camera_projected
            else MotionMeshRenderer(render_width, height, backend=backend)
        )
        with context as renderer:
            rgb = _composite(
                renderer, image, records.get(0), topology, variant, size=(render_width, height)
            )
        output = output_dir / "overlay.png"
        if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Cannot write image {output}")
        metadata["images"]["overlay"] = str(output.resolve())
        metadata["frame_count"] = 1
    elif result["mode"] == "framewise_video":
        with VideoFrames(path) as video:
            if records and max(records) >= len(video):
                raise ValueError("Source video ends before a saved Pose frame")
            if len(video) != result.get("source_frame_count", len(video)) or not np.isclose(
                video.fps, result["source_fps"]
            ):
                raise ValueError("Source video timeline differs from the saved Pose result")
            render_width = width or video.width
            render_width -= render_width % 2
            height = 2 * round(video.height * render_width / video.width / 2)
            output = output_dir / "overlay.mp4"
            context = (
                nullcontext()
                if camera_projected
                else MotionMeshRenderer(render_width, height, backend=backend)
            )
            with (
                context as renderer,
                RGBVideoWriter(output, render_width, height, video.fps) as writer,
            ):
                for index in range(len(video)):
                    rgb = _composite(
                        renderer,
                        video[index],
                        records.get(index),
                        topology,
                        variant,
                        size=(render_width, height),
                    )
                    writer.write(_label(rgb, title, index, video.fps))
                    if progress is not None and (index + 1) % 30 == 0:
                        progress(f"Pose overlay: {index + 1}/{len(video)}")
            metadata["videos"]["overlay"] = str(output.resolve())
            metadata.update(frame_count=len(video), fps=video.fps)
    else:
        raise ValueError("Pose rendering requires an image or framewise_video result")
    return metadata
