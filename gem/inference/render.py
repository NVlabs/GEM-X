# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render saved Motion parameters without loading any neural checkpoints."""

import json
import subprocess
from contextlib import ExitStack, nullcontext
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from .timing import prediction_fps
from .video import VideoFrames, normalize_video


def world_view(minimum, maximum, first_root, width, height):
    """One display translation and one camera for the entire Y-up sequence.

    Camera extrinsics map world coordinates into OpenCV x-right/y-down/z-forward.
    Bounds include every mesh frame, so the view does not chase or recenter roots.
    """
    offset = np.asarray(first_root, dtype=np.float64).copy()
    offset[1] = minimum[1]
    lower, upper = np.asarray(minimum) - offset, np.asarray(maximum) - offset
    target = (lower + upper) / 2
    direction = np.array([1.0, 0.65, 1.0])
    direction /= np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    corners = np.array(list(product(*zip(lower, upper))))
    camera_corners = (corners - target) @ rotation.T
    focal = height / (2 * np.tan(np.deg2rad(40) / 2))
    # Reserve 15% of each half-frame for clear margins around the full sequence.
    distance = max(
        np.max(np.abs(camera_corners[:, 0]) * focal / (width * 0.425) - camera_corners[:, 2]),
        np.max(np.abs(camera_corners[:, 1]) * focal / (height * 0.425) - camera_corners[:, 2]),
        0.1 - camera_corners[:, 2].min(),
    )
    eye = target + direction * distance
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    ground = [lower[0] - 0.75, upper[0] + 0.75, lower[2] - 0.75, upper[2] + 0.75]
    return {
        "display_offset": offset,
        "intrinsics": intrinsics,
        "extrinsic": extrinsic,
        "ground_bounds": ground,
    }


def resized_intrinsics(intrinsics, source_size, target_size):
    """Match OpenCV resize's pixel-center convention, including fx != fy."""
    scale = np.asarray(target_size, dtype=np.float64) / source_size
    transform = np.eye(3)
    transform[0, 0], transform[1, 1] = scale
    transform[:2, 2] = (scale - 1) / 2
    return transform @ np.asarray(intrinsics)


class RGBVideoWriter:
    """Stream RGB directly to browser-playable H.264, without an intermediate AVI."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.close()
        error = self.process.stderr.read().decode()
        code = self.process.wait()
        self.process.stderr.close()
        if code and exc_type is None:
            raise RuntimeError(f"Video encoding failed: {error}")


def _label(rgb, title, frame, fps):
    result = rgb.copy()
    text = f"{title}  |  {frame / fps:.2f}s"
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    right = min(result.shape[1] - 12, 32 + text_width)
    panel = result[12:42, 12:right]
    panel[:] = np.rint(panel * 0.25 + np.array([25, 30, 34]) * 0.75).astype(np.uint8)
    cv2.putText(
        result, text, (22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA
    )
    return result


def _mesh_cache(geometry, parameters, directory, name):
    """Materialize only disk-backed vertices; GPU tensors are released by chunk."""
    length = parameters["body_pose"].shape[1]
    vertices, joints = None, None
    for start, chunk in geometry.iter_chunks(parameters):
        values = chunk["vertices"][0].numpy()
        joint_values = chunk["joints"][0].numpy()
        if vertices is None:
            vertices = np.lib.format.open_memmap(
                Path(directory) / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(length, *values.shape[1:]),
            )
            joints = np.empty((length, *joint_values.shape[1:]), dtype=np.float32)
        stop = start + len(values)
        vertices[start:stop], joints[start:stop] = values, joint_values
    return vertices, joints


def render_motion(
    result,
    output_dir,
    *,
    soma_assets,
    device="cpu",
    views=("overlay", "world"),
    source_video=None,
    width=None,
    backend="software",
    chunk_size=32,
    variant=None,
    progress=None,
):
    """Render one saved Motion sequence; output parameters are never changed.

    A body-video result carries the original source interval and K. A bare
    Motion result needs conditions.K_fullimg and an already aligned source video
    for overlays; world-only export requires neither. Optional visualization
    correctives affect the mesh surface, not the predicted motion or contacts.
    """
    from gem.common.rendering import MotionMeshRenderer
    from gem.motion.geometry import MotionGeometry
    from gem.motion.postprocess.api import ieee_geometry

    from .postprocess import select_motion_result

    result = select_motion_result(result, variant)
    processed = bool(result.get("postprocessing"))
    parameters = result["pred_soma_params_global"]
    if parameters["body_pose"].shape[0] != 1:
        raise ValueError("Render one Motion sequence at a time (batch size 1)")
    length = int(torch.as_tensor(result["length"]).item())
    if length != parameters["body_pose"].shape[1] or length == 0:
        raise ValueError("Render input must contain exactly its real, unpadded sequence length")
    if width is not None and (width < 64 or width % 2):
        raise ValueError("Render width must be an even integer of at least 64 pixels")
    if not views or any(view not in ("overlay", "world") for view in views):
        raise ValueError("Render views must contain overlay and/or world")
    fps = prediction_fps(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "frame_count": length,
        "fps": fps,
        "body_model": "SOMA0.3.3",
        "lod": "mid",
        "pose_correctives": True,
        "postprocess": processed,
        "world_coordinates": "Y_up_meters",
        "camera_coordinates": "OpenCV_meters",
        "render_backend": backend,
        "videos": {},
    }

    def stage(message):
        if progress is not None:
            progress(message)

    with TemporaryDirectory(prefix="gem-render-") as temporary, ExitStack() as stack:
        frames, camera = None, None
        if "overlay" in views:
            camera = torch.as_tensor(result["conditions"]["K_fullimg"]).cpu().numpy()[0]
            if "video" in result:
                source = result["video"]["source"]
                source_video = source_video or source["path"]
                normalized = Path(temporary) / "source.mp4"
                normalize_video(
                    source_video,
                    normalized,
                    start=source["requested_start"],
                    duration=source["selected_duration"],
                )
                source_video = normalized
            elif source_video is None:
                raise ValueError("Overlay requires --source-video or saved source video metadata")
            frames = stack.enter_context(VideoFrames(source_video))
            if len(frames) != length or not np.isclose(frames.fps, fps, rtol=1e-4):
                raise ValueError(
                    "Overlay video frame count/FPS differs from the saved Motion timeline"
                )

        stage("Reconstructing SOMA visualization meshes")
        geometry = MotionGeometry(soma_assets, device=device, chunk_size=chunk_size)
        faces = geometry.faces.numpy()
        metadata["vertex_count"] = None
        for view in dict.fromkeys(views):
            coordinate = "incam" if view == "overlay" else "global"
            with ieee_geometry() if processed and view == "world" else nullcontext():
                vertices, joints = _mesh_cache(
                    geometry, result[f"pred_soma_params_{coordinate}"], temporary, view
                )
            metadata["vertex_count"] = vertices.shape[1]
            output = output_dir / f"{view}.mp4"
            render_width = width or (frames.width if view == "overlay" else 1280)
            render_width -= render_width % 2
            height = 2 * round(
                (
                    frames.height * render_width / frames.width
                    if view == "overlay"
                    else render_width * 0.75
                )
                / 2
            )
            world = None
            if view == "world":
                world = world_view(
                    vertices.min(axis=(0, 1)),
                    vertices.max(axis=(0, 1)),
                    joints[0, 0],
                    render_width,
                    height,
                )
                metadata["world_display"] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in world.items()
                }
                roots = joints[:, 0] - world["display_offset"]
                roots[:, 1] = 0.015
                metadata["trajectory"] = "Hips path projected onto display ground"
            stage(f"Rendering {view} ({length} frames)")
            with (
                MotionMeshRenderer(render_width, height, backend=backend) as renderer,
                RGBVideoWriter(output, render_width, height, fps) as writer,
            ):
                for index in range(length):
                    if view == "overlay":
                        image = cv2.resize(
                            frames[index], (render_width, height), interpolation=cv2.INTER_AREA
                        )
                        intrinsics = resized_intrinsics(
                            camera[index], (frames.width, frames.height), (render_width, height)
                        )
                        rgb = renderer.render(
                            vertices[index], faces, intrinsics, np.eye(4), background=image
                        )
                    else:
                        rgb = renderer.render(
                            vertices[index] - world["display_offset"],
                            faces,
                            world["intrinsics"],
                            world["extrinsic"],
                            ground_bounds=world["ground_bounds"],
                            trajectory=roots[: index + 1],
                        )
                    writer.write(
                        _label(
                            rgb,
                            ("Camera overlay" if view == "overlay" else "World view")
                            + (" - contact/IK" if processed else " - raw Motion"),
                            index,
                            fps,
                        )
                    )
                    if progress is not None and (index + 1) % 30 == 0:
                        stage(f"{view}: {index + 1}/{length}")
            metadata["videos"][view] = str(output.resolve())
            del vertices
    (output_dir / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
