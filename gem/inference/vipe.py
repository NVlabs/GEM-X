# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional ViPE CLI adapter for the normalized 30 Hz Body Motion video."""

import subprocess
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from gem.inference.camera_motion import camera_angular_velocity


def _load_dense_artifact(path, frame_count, item_shape):
    """Read ViPE's indexed arrays, requiring every frame on the input timeline."""
    try:
        with np.load(path) as artifact:
            values = np.asarray(artifact["data"], dtype=np.float32)
            indices = artifact["inds"]
    except FileNotFoundError as error:
        raise ValueError(f"ViPE did not produce the required camera artifact: {path}") from error
    if values.shape != (frame_count, *item_shape):
        raise ValueError(
            f"ViPE {path.parent.name} data must have shape {(frame_count, *item_shape)} "
            f"for the normalized video; got {values.shape}"
        )
    if indices.shape != (frame_count,):
        raise ValueError(
            f"ViPE {path.parent.name} indices must contain every normalized video frame"
        )
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(frame_count)):
        raise ValueError(
            f"ViPE {path.parent.name} indices must be exactly 0..{frame_count - 1} "
            "on the normalized video timeline; sparse or mismatched results are unsupported"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"ViPE {path.parent.name} data must contain only finite values")
    return values[order]


def run_vipe_camera(
    video_path: str | PathLike,
    *,
    frame_count: int,
    executable: str | PathLike = "vipe",
    pipeline: str = "pose_only_long",
):
    """Run ViPE and return dense camera conditioning for the supplied 30 Hz clip.

    The caller must pass the same normalized video used for Body Motion, with
    its decoded frame count. ViPE runs synchronously in its own environment;
    ``executable`` may name that environment's absolute ``bin/vipe`` path.
    Its temporary output directory is removed after the arrays are loaded.

    Official ViPE poses are OpenCV camera-to-world matrices. Their rotation
    transposes become GEM's world-to-camera rotations; per-frame pinhole
    intrinsics are retained. Camera translations are returned in ``C2W`` for
    inspection, but are not used for Motion conditioning or trajectory fusion.
    Failures propagate without substituting a static camera or interpolating
    an incomplete trajectory. GEM does not import the ViPE Python package.
    """
    if frame_count < 1:
        raise ValueError("ViPE camera estimation requires at least one normalized video frame")
    video_path = Path(video_path)
    with TemporaryDirectory(prefix="gem-vipe-") as output:
        try:
            subprocess.run(
                [str(executable), "infer", str(video_path), "-p", pipeline, "-o", output],
                check=True,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"ViPE executable {str(executable)!r} was not found. Install ViPE in a separate "
                "environment and select that environment's bin/vipe executable."
            ) from error
        output = Path(output)
        poses = _load_dense_artifact(
            output / "pose" / f"{video_path.stem}.npz", frame_count, (4, 4)
        )
        intrinsics = _load_dense_artifact(
            output / "intrinsics" / f"{video_path.stem}.npz", frame_count, (4,)
        )
        camera_path = output / "intrinsics" / f"{video_path.stem}_camera.txt"
        if camera_path.exists():
            for line in camera_path.read_text().splitlines():
                _, camera_type = line.split(":", 1)
                if camera_type.strip() != "PINHOLE":
                    raise ValueError(
                        f"GEM's ViPE adapter requires PINHOLE intrinsics; got {camera_type.strip()}"
                    )
        if (intrinsics[:, :2] <= 0).any():
            raise ValueError("ViPE pinhole focal lengths fx and fy must be positive")

    camera_to_world = torch.from_numpy(poses)
    rotations = camera_to_world[:, :3, :3].transpose(-1, -2).contiguous()
    calibration = torch.eye(3, dtype=torch.float32).repeat(frame_count, 1, 1)
    calibration[:, 0, 0] = torch.from_numpy(intrinsics[:, 0])
    calibration[:, 1, 1] = torch.from_numpy(intrinsics[:, 1])
    calibration[:, 0, 2] = torch.from_numpy(intrinsics[:, 2])
    calibration[:, 1, 2] = torch.from_numpy(intrinsics[:, 3])
    return {
        "R_w2c": rotations,
        "cam_angvel": camera_angular_velocity(rotations),
        "K_fullimg": calibration,
        "C2W": camera_to_world,
        "diagnostics": {
            "method": "vipe",
            "pipeline": pipeline,
            "camera_model": "PINHOLE",
            "frame_count": frame_count,
            "sampling": "dense_per_frame",
            "translation_used": False,
        },
    }
