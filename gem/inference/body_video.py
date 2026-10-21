# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic single-person RGB video to native temporal SOMA parameters."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from gem.common.detectors import YOLOXONNX, detect_track
from gem.common.pose2d import load_native77_detector
from gem.motion.api import MotionConditionSpec
from gem.motion.frontend.joint_sets import get_observation_schema
from gem.pose.api import PoseReleaseConfig, body_config, body_sg6_config, load_pose_checkpoint
from gem.pose.data.crop import prepare_pose_batch

from .camera_motion import camera_motion
from .inputs import load_mapping
from .motion import load_predictor, run_motion
from .serialization import tensor_tree


def load_body_video_config(path):
    """Resolve checkpoint/sidecar paths relative to the release configuration."""
    path = Path(path).resolve()
    config = load_mapping(path)
    for section, keys in (
        ("pose", ("checkpoint",)),
        ("observations", ("checkpoint",)),
        ("detector", ("checkpoint",)),
        ("motion", ("checkpoint", "stats")),
        ("camera", ("rotations",)),
    ):
        for key in keys:
            value = config.get(section, {}).get(key)
            if value is not None:
                config[section][key] = str((path.parent / value).resolve())
    return config


def body_release(config):
    release = config["pose"].get("release", "body")
    if release == "body":
        return body_config()
    if release == "body_sg6":
        return body_sg6_config()
    if not isinstance(release, dict):
        raise ValueError("Body video release must be body or a PoseReleaseConfig mapping")
    release = PoseReleaseConfig(**release)
    if release.task != "body":
        raise ValueError("Body video requires a body Pose release")
    return release


def motion_intrinsics(width, height, provided=None):
    """Motion deployment's diagonal focal estimate, on the normalized image."""
    if provided is not None:
        camera = torch.as_tensor(provided, dtype=torch.float32)
        if camera.shape != (3, 3):
            raise ValueError("Body video intrinsics must be a constant 3x3 matrix")
        return camera
    focal = float(np.hypot(width, height))
    return torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


@torch.inference_mode()
def extract_body_features(frames, boxes, intrinsics, model, *, batch_size, device):
    """Use the training crop and final query token; no body FK or mesh allocation."""
    if batch_size < 1:
        raise ValueError("Pose batch_size must be positive")
    tokens = []
    feature_version = None
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            batch = prepare_pose_batch(
                frames[start:stop],
                boxes[start:stop],
                intrinsics,
                device=device,
                crop_convention="motion_training",
            )
            condition = model.encode(batch)
            feature_version = condition.feature_version
            tokens.append(condition.pose_tokens.cpu())
    return torch.cat(tokens), feature_version


def build_body_conditions(frames, config, *, device="cpu", progress=None):
    """Extract CPU conditions on an already normalized 30Hz video timeline.

    Models are loaded per stage so their GPU allocations can be reused. Full
    native77 observations stay available for camera masking and cached replay;
    MotionPredictor owns selection of the trained native joint schema.
    """

    def stage(message):
        if progress is not None:
            progress(message)

    expected = MotionConditionSpec(**config["motion"]["condition_spec"])
    schema = get_observation_schema(expected.observation_joint_set)
    if schema.num_joints not in (29, 77):
        raise ValueError("Body video requires a native29 or native77 Motion checkpoint")
    release = body_release(config)
    actual = MotionConditionSpec(
        f"{release.pose_release_id}:pose_tokens",
        config["observations"]["release_id"],
        schema.name,
    )
    if actual != expected:
        raise ValueError(f"Body video feature/detector release mismatch: {actual} != {expected}")
    camera_config = config.get("camera", {})
    intrinsics = motion_intrinsics(frames.width, frames.height, camera_config.get("intrinsics"))
    stage("Detecting and tracking the primary person")
    options = dict(config["detector"])
    detector = YOLOXONNX(options.pop("checkpoint"), **options)
    tracks = detect_track(frames, detector, **config.get("tracking", {}))
    del detector

    stage("Extracting native77 2D observations")
    observations = config["observations"]
    detector = load_native77_detector(
        observations["checkpoint"],
        release_id=observations["release_id"],
        device=device,
        batch_size=observations.get("batch_size", 4),
    )
    keypoints = detector.predict(frames, tracks.boxes_xys)
    detector_release = detector.release_id
    del detector

    stage("Estimating camera rotation")
    rotations = camera_config.get("rotations")
    if rotations is not None:
        rotations = np.load(rotations)
    camera = camera_motion(
        frames,
        intrinsics.numpy(),
        tracks.boxes_xyxy,
        mode=camera_config.get("mode", "background"),
        rotations=rotations,
        keypoints=keypoints.numpy(),
        **camera_config.get("options", {}),
    )

    stage("Extracting frozen body Pose features")
    pose = load_pose_checkpoint(config["pose"]["checkpoint"], release, device=device)
    features, feature_version = extract_body_features(
        frames,
        tracks.boxes_xys,
        intrinsics,
        pose,
        batch_size=config["pose"].get("batch_size", 4),
        device=device,
    )
    del pose
    actual = MotionConditionSpec(feature_version, detector_release, schema.name)
    sequence = {
        "condition_spec": asdict(actual),
        "length": torch.tensor([len(frames)]),
        "f_imgseq": features.unsqueeze(0),
        "kp2d_native77": keypoints.unsqueeze(0),
        "bbx_xys": torch.from_numpy(tracks.boxes_xys).unsqueeze(0),
        "K_fullimg": intrinsics.expand(1, len(frames), 3, 3).clone(),
        "cam_angvel": camera["cam_angvel"].unsqueeze(0),
        "frame_times": (torch.arange(len(frames), dtype=torch.float64) / 30).unsqueeze(0),
        "sampling_fps": 30.0,
    }
    return sequence, tracks, camera


def run_body_video(input_path, config, *, device="cpu", start=0.0, duration=None, progress=None):
    """Normalize one <=60s clip and predict raw camera/world SOMA parameters.

    The returned `conditions` dictionary is directly replayable by `gem-infer
    motion-cache --task body`. Frame IDs refer to the normalized 30Hz clip,
    not source decode IDs.
    No contact/IK processing or mesh reconstruction is applied.
    """
    from .video import VideoFrames, normalize_video

    with TemporaryDirectory(prefix="gem-body-video-") as temporary:
        normalized = Path(temporary) / "input.mp4"
        if progress is not None:
            progress("Normalizing the input clip to 30fps")
        metadata = normalize_video(input_path, normalized, start=start, duration=duration)
        with VideoFrames(normalized) as frames:
            sequence, tracks, camera = build_body_conditions(
                frames, config, device=device, progress=progress
            )
        if progress is not None:
            progress("Predicting native temporal Motion")
        predictor = load_predictor(
            config["motion"]["checkpoint"],
            config["motion"],
            stats_path=config["motion"].get("stats"),
            device=device,
        )
        result = run_motion(predictor, sequence, device=device)
    # The normalized pixels are temporary; output metadata describes their
    # sampling convention without retaining a path that no longer exists.
    metadata["normalized"].pop("path", None)
    return tensor_tree(
        {
            **result,
            "format": "gem_body_video_predictions_v1",
            "video": metadata,
            "release": config,
            "conditions": sequence,
            "tracking": asdict(tracks),
            "camera": camera,
        }
    )
