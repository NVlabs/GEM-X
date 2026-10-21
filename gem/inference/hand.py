# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand detection and saved-box providers for the existing Pose commands."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.detectors.hand import RTDETRHandONNX
from gem.common.detectors.hand_boxes import BBoxSmoothingConfig, smooth_hand_boxes
from gem.common.detectors.hand_tracking import track_hand_detections

from .inputs import input_frames as _input_frames
from .inputs import load_mapping
from .inputs import pose_intrinsics as _intrinsics


def _records(count, camera):
    return {index: {"intrinsics": camera, "instances": []} for index in range(count)}


def _instance(side, bbox, observed=True):
    return {
        "track_id": side,
        "side": side,
        "bbox": [float(value) for value in bbox],
        "bbox_observed": bool(observed),
    }


def detect_hand_annotations(
    input_path,
    checkpoint,
    *,
    video=False,
    intrinsics=None,
    device="cpu",
    tracker="native",
    batch_size=8,
    hands=("left", "right"),
    square_pixels=False,
    smooth_boxes=False,
    undistort_camera=None,
):
    """One egocentric left/right pair, with the original frame and crop policies.

    Native video tracking sees the full timeline and both detector classes
    before requested sides are selected. Framewise mode uses anatomical labels,
    per-class top1 and side×1.3 crops; no missing hand is synthesized there.
    Optional full-image undistortion precedes pixel-aspect correction and
    detection/tracking. Returned boxes undo only the pixel-aspect resize: they
    remain in the undistorted raster when ``undistort_camera`` is supplied,
    otherwise in the original source raster. The calibration owns the base K.
    ``smooth_boxes`` applies the offline crop filter to native video tracks,
    using the video's physical frame times before source-coordinate mapping.
    """
    if tracker not in ("native", "none"):
        raise ValueError("Hand tracker must be native or none")
    if batch_size < 1:
        raise ValueError("Hand detector batch size must be positive")
    tracked = video and tracker == "native"
    if smooth_boxes and not tracked:
        raise ValueError("Hand bbox smoothing requires native video tracking")
    undistortion = None
    if undistort_camera is not None:
        from gem.common.camera_models import camera_from_dict
        from gem.common.undistortion import FullImageUndistortion

        undistortion = FullImageUndistortion(camera_from_dict(undistort_camera))
    detector = RTDETRHandONNX(
        checkpoint,
        device=device,
        score_thr=0.15 if tracked else 0.5,
        top1_per_class=not tracked,
        pad=0.0 if tracked else 0.15,
        square=not tracked,
        clip=tracked,
    )
    with _input_frames(input_path, video) as frames:
        height, width = frames[0].shape[:2]
        camera = _intrinsics(
            width, height, undistortion.K if undistortion is not None else intrinsics
        )
        resize = (
            SquarePixelTransform.from_intrinsics(camera, width, height) if square_pixels else None
        )
        detection_width = resize.width if resize is not None else width
        detection_height = resize.height if resize is not None else height
        annotations = _records(len(frames), camera)
        frame_times = np.arange(len(frames)) / frames.fps if smooth_boxes else None
        candidates = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            if undistortion is not None:
                batch = [undistortion.apply(image) for image in batch]
            if resize is not None:
                batch = [resize.resize_image(image) for image in batch]
            candidates.extend(detector.detect_batch(batch))
    annotation_coordinates = "undistorted_image" if undistortion is not None else "source_image"
    if resize is not None:
        candidate_coordinates = (
            "square_pixel_undistorted_image" if undistortion is not None else "square_pixel_image"
        )
    else:
        candidate_coordinates = annotation_coordinates
    metadata = {
        "provider": "rtdetr_hand",
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": device,
        "tracking": "native" if tracked else "framewise",
        "camera_source": (
            "provided"
            if intrinsics is not None or undistortion is not None
            else "pose_max_side_estimate"
        ),
        "image_size": [width, height],
        "detection_image_size": [detection_width, detection_height],
        "raw_candidate_coordinates": candidate_coordinates,
        "annotation_coordinates": annotation_coordinates,
        "raw_candidates": [[asdict(item) for item in frame] for frame in candidates],
    }
    if undistortion is not None:
        metadata["full_image_undistortion"] = True
    if tracked:
        tracks = track_hand_detections(candidates, detection_width)
        if smooth_boxes:
            config = BBoxSmoothingConfig()
            tracks = smooth_hand_boxes(tracks, frame_times, config)
            metadata["bbox_smoothing"] = asdict(config)
        metadata["tracking_stats"] = tracks.stats
        metadata["side_policy"] = "egocentric_screen_order_with_native_association"
        for frame_id, record in annotations.items():
            for side in hands:
                if tracks.emitted[side][frame_id]:
                    record["instances"].append(
                        _instance(
                            side,
                            tracks.boxes_xys[side][frame_id],
                            tracks.raw_present[side][frame_id],
                        )
                    )
    else:
        metadata["side_policy"] = "detector_anatomical_labels"
        for frame_id, frame in enumerate(candidates):
            by_side = {detection.side: detection for detection in frame if detection.side in hands}
            for side in hands:
                if side in by_side:
                    detection = by_side[side]
                    annotations[frame_id]["instances"].append(
                        {
                            **_instance(side, detection.cxcys),
                            "detection_score": detection.score,
                        }
                    )
    if resize is not None:
        metadata["square_pixels"] = {
            "active": not resize.is_identity,
            "intrinsics": resize.intrinsics.tolist(),
        }
        for record in annotations.values():
            for instance in record["instances"]:
                instance["bbox"] = resize.inverse_boxes(instance["bbox"]).tolist()
    return annotations, metadata


def saved_hand_annotations(
    input_path,
    bbox_root,
    *,
    video=False,
    intrinsics=None,
    hands=("left", "right"),
):
    """Read GA ``root/stem/stem_{left,right}_bbox.json`` without modifying boxes.

    Missing sides or frame IDs remain absent. These files already contain final
    square crops; no tracking, padding, smoothing or clamping is applied here.
    """
    stem = Path(input_path).stem
    paths = {side: Path(bbox_root) / stem / f"{stem}_{side}_bbox.json" for side in hands}
    found = {side: path for side, path in paths.items() if path.is_file()}
    if not found:
        raise FileNotFoundError(f"No saved hand box files under {Path(bbox_root) / stem}")
    with _input_frames(input_path, video) as frames:
        height, width = frames[0].shape[:2]
        annotations = _records(len(frames), _intrinsics(width, height, intrinsics))
    for side, path in found.items():
        seen = set()
        for key, box in load_mapping(path).items():
            if not str(key).isdecimal():
                raise ValueError("Saved hand frame IDs must be non-negative integers")
            frame_id = int(key)
            if frame_id in seen:
                raise ValueError(f"Duplicate saved hand frame ID {frame_id}")
            seen.add(frame_id)
            if frame_id not in annotations:
                raise ValueError(f"Saved hand frame {frame_id} is outside the input timeline")
            annotations[frame_id]["instances"].append(_instance(side, box))
    return annotations, {
        "provider": "saved_hand_boxes",
        "root": str(Path(bbox_root).resolve()),
        "tracking": "supplied",
        "camera_source": "provided" if intrinsics is not None else "pose_max_side_estimate",
        "image_size": [width, height],
    }
