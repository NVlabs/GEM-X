# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background-based camera rotation for uploaded monocular demo videos.

Uses the existing SimpleVO SIFT/Lowe matching pattern with an OpenCV
homography fit. The demo samples each frame and masks the visible subject
using its confident person-keypoint observations. The polar factor of K^-1 H K
estimates rotation, followed by SciPy SLERP for intermediate frames. This is
an approximate rotation-only demo estimator, not benchmark GT camera motion;
translation, foreground motion and changing focal length can affect it.
"""

from os import PathLike

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from gem.common.geometry.rotations import matrix_to_rotation_6d


def rotation_from_homography(homography, intrinsics):
    """Return the closest proper rotation to a calibrated image homography."""
    calibrated = np.linalg.solve(intrinsics, homography @ intrinsics)
    calibrated /= np.cbrt(np.linalg.det(calibrated))
    u, _, vt = np.linalg.svd(calibrated)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _rgb_frames(frames):
    if isinstance(frames, str | PathLike):
        capture = cv2.VideoCapture(str(frames))
        if not capture.isOpened():
            raise OSError(f"Cannot open video {frames}")
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
    else:
        yield from frames


def _mask_box(mask, scaled_box):
    x1, y1, x2, y2 = scaled_box
    pad_x, pad_y = (x2 - x1) * 0.05, (y2 - y1) * 0.05
    top_left = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)))
    bottom_right = (
        min(mask.shape[1] - 1, int(x2 + pad_x)),
        min(mask.shape[0] - 1, int(y2 + pad_y)),
    )
    cv2.rectangle(mask, top_left, bottom_right, 0, thickness=-1)


def background_mask(shape, box, *, image_scale, keypoints=None, other_person_boxes=None):
    """Deployed primary hull/box mask, optionally unioned with other people.

    Other-person masking is an explicit variation; primary-only is the default.
    """
    mask = np.full(shape, 255, dtype=np.uint8)
    x1, y1, x2, y2 = np.asarray(box) * image_scale
    points = None
    if keypoints is not None:
        points = keypoints[keypoints[:, 2] > 0.5, :2] * image_scale
    if points is not None and len(points) >= 6:
        person = np.zeros(shape, dtype=np.uint8)
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(person, hull, 255)
        radius = max(3, round(max(x2 - x1, y2 - y1) * 0.035))
        person = cv2.dilate(
            person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        )
        mask[person > 0] = 0
    else:
        _mask_box(mask, (x1, y1, x2, y2))
    for other in () if other_person_boxes is None else other_person_boxes:
        _mask_box(mask, np.asarray(other) * image_scale)
    return mask


def estimate_camera_rotations(
    frames,
    intrinsics,
    boxes,
    step=1,
    image_scale=0.5,
    return_diagnostics=False,
    keypoints=None,
    other_person_boxes=None,
):
    """Rotation-only estimate on the caller's normalized30Hz clip timeline.

    RGB arrays or a video path are accepted. Constant K and30Hz frame sampling
    match the deployed estimator; a gap over30frames raises instead of silently
    selecting a static camera. Caller supplies boxes for every decoded frame.
    """
    frame_count = len(boxes)
    if frame_count < 2:
        raise ValueError("Background camera estimation needs at least two frames")
    sample_indices = np.unique(np.r_[np.arange(0, frame_count, step), frame_count - 1])
    selected = set(sample_indices.tolist())
    sift = cv2.SIFT_create()
    observations = []
    observed_indices = []
    skipped = []
    for frame_index, rgb in enumerate(_rgb_frames(frames)):
        if frame_index >= frame_count:
            break
        if frame_index in selected:
            gray = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                None,
                fx=image_scale,
                fy=image_scale,
                interpolation=cv2.INTER_AREA,
            )
            mask = background_mask(
                gray.shape,
                boxes[frame_index],
                image_scale=image_scale,
                keypoints=None if keypoints is None else keypoints[frame_index],
                other_person_boxes=(
                    None if other_person_boxes is None else other_person_boxes[frame_index]
                ),
            )
            cv_keypoints, descriptors = sift.detectAndCompute(gray, mask)
            if descriptors is None or len(cv_keypoints) < 8:
                skipped.append(
                    {
                        "frame": frame_index,
                        "reason": "foreground_occlusion_or_no_background_features",
                    }
                )
            else:
                observations.append((cv_keypoints, descriptors))
                observed_indices.append(frame_index)
    if not observed_indices or observed_indices[0] != 0 or observed_indices[-1] != frame_count - 1:
        raise ValueError(
            "The segment starts or ends without enough visible background to estimate camera motion; choose a different segment or the static camera setting for a fixed camera clip"
        )
    K = np.asarray(intrinsics, dtype=np.float64).copy()
    K[:2] *= image_scale
    rotations = [np.eye(3)]
    accepted_indices = [0]
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    previous = observations[0]
    for current_index, current in zip(observed_indices[1:], observations[1:]):
        kp0, descriptors0 = previous
        kp1, descriptors1 = current
        matches = matcher.knnMatch(descriptors0, descriptors1, k=2)
        good = [
            a
            for pair in matches
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < 0.7 * b.distance
        ]
        if len(good) < 8:
            skipped.append({"frame": current_index, "reason": "insufficient_background_matches"})
            continue
        points0 = np.float32([kp0[item.queryIdx].pt for item in good])
        points1 = np.float32([kp1[item.trainIdx].pt for item in good])
        homography, inliers = cv2.findHomography(points0, points1, cv2.RANSAC, 2.0)
        if homography is None or int(inliers.sum()) < 8:
            skipped.append({"frame": current_index, "reason": "insufficient_homography_inliers"})
            continue
        if current_index - accepted_indices[-1] > 30:
            raise ValueError(
                "Background camera tracking was unavailable for more than one second; choose a different segment or use the static camera setting for a fixed camera clip"
            )
        relative_rotation = rotation_from_homography(homography, K)
        rotations.append(relative_rotation @ rotations[-1])
        accepted_indices.append(current_index)
        previous = current
    if accepted_indices[-1] != frame_count - 1:
        raise ValueError(
            "Background camera tracking was lost at the end of the segment; choose a shorter segment"
        )
    result = (
        Slerp(accepted_indices, Rotation.from_matrix(rotations))(np.arange(frame_count))
        .as_matrix()
        .astype(np.float32)
    )
    diagnostics = {
        "method": "background_sift_homography_rotation",
        "sample_step": step,
        "sampling": "dense_per_frame" if step == 1 else "sampled",
        "person_mask": "confident_keypoint_hull" if keypoints is not None else "tracked_box",
        "mask_scope": (
            "primary_only" if other_person_boxes is None else "primary_and_other_tracked_people"
        ),
        "accepted_keyframes": accepted_indices,
        "skipped_samples": skipped,
        "maximum_interpolation_gap_frames": int(np.diff(accepted_indices).max()),
        "maximum_allowed_gap_frames": 30,
    }
    return (result, diagnostics) if return_diagnostics else result


def camera_angular_velocity(rotations):
    """Source-interval6D rotation; terminal value repeats the last interval."""
    rotations = torch.as_tensor(rotations, dtype=torch.float32)
    if len(rotations) == 1:
        return matrix_to_rotation_6d(torch.eye(3, device=rotations.device)).unsqueeze(0)
    increments = matrix_to_rotation_6d(rotations[1:] @ rotations[:-1].transpose(-1, -2))
    return torch.cat((increments, increments[-1:]), dim=0)


def camera_motion(frames, intrinsics, boxes, *, mode, rotations=None, **options):
    """Return R_w2c[T,3,3]/cam_angvel[T,6] tensors plus method diagnostics.

    Modes are explicit. Background failure never falls back to static.
    Provided rotations belong to the same normalized30Hz decoded timeline.
    """
    if mode == "static":
        result = torch.eye(3).repeat(len(boxes), 1, 1)
        diagnostics = {"method": "static", "skipped_samples": []}
    elif mode == "provided":
        result = torch.as_tensor(rotations, dtype=torch.float32)
        if result.shape != (len(boxes), 3, 3):
            raise ValueError("Provided R_w2c must have shape (T,3,3) on the decoded timeline")
        diagnostics = {"method": "provided_R_w2c", "skipped_samples": []}
    elif mode == "background":
        result, diagnostics = estimate_camera_rotations(
            frames, intrinsics, boxes, return_diagnostics=True, **options
        )
        result = torch.from_numpy(result)
    else:
        raise ValueError("Camera mode must be static, background, or provided")
    return {
        "R_w2c": result,
        "cam_angvel": camera_angular_velocity(result),
        "diagnostics": diagnostics,
    }
