# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable calibrated-camera numerics and perspective crop contracts."""

import json

import cv2
import numpy as np
import pytest

from gem.common.camera_models import PerspectiveCamera, camera_from_dict
from gem.common.fisheye import FisheyeCamera, perspective_crop

MODELS = ("PINHOLE", "RATIONAL_POLYNOMIAL", "KB4", "FISHEYE624")


def calibration(model, schema="gem"):
    K = [[410.0, 0, 319.5], [0, 385.0, 239.5], [0, 0, 1]]
    radial = [0.03, -0.002, 0.0005, -0.0001]
    if model == "PINHOLE":
        coefficients = []
    elif model == "RATIONAL_POLYNOMIAL":
        # All OpenCV terms are active: numerator/denominator, tangential,
        # thin prism, and two sensor-tilt angles.
        coefficients = [
            -0.08,
            0.015,
            0.001,
            -0.002,
            -0.001,
            0.02,
            -0.003,
            0.0002,
            0.0005,
            -0.0001,
            0.0003,
            -0.0001,
            0.015,
            -0.02,
        ]
    elif model == "KB4":
        coefficients = radial
    else:
        coefficients = (
            radial + [0.00002, -0.000001] + [0.001, -0.002, 0.0002, -0.0001, 0.0003, -0.0002]
        )
    if schema == "shared":
        return {
            "model": model,
            "image_width": 640,
            "image_height": 480,
            "camera_matrix": {"matrix_3x3": K},
            "distortion_coefficients": coefficients,
        }
    result = {"model": model.lower(), "width": 640, "height": 480, "intrinsics": K}
    if model in ("PINHOLE", "RATIONAL_POLYNOMIAL"):
        result["distortion_coefficients"] = coefficients
    else:
        count = 4 if model == "KB4" else 6
        result["radial"] = coefficients[:count]
        if model == "FISHEYE624":
            result["tangential_prism"] = coefficients[count:]
    return result


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("schema", ["gem", "shared"])
def test_calibration_json_readback_preserves_all_model_parameters(model, schema):
    camera = camera_from_dict(json.loads(json.dumps(calibration(model, schema))))
    expected_type = (
        PerspectiveCamera if model in ("PINHOLE", "RATIONAL_POLYNOMIAL") else FisheyeCamera
    )
    assert isinstance(camera, expected_type)
    assert (camera.width, camera.height, camera.model_name) == (640, 480, model)
    np.testing.assert_array_equal(camera.K, calibration(model)["intrinsics"])
    assert camera.to_dict() == calibration(model)
    restored = camera_from_dict(json.loads(json.dumps(camera.to_dict())))
    rays = np.array([[0, 0, 1], [-0.5, 0.2, 1], [0.6, -0.4, 0.8]])
    np.testing.assert_array_equal(restored.project(rays), camera.project(rays))
    np.testing.assert_array_equal(
        restored.unproject(camera.project(rays)), camera.unproject(camera.project(rays))
    )


@pytest.mark.parametrize("model", MODELS)
def test_camera_roundtrip_with_batched_off_axis_rays_and_unequal_focals(model):
    camera = camera_from_dict(calibration(model))
    rays = np.array(
        [
            [[0, 0, 1], [-0.5, 0.2, 1.2], [0.7, -0.4, 0.8]],
            [[0.1, 0.6, 2], [-0.8, -0.5, 1], [0.4, 0.3, 0.7]],
        ]
    )
    directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    pixels = camera.project(rays)
    assert pixels.shape == (2, 3, 2)
    assert np.isfinite(pixels).all()
    actual = camera.unproject(pixels)
    assert actual.shape == rays.shape
    np.testing.assert_allclose(actual, directions, rtol=0, atol=1e-8)
    np.testing.assert_allclose(camera.project(actual), pixels, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(pixels[0, 0], camera.K[:2, 2])


@pytest.mark.parametrize("model", ["RATIONAL_POLYNOMIAL", "KB4"])
def test_projection_matches_opencv_with_every_distortion_term(model):
    camera = camera_from_dict(calibration(model))
    random = np.random.default_rng(75)
    rays = np.column_stack([random.uniform(-0.8, 0.8, (40, 2)), random.uniform(0.8, 2, 40)])
    if model == "RATIONAL_POLYNOMIAL":
        expected, _ = cv2.projectPoints(
            rays, np.zeros(3), np.zeros(3), camera.K, camera.distortion_coefficients
        )
    else:
        expected, _ = cv2.fisheye.projectPoints(
            rays[:, None], np.zeros(3), np.zeros(3), camera.K, camera.radial
        )
    np.testing.assert_allclose(camera.project(rays), expected[:, 0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("radial_strength", [2.0, 10.0])
def test_strong_rational_distortion_inverts_to_original_forward_ray(radial_strength):
    values = calibration("RATIONAL_POLYNOMIAL")
    values["distortion_coefficients"] = [radial_strength] + [0.0] * 13
    camera = camera_from_dict(values)
    rays = np.array([[0.8, 0.3, 1], [1.0, -0.1, 1], [-0.7, 0.4, 1]])
    pixels = camera.project(rays)
    recovered = camera.unproject(pixels)
    np.testing.assert_allclose(
        recovered, rays / np.linalg.norm(rays, axis=1, keepdims=True), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(camera.project(recovered), pixels, rtol=0, atol=1e-6)


@pytest.mark.parametrize("branch", ["radial_fold", "radial_pole", "tangential_fold"])
def test_rational_rejects_nonphysical_forward_branches(branch):
    values = calibration("RATIONAL_POLYNOMIAL")
    coefficients = np.zeros(14)
    if branch == "radial_fold":
        coefficients[0] = -1
        valid_ray, invalid_ray = [0.4, 0, 1], [0.6, 0, 1]
    elif branch == "radial_pole":
        coefficients[5] = -1
        valid_ray, invalid_ray = [0.8, 0, 1], [1.1, 0, 1]
    else:
        coefficients[3] = 1
        valid_ray, invalid_ray = [-0.1, 0, 1], [-0.3, 0, 1]
    values["distortion_coefficients"] = coefficients.tolist()
    camera = camera_from_dict(values)
    pixels = camera.project([valid_ray, invalid_ray, [0, 0, -1], [np.nan, 0, 1]])
    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1:]).all()
    np.testing.assert_allclose(
        camera.unproject(pixels[0]), np.array(valid_ray) / np.linalg.norm(valid_ray), atol=1e-9
    )
    if branch == "radial_fold":
        # r-r^3 reaches only 2/(3*sqrt(3)) on its physical radial branch.
        unreachable = camera.K[:2, 2] + [camera.K[0, 0] * 0.45, 0]
        assert np.isnan(camera.unproject(unreachable)).all()


@pytest.mark.parametrize("model", MODELS)
def test_off_axis_perspective_crop_preserves_rgb_and_covers_roi_perimeter(model):
    camera = camera_from_dict(calibration(model, "shared"))
    color = np.array([23, 79, 173], np.uint8)
    image = np.broadcast_to(color, (480, 640, 3)).copy()
    bbox = np.array([460, 155, 100])
    crop, K, crop_bbox, rotation = perspective_crop(image, bbox, camera, crop_side_pixels=64)
    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    np.testing.assert_array_equal(crop[32, 32], color)
    np.testing.assert_array_equal(crop_bbox, [31.5, 31.5, 64])
    np.testing.assert_allclose(rotation[:, 2], camera.unproject(bbox[:2]), atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert K[0, 0] == K[1, 1]
    np.testing.assert_array_equal(K[:2, 2], [31.5, 31.5])

    edge = np.linspace(-50, 50, 257)
    perimeter = np.concatenate(
        [
            np.column_stack([460 + edge, np.full(257, 105)]),
            np.column_stack([460 + edge, np.full(257, 205)]),
            np.column_stack([np.full(257, 410), 155 + edge]),
            np.column_stack([np.full(257, 510), 155 + edge]),
        ]
    )
    virtual_rays = camera.unproject(perimeter) @ rotation
    projected = virtual_rays @ K.T
    projected = projected[:, :2] / projected[:, 2, None]
    assert np.isfinite(projected).all()
    assert projected.min() >= -0.5 - 1e-9
    assert projected.max() <= 63.5 + 1e-9
