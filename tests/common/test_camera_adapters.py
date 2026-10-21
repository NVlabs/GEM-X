# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical checks for calibrated image-plane and fisheye adapters."""

import cv2
import numpy as np
import pytest

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.fisheye import FisheyeCamera, perspective_crop


def test_square_pixel_rays_and_roundtrip_use_realized_half_pixel_scale():
    K = np.array([[501.3, 0, 301.25], [0, 720.2, 238.4], [0, 0, 1]])
    adapter = SquarePixelTransform.from_intrinsics(K, 641, 479)
    assert (adapter.width, adapter.height) == (921, 479)
    rays = np.array([[-0.2, 0.1, 1], [0.4, -0.3, 1], [0, 0, 1]])
    source_pixels = (rays @ K.T)[:, :2]
    resized_pixels = (rays @ adapter.intrinsics.T)[:, :2]
    np.testing.assert_allclose(adapter.transform_points(source_pixels), resized_pixels, atol=4e-5)
    np.testing.assert_allclose(adapter.inverse_points(resized_pixels), source_pixels, atol=4e-5)
    boxes = np.array([[312.2, 246.4, 120], [3, 5, 25]], dtype=np.float32)
    corrected = adapter.transform_boxes(boxes)
    np.testing.assert_allclose(adapter.inverse_boxes(corrected), boxes, atol=3e-5)
    np.testing.assert_allclose(corrected[:, 2], boxes[:, 2] * 921 / 641)


def test_square_pixel_identity_preserves_image():
    K = np.array([[578.3819, 0, 336.98083], [0, 578.2392, 215.1156], [0, 0, 1]])
    adapter = SquarePixelTransform.from_intrinsics(K, 672, 480)
    image = np.zeros((480, 672, 3), dtype=np.uint8)
    assert adapter.is_identity
    assert adapter.resize_image(image) is image
    np.testing.assert_array_equal(adapter.intrinsics, K.astype(np.float32))


def test_square_pixel_image_interpolation_matches_half_pixel_coordinates():
    # A linear coordinate ramp lets us test where OpenCV actually sampled the
    # source, away from the constant border where linear extrapolation ends.
    K = [[8, 0, 3.5], [0, 12, 2.5], [0, 0, 1]]
    adapter = SquarePixelTransform.from_intrinsics(K, 8, 6)
    image = np.broadcast_to(np.arange(8, dtype=np.float32)[None, :, None], (6, 8, 3)).copy()
    resized = adapter.resize_image(image)
    output_pixels = np.stack([np.arange(1, 11), np.zeros(10)], axis=-1)
    source_pixels = adapter.inverse_points(output_pixels)
    np.testing.assert_allclose(resized[0, 1:-1, 0], source_pixels[:, 0], atol=1e-6)


@pytest.mark.parametrize("model", ["KB4", "FISHEYE624"])
def test_fisheye_projection_unprojection_and_calibration_roundtrip(model):
    camera = _camera(model)
    rays = np.array([[-0.7, 0.2, 0.8], [0.9, -0.5, 0.4], [0, 0, 1], [0.1, 0.5, 1]])
    expected_directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        camera.unproject(camera.project(rays)), expected_directions, atol=2e-8
    )
    restored = FisheyeCamera.from_dict(camera.to_dict())
    np.testing.assert_array_equal(restored.project(rays), camera.project(rays))
    assert restored.model_name == model


def test_kb4_projection_matches_opencv():
    camera = _camera("KB4")
    points = np.array([[-0.7, 0.2, 0.8], [0.9, -0.5, 0.4], [0, 0, 1], [0.1, 0.5, 1]])
    expected, _ = cv2.fisheye.projectPoints(
        points[:, None], np.zeros(3), np.zeros(3), camera.K, camera.radial
    )
    np.testing.assert_allclose(camera.project(points), expected[:, 0], atol=1e-10)


@pytest.mark.parametrize("model", ["KB4", "FISHEYE624"])
def test_fisheye_invalid_rays_and_pixels(model):
    camera = _camera(model)
    assert np.isnan(camera.project([[1, 0, -1], [np.nan, 0, 1]])).all()
    assert np.isnan(camera.unproject([[np.nan, 0], [1e8, 1e8]])).all()


def test_fisheye_radial_domain_is_angle_not_distorted_radius():
    # Positive radial distortion can push a valid forward ray beyond a
    # normalized pixel radius of pi/2; inverse projection must retain it.
    camera = FisheyeCamera(np.eye(3), [0.5, 0, 0, 0], 20, 20)
    ray = np.array([np.sin(1.2), 0, np.cos(1.2)])
    pixel = camera.project(ray)
    assert pixel[0] > np.pi / 2
    np.testing.assert_allclose(camera.unproject(pixel), ray, atol=1e-9)


@pytest.mark.parametrize("model", ["KB4", "FISHEYE624"])
def test_perspective_crop_looks_at_box_center_and_preserves_rgb(model):
    camera = _camera(model)
    image = np.broadcast_to(np.array([31, 73, 151], dtype=np.uint8), (480, 640, 3)).copy()
    bbox = np.array([440, 170, 90])
    crop, K, crop_bbox, rotation = perspective_crop(image, bbox, camera, 64)
    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    np.testing.assert_array_equal(crop[32, 32], [31, 73, 151])
    np.testing.assert_allclose(
        rotation @ np.array([0, 0, 1]), camera.unproject(bbox[:2]), atol=1e-12
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-12)
    np.testing.assert_array_equal(crop_bbox, [31.5, 31.5, 64])
    assert K[0, 0] == K[1, 1]
    np.testing.assert_array_equal(K[:2, 2], [31.5, 31.5])
    # Dense edge rays must fit inside the full virtual crop with half-pixel
    # boundary coordinates, even when the fisheye edge bows between corners.
    edge = np.linspace(-45, 45, 257)
    pixels = np.concatenate(
        [
            np.column_stack([440 + edge, np.full(257, 125)]),
            np.column_stack([440 + edge, np.full(257, 215)]),
            np.column_stack([np.full(257, 395), 170 + edge]),
            np.column_stack([np.full(257, 485), 170 + edge]),
        ]
    )
    virtual = camera.unproject(pixels) @ rotation
    projected = virtual @ K.T
    projected = projected[:, :2] / projected[:, 2, None]
    assert projected.min() >= -0.5 - 1e-9
    assert projected.max() <= 63.5 + 1e-9


def _camera(model):
    K = np.array([[320, 0, 319.5], [0, 335, 239.5], [0, 0, 1]])
    radial = [0.03, -0.002, 0.0005, -0.0001]
    offsets = None
    if model == "FISHEYE624":
        radial += [0.00002, -0.000001]
        offsets = [0.001, -0.002, 0.0002, -0.0001, 0.0003, -0.0002]
    return FisheyeCamera(K, radial, 640, 480, offsets, model_name=model)
