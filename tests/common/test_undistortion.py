# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-image remapping and restoration of predictions to the source camera."""

from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.camera_models import camera_from_dict
from gem.common.undistortion import FullImageUndistortion
from gem.inference.camera import restore_source_camera_predictions


def camera_for(model):
    values = {
        "model": model,
        "width": 160,
        "height": 192,
        "intrinsics": [[150, 0, 79.5], [0, 157, 95.5], [0, 0, 1]],
    }
    if model == "PINHOLE":
        values["distortion_coefficients"] = []
    elif model == "RATIONAL_POLYNOMIAL":
        values["distortion_coefficients"] = [
            0.1,
            0.01,
            0.002,
            -0.003,
            0.001,
            0.002,
            0,
            0,
            0.001,
            0,
            -0.001,
            0,
            0.01,
            -0.015,
        ]
    else:
        values["radial"] = [0.02, -0.001, 0.0001, 0]
        if model == "FISHEYE624":
            values["radial"] += [0, 0]
            values["tangential_prism"] = [0.002, -0.001, 0.0003, 0, -0.0002, 0]
    return camera_from_dict(values)


def rgb_image(camera):
    yy, xx = np.indices((camera.height, camera.width))
    return np.stack([(3 * xx + yy) % 256, (xx + 2 * yy) % 256, np.full_like(xx, 173)], -1).astype(
        np.uint8
    )


def reference_project(camera, rays):
    flat = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    if camera.model_name == "RATIONAL_POLYNOMIAL":
        result, _ = cv2.projectPoints(
            flat, np.zeros(3), np.zeros(3), camera.K, camera.distortion_coefficients
        )
    elif camera.model_name == "KB4":
        result, _ = cv2.fisheye.projectPoints(
            flat[:, None], np.zeros(3), np.zeros(3), camera.K, camera.radial
        )
    else:
        result = camera.project(flat)
    return result.reshape(*rays.shape[:-1], 2)


def test_pinhole_undistortion_preserves_rgb_raster_and_boxes():
    camera = camera_for("PINHOLE")
    adapter = FullImageUndistortion(camera)
    image = rgb_image(camera)
    before = image.copy()
    np.testing.assert_array_equal(adapter.apply(image), image)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(adapter.K, camera.K.astype(np.float32))
    assert adapter.K.dtype == np.float32
    assert (adapter.width, adapter.height) == (camera.width, camera.height)
    boxes = np.array([[79.5, 95.5, 40], [10, 170, 16]], np.float32)
    mapped, valid = adapter.source_boxes(boxes)
    np.testing.assert_array_equal(mapped, boxes)
    np.testing.assert_array_equal(valid, [True, True])
    assert mapped.dtype == np.float32
    assert valid.dtype == bool


@pytest.mark.parametrize("model", ["RATIONAL_POLYNOMIAL", "KB4", "FISHEYE624"])
def test_full_rgb_remap_matches_source_projection_across_row_strip_boundary(model):
    camera = camera_for(model)
    adapter = FullImageUndistortion(camera)
    image = rgb_image(camera)
    yy, xx = np.indices(image.shape[:2], dtype=np.float64)
    homogeneous = np.stack([xx, yy, np.ones_like(xx)], -1)
    rays = homogeneous @ np.linalg.inv(camera.K.astype(np.float32).astype(np.float64)).T
    source = reference_project(camera, rays).astype(np.float32)
    expected = cv2.remap(
        image,
        source[..., 0],
        source[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    actual = adapter.apply(image)
    assert actual.shape == image.shape
    assert actual.dtype == image.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[127:129], expected[127:129])
    assert np.all(actual[80:110, 60:100, 2] == 173)


@pytest.mark.parametrize("model", ["RATIONAL_POLYNOMIAL", "KB4", "FISHEYE624"])
def test_source_boxes_enclose_exact_17_by_17_sampled_source_region(model):
    camera = camera_for(model)
    adapter = FullImageUndistortion(camera)
    boxes = np.array([[110, 60, 50], [35, 140, 60]], np.float32)
    before = boxes.copy()
    mapped, valid = adapter.source_boxes(boxes)
    expected = []
    for cx, cy, side in boxes:
        xs, ys = np.meshgrid(
            np.linspace(cx - side / 2, cx + side / 2, 17),
            np.linspace(cy - side / 2, cy + side / 2, 17),
        )
        rays = (
            np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(adapter.K.astype(np.float64)).T
        )
        pixels = reference_project(camera, rays).reshape(-1, 2)
        lo, hi = pixels.min(0), pixels.max(0)
        expected.append([*((lo + hi) / 2), np.max(hi - lo)])
    np.testing.assert_allclose(mapped, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(boxes, before)


class InteriorExtremaCamera:
    """Small synthetic camera whose ROI x-maximum lies inside the square."""

    K = np.eye(3)
    width, height = 8, 8

    def project(self, rays):
        rays = np.asarray(rays)
        xy = rays[..., :2] / rays[..., 2:]
        pixels = np.stack([10 - (xy[..., 0] - 1) ** 2, xy[..., 1]], -1)
        pixels[xy[..., 0] < 0] = np.nan
        return pixels


def test_source_box_sampling_retains_interior_extrema_and_partial_valid_regions():
    adapter = FullImageUndistortion(InteriorExtremaCamera())
    boxes = np.array([[1, 1, 2], [0, 1, 2], [-3, 1, 1]], np.float32)
    mapped, valid = adapter.source_boxes(boxes)
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_array_equal(mapped[:2], [[9.5, 1, 2], [9.5, 1, 2]])
    assert np.isnan(mapped[2]).all()


def test_source_boxes_accept_shared_effective_intrinsics():
    camera = camera_for("PINHOLE")
    adapter = FullImageUndistortion(camera)
    boxes = np.array([[89.5, 90.5, 40], [109.5, 70.5, 40]], np.float32)
    intrinsics = adapter.K.copy()
    intrinsics[0, 0] *= 2
    intrinsics[1, 1] *= 2
    intrinsics[:2, 2] += [10, -5]
    mapped, valid = adapter.source_boxes(boxes, intrinsics=intrinsics)
    np.testing.assert_allclose(mapped, [[79.5, 95.5, 20], [89.5, 85.5, 20]], atol=1e-5)
    assert valid.all()


def prediction_row(camera, *, tensor=False):
    def array(value):
        value = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(value) if tensor else value

    joints = [
        [0.1, 0.05, 1],
        [-0.1, 0.05, 0.8],
        [0, 0, 5e-5],
        [0, 0, -1],
        [np.nan, 0, 1],
        [0, 0, 0.05],
    ]
    row = {
        "track_id": "right0",
        "side": "right",
        "bbox_observed": True,
        "bbox": array([110, 70, 50]),
        "intrinsics": array(camera.K),
        "parameters": {"wrist_orient": array(np.eye(3)), "identity_coeffs": array([0.2, -0.1])},
        "camera_translation": array([0.5, 0.25, 2]),
        "joints_camera": array(joints),
        "vertices_camera": array(joints[:2]),
        "keypoints_2d": array(np.zeros((6, 2))),
        "keypoints_crop": array(np.zeros((6, 2))),
        "projection_valid": torch.zeros(6, dtype=torch.bool) if tensor else np.zeros(6, bool),
    }
    smooth = {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in ("track_id", "side", "bbox_observed", "bbox", "intrinsics")
    }
    smooth["joints_camera"][0, 0] += 0.03
    row["smoothed"] = smooth
    return row


def assert_same_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, baseline in zip(actual, expected):
            assert_same_tree(value, baseline)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("tensor", [False, True])
def test_restore_raw_and_smoothed_source_projection_preserves_3d_and_input(tensor):
    camera = camera_for("RATIONAL_POLYNOMIAL")
    row = prediction_row(camera, tensor=tensor)
    result = {
        "task": "hand",
        "source_fps": 20.0,
        "camera_adapter": {"square_pixels": False},
        "box_provider": {"annotation_coordinates": "undistorted_image", "custom": "retained"},
        "frames": [
            {
                "frame_id": 3,
                "image_size": [160, 192],
                "intrinsics": row["intrinsics"],
                "instances": [row],
            }
        ],
    }
    if tensor:
        result["release"] = {"inference": {"minimum_depth": 0.1}}
    minimum_depth = 0.1 if tensor else 1e-4
    before = deepcopy(result)
    restored = restore_source_camera_predictions(result, camera)
    assert_same_tree(result, before)
    frame = restored["frames"][0]
    actual = frame["instances"][0]
    assert frame["frame_id"] == 3
    assert frame["image_size"] == [160, 192]
    assert frame["camera"] == camera.to_dict()
    assert frame["intrinsics"] is actual["intrinsics"] is None
    expected_bbox = FullImageUndistortion(camera).source_boxes(np.asarray(row["bbox"])[None])[0][0]
    np.testing.assert_allclose(actual["bbox"], expected_bbox, atol=1e-5)
    for variant, original in ((actual, row), (actual["smoothed"], row["smoothed"])):
        points = np.asarray(original["joints_camera"], dtype=np.float64)
        valid = np.isfinite(points).all(-1) & (points[:, 2] > minimum_depth)
        expected_pixels = np.full((len(points), 2), np.nan)
        expected_pixels[valid] = camera.project(points[valid])
        valid &= np.isfinite(expected_pixels).all(-1)
        expected_pixels[~valid] = np.nan
        np.testing.assert_allclose(variant["keypoints_2d"], expected_pixels, rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            variant["keypoints_crop"],
            (expected_pixels - expected_bbox[:2]) / expected_bbox[2],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(variant["projection_valid"], valid)
        for field in ("parameters", "camera_translation", "joints_camera", "vertices_camera"):
            assert variant[field] is original[field]
        assert isinstance(variant["keypoints_2d"], torch.Tensor if tensor else np.ndarray)
    assert restored["camera_adapter"]["full_image_undistortion"] is True
    assert restored["camera_adapter"]["output_raster"] == "source"
    assert restored["box_provider"]["inference_annotation_coordinates"] == "undistorted_image"
    assert restored["box_provider"]["annotation_coordinates"] == "source_image"
    assert restored["box_provider"]["custom"] == "retained"


def test_restore_drops_unmappable_boxes_and_keeps_empty_frames_and_lens_invalid_mask():
    calibration = camera_for("RATIONAL_POLYNOMIAL").to_dict()
    calibration["distortion_coefficients"] = [-1] + [0] * 13
    camera = camera_from_dict(calibration)
    accepted = prediction_row(camera)
    accepted["bbox"] = np.array([79.5, 95.5, 20], np.float32)
    accepted["joints_camera"][0] = [1, 0, 1]
    skipped = prediction_row(camera)
    skipped["track_id"] = "outside"
    skipped["bbox"] = np.array([500, 95.5, 10], np.float32)
    result = {
        "frames": [
            {"frame_id": 0, "intrinsics": camera.K, "instances": [accepted, skipped]},
            {"frame_id": 1, "intrinsics": camera.K, "instances": []},
        ]
    }
    before = deepcopy(result)
    restored = restore_source_camera_predictions(result, camera)
    assert_same_tree(result, before)
    assert [frame["frame_id"] for frame in restored["frames"]] == [0, 1]
    assert [row["track_id"] for row in restored["frames"][0]["instances"]] == ["right0"]
    assert restored["frames"][1]["instances"] == []
    assert restored["frames"][1]["camera"] == camera.to_dict()
    row = restored["frames"][0]["instances"][0]
    assert not row["projection_valid"][0]
    assert np.isnan(row["keypoints_2d"][0]).all()
    assert np.isnan(row["keypoints_crop"][0]).all()


@pytest.mark.parametrize("focals", [(150, 300), (300, 150)])
@pytest.mark.parametrize("model", ["RATIONAL_POLYNOMIAL", "KB4", "FISHEYE624"])
def test_restore_square_pixel_boxes_preserves_full_model_crop_footprint(model, focals):
    calibration = camera_for(model).to_dict()
    calibration["width"], calibration["height"] = 320, 240
    calibration["intrinsics"] = [[focals[0], 0, 159.5], [0, focals[1], 119.5], [0, 0, 1]]
    camera = camera_from_dict(calibration)
    resize = SquarePixelTransform.from_intrinsics(camera.K.astype(np.float32), 320, 240)
    model_box = np.array(
        [[(resize.width - 1) / 2 + 20, (resize.height - 1) / 2 - 10, 100]], np.float32
    )
    base_box = resize.inverse_boxes(model_box)[0]
    row = prediction_row(camera)
    row["bbox"] = base_box
    result = {
        "camera_adapter": {"square_pixels": True},
        "frames": [
            {
                "frame_id": 0,
                "image_size": [320, 240],
                "intrinsics": camera.K.astype(np.float32),
                "instances": [row],
            },
            {
                "frame_id": 1,
                "image_size": [320, 240],
                "intrinsics": camera.K.astype(np.float32),
                "instances": [],
            },
        ],
    }
    restored = restore_source_camera_predictions(result, camera)
    actual = restored["frames"][0]["instances"][0]
    adapter = FullImageUndistortion(camera)
    expected = adapter.source_boxes(model_box, resize.intrinsics)[0][0]
    wrong_base_square = adapter.source_boxes(base_box[None])[0][0]
    assert expected[2] > 1.5 * wrong_base_square[2]
    np.testing.assert_allclose(actual["bbox"], expected, rtol=0, atol=2e-5)
    assert restored["frames"][1]["instances"] == []
    for variant in (actual, actual["smoothed"]):
        valid = variant["projection_valid"]
        np.testing.assert_allclose(
            variant["keypoints_crop"][valid],
            (variant["keypoints_2d"][valid] - expected[:2]) / expected[2],
            rtol=0,
            atol=1e-6,
        )
    assert actual["joints_camera"] is row["joints_camera"]
    np.testing.assert_array_equal(row["bbox"], base_box)
