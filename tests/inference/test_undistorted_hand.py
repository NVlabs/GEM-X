# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-image rectification feeds detector and Pose before source restoration."""

from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.camera_models import camera_from_dict
from gem.common.detectors.hand import HandDetection
from gem.common.undistortion import FullImageUndistortion
from gem.inference import hand, pose
from gem.pose.predictor import PosePrediction


def calibration(model="kb4"):
    value = {
        "model": model,
        "width": 96,
        "height": 64,
        "intrinsics": [[55.0, 0, 47.5], [0, 80.0, 31.5], [0, 0, 1]],
    }
    if model == "kb4":
        value["radial"] = [0.08, -0.005, 0.0001, 0.0]
    else:
        value["distortion_coefficients"] = [0.12, -0.01, 0.002, -0.001] + [0.0] * 10
    return value


def image(index=0):
    yy, xx = np.indices((64, 96))
    return np.stack(((3 * xx + index * 11) % 256, 4 * yy, (xx + 2 * yy) % 256), -1).astype(np.uint8)


def write_media(tmp_path, video):
    images = [image(index) for index in range(4 if video else 1)]
    path = tmp_path / ("input.avi" if video else "input.png")
    if video:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (96, 64))
        assert writer.isOpened()
        for rgb in images:
            writer.write(rgb[..., ::-1])
        writer.release()
    else:
        assert cv2.imwrite(str(path), images[0][..., ::-1])
    return path, images


class RecordingDetector:
    instances = []

    def __init__(self, *args, **kwargs):
        self.images = []
        self.instances.append(self)

    def detect_batch(self, images):
        self.images.extend(image.copy() for image in images)
        return [[HandDetection(1, 0.9, (20.0, 20.0, 40.0, 40.0))] for _ in images]


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("square_pixels", [False, True])
@pytest.mark.parametrize("model", ["kb4", "rational_polynomial"])
def test_detector_receives_full_rectification_before_pixel_aspect_resize(
    tmp_path, monkeypatch, video, square_pixels, model
):
    path, images = write_media(tmp_path, video)
    camera = camera_from_dict(calibration(model))
    adapter = FullImageUndistortion(camera)
    resize = SquarePixelTransform.from_intrinsics(adapter.K, 96, 64)
    monkeypatch.setattr(hand, "RTDETRHandONNX", RecordingDetector)
    annotations, metadata = hand.detect_hand_annotations(
        path,
        "detector.onnx",
        video=video,
        tracker="none",
        batch_size=3,
        square_pixels=square_pixels,
        undistort_camera=calibration(model),
    )
    actual = RecordingDetector.instances[-1].images
    assert len(actual) == len(images)
    for source, detected in zip(images, actual):
        rectified = adapter.apply(source)
        assert not np.array_equal(rectified, source)
        expected = resize.resize_image(rectified) if square_pixels else rectified
        np.testing.assert_array_equal(detected, expected)
    expected_box = resize.inverse_boxes([30.0, 30.0, 20.0]) if square_pixels else [30, 30, 20]
    for record in annotations.values():
        np.testing.assert_array_equal(record["intrinsics"], adapter.K)
        np.testing.assert_array_equal(record["instances"][0]["bbox"], expected_box)
    assert metadata["annotation_coordinates"] == "undistorted_image"
    assert metadata["raw_candidate_coordinates"] == (
        "square_pixel_undistorted_image" if square_pixels else "undistorted_image"
    )
    assert metadata["full_image_undistortion"] is True
    assert metadata["camera_source"] == "provided"


class RecordingPredictor:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def predict(self, image, boxes, intrinsics, *, right_hand):
        self.inputs.append((image.copy(), boxes.clone(), intrinsics.clone(), right_hand.clone()))
        count = len(boxes)
        K = intrinsics.expand(count, -1, -1).clone()
        translation = torch.tensor([0.12, -0.04, 0.9]).expand(count, -1).clone()
        translation[right_hand, 0] *= -1
        joints = (
            translation[:, None]
            + torch.tensor([[0.02, 0.01, 0.0], [-0.03, 0.04, 0.01], [0.01, -0.02, 0.03]])[None]
        )
        projected = joints @ K.transpose(-1, -2)
        pixels = projected[..., :2] / projected[..., 2:]
        parameters = {
            "wrist_orient": torch.eye(3).expand(count, 1, 3, 3).clone(),
            "hand_pose": torch.eye(3).expand(count, 24, 3, 3).clone(),
            "identity_coeffs": torch.zeros(count, 5),
            "scale_params": torch.ones(count, 27),
        }
        result = PosePrediction(
            task="hand",
            parameters=parameters,
            camera_translation=translation,
            joints_camera=joints,
            vertices_camera=joints.clone(),
            keypoints_2d=pixels,
            keypoints_crop=(pixels - boxes[:, None, :2]) / boxes[:, None, 2:],
            projection_valid=torch.ones(count, 3, dtype=torch.bool),
            faces=torch.tensor([[[0, 1, 2]]]).expand(count, -1, -1),
            right_hand=right_hand,
            head=SimpleNamespace(token=torch.zeros(count, 4)),
            pose_release_id=None,
            intrinsics=K,
        )
        self.outputs.append(result)
        return result


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("square_pixels", [False, True])
def test_pose_keeps_undistorted_boxes_geometry_and_parameters_until_final_restoration(
    tmp_path, video, square_pixels
):
    path, images = write_media(tmp_path, video)
    camera = camera_from_dict(calibration())
    adapter = FullImageUndistortion(camera)
    resize = SquarePixelTransform.from_intrinsics(adapter.K, 96, 64)
    ids = [0, 2] if video else [0]
    boxes = [[30.0, 28.0, 20.0], [65.0, 35.0, 18.0]]
    annotations = {
        frame: {
            "intrinsics": np.eye(3).tolist(),
            "instances": [
                {"track_id": "person", "side": side, "bbox": box}
                for side, box in zip(("left", "right"), boxes)
            ],
        }
        for frame in ids
    }
    original = deepcopy(annotations)
    predictor = RecordingPredictor()
    result = pose.run_pose(
        predictor,
        path,
        annotations,
        task="hand",
        video=video,
        square_pixels=square_pixels,
        undistort_camera=calibration(),
    )
    assert annotations == original
    assert [frame["frame_id"] for frame in result["frames"]] == ids
    assert result["source_frame_count"] == len(images)
    assert result["camera_adapter"] == {
        "square_pixels": square_pixels,
        "perspective_crops": False,
        "full_image_undistortion": True,
        "output_raster": "undistorted",
    }
    for frame, network_input, prediction in zip(
        result["frames"], predictor.inputs, predictor.outputs
    ):
        passed_image, passed_boxes, passed_K, passed_sides = network_input
        rectified = adapter.apply(images[frame["frame_id"]])
        np.testing.assert_array_equal(
            passed_image, resize.resize_image(rectified) if square_pixels else rectified
        )
        np.testing.assert_array_equal(
            passed_boxes, resize.transform_boxes(boxes) if square_pixels else boxes
        )
        np.testing.assert_array_equal(passed_K, resize.intrinsics if square_pixels else adapter.K)
        assert passed_sides.tolist() == [False, True]
        assert "camera" not in frame
        np.testing.assert_array_equal(frame["intrinsics"], adapter.K)
        for index, row in enumerate(frame["instances"]):
            np.testing.assert_array_equal(row["bbox"], boxes[index])
            torch.testing.assert_close(row["intrinsics"], torch.as_tensor(adapter.K))
            for name in ("joints_camera", "vertices_camera", "camera_translation"):
                torch.testing.assert_close(
                    row[name], getattr(prediction, name)[index], atol=0, rtol=0
                )
            for name, value in prediction.parameters.items():
                torch.testing.assert_close(row["parameters"][name], value[index], atol=0, rtol=0)
            projected = row["joints_camera"] @ row["intrinsics"].T
            torch.testing.assert_close(
                row["keypoints_2d"], projected[:, :2] / projected[:, 2:], atol=1e-5, rtol=1e-6
            )
            assert "perspective_crop" not in row and "smoothed" not in row


def test_undistortion_calibration_size_and_conflicting_modes_are_rejected(tmp_path, monkeypatch):
    path, _ = write_media(tmp_path, False)
    wrong = {**calibration(), "width": 95}
    monkeypatch.setattr(hand, "RTDETRHandONNX", RecordingDetector)
    with pytest.raises(ValueError, match="(dimensions|size|raster)"):
        hand.detect_hand_annotations(path, "detector.onnx", tracker="none", undistort_camera=wrong)
    annotations = {0: {"intrinsics": np.eye(3).tolist(), "instances": []}}
    with pytest.raises(ValueError, match="(dimensions|size|raster)"):
        pose.run_pose(None, path, annotations, task="hand", undistort_camera=wrong)
    with pytest.raises(ValueError, match="not both"):
        pose.run_pose(
            None,
            path,
            annotations,
            task="hand",
            perspective_camera=calibration(),
            undistort_camera=calibration(),
        )
    with pytest.raises(ValueError, match="hand inference"):
        pose.run_pose(None, path, annotations, task="body", undistort_camera=calibration())
