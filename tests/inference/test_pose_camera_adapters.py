# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks at the camera-adapter/predictor boundary."""

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from gem.common.fisheye import FisheyeCamera, perspective_crop
from gem.inference.camera import predict_perspective, predict_square_pixels
from gem.inference.pose import predict_frame
from gem.pose.data.crop import IMAGENET_MEAN, IMAGENET_STD, prepare_pose_batch
from gem.pose.geometry.hand import HandGeometry
from gem.pose.heads.output import PoseHeadOutput
from gem.pose.predictor import PoseInferenceConfig, PosePrediction


class _WristLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bind_pose_world", torch.eye(4)[None])
        self.register_buffer("t_pose_world", torch.eye(4)[None])


class _KnownGeometryPredictor:
    """Use real RGB/camera preparation, replacing only the learned predictions."""

    def __init__(self, task, *, synthetic_camera=False):
        self.task = task
        self.synthetic_camera = synthetic_camera
        self.config = PoseInferenceConfig(image_size=16)
        self.device = torch.device("cpu")
        self.model = nn.Linear(1, 1)
        self.geometry = HandGeometry(runtime_factory=lambda *_: _WristLayer())

    def predict(self, image, boxes, intrinsics, *, right_hand):
        self.inputs = (image, boxes, intrinsics, right_hand)
        batch = prepare_pose_batch(
            image,
            boxes,
            intrinsics,
            right_hand=right_hand,
            image_size=self.config.image_size,
        )
        return self.predict_batch(batch)

    @torch.autocast(device_type="cpu", enabled=False)
    def predict_batch(self, batch):
        self.batch = batch
        count = len(batch["img"])
        K = batch["original_K_fullimg"].clone()
        if self.synthetic_camera:
            K[:, 0, 0] = K[:, 1, 1] = 17
            K[:, :2, 2] = batch["img_size"] / 2
            right = batch["right_hand"]
            K[right, 0, 2] = batch["img_size"][right, 0] - 1 - K[right, 0, 2]
        translation = torch.tensor([0.1, -0.05, 0.9]).expand(count, -1).clone()
        local_points = torch.tensor([[0, 0, 0], [0.06, -0.03, 0.01], [-0.04, 0.05, 0.03]])
        joints = local_points[None] + translation[:, None]
        # Return already reconstructed physical-side geometry, as PosePredictor
        # does. The adapter must not reflect a right hand a second time.
        joints[batch["right_hand"], :, 0] *= -1
        translation[batch["right_hand"], 0] *= -1
        homogeneous = joints @ K.transpose(-1, -2)
        pixels = homogeneous[..., :2] / homogeneous[..., 2:]
        bbox = batch["original_bbx_xys"]
        parameters = {
            "wrist_orient": torch.eye(3).expand(count, 1, 3, 3).clone(),
            "hand_pose": torch.eye(3).expand(count, 24, 3, 3).clone(),
            "identity_coeffs": torch.zeros(count, 5),
            "scale_params": torch.ones(count, 27),
        }
        head = PoseHeadOutput(parameters, torch.ones(count, 3), {}, torch.zeros(count, 4))
        self.latest = PosePrediction(
            task=self.task,
            parameters=parameters,
            camera_translation=translation,
            joints_camera=joints,
            vertices_camera=joints.clone(),
            keypoints_2d=pixels,
            keypoints_crop=(pixels - bbox[:, None, :2]) / bbox[:, None, 2:],
            projection_valid=torch.ones(count, 3, dtype=torch.bool),
            faces=torch.tensor([[[0, 1, 2]]]).expand(count, -1, -1),
            right_hand=batch["right_hand"] if self.task == "hand" else None,
            head=head,
            pose_release_id=None,
            intrinsics=K,
        )
        return self.latest


@pytest.mark.parametrize("task", ["body", "hand"])
@pytest.mark.parametrize("synthetic_camera", [False, True])
def test_square_pixels_prepare_calibrated_crops_and_restore_source_projection(
    task, synthetic_camera
):
    predictor = _KnownGeometryPredictor(task, synthetic_camera=synthetic_camera)
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    K = torch.tensor([[8.0, 0, 3.5], [0, 12.0, 2.5], [0, 0, 1]])
    boxes = torch.tensor([[2.0, 3, 4], [5.0, 2, 3]])
    right = torch.tensor([False, True]) if task == "hand" else None
    result = predict_square_pixels(predictor, image, boxes, K, right_hand=right)

    passed_image, passed_boxes, passed_K, passed_right = predictor.inputs
    np.testing.assert_array_equal(passed_image, cv2.resize(image, (12, 6)))
    expected_boxes = torch.tensor([[3.25, 3, 6], [7.75, 2, 4.5]])
    torch.testing.assert_close(passed_boxes, expected_boxes)
    expected_K = torch.tensor([[12.0, 0, 5.5], [0, 12.0, 2.5], [0, 0, 1]])
    torch.testing.assert_close(passed_K, expected_K)
    assert passed_right is right
    if task == "hand":
        # Mirror in the resized raster about W-1, retaining anatomical side.
        assert predictor.batch["bbx_xys"][1, 0] == 11 - expected_boxes[1, 0]
        assert predictor.batch["K_fullimg"][1, 0, 2] == 11 - expected_K[0, 2]
        assert result.joints_camera[1, :, 0].lt(0).all()
    expected_source_K = K.expand(2, -1, -1).clone()
    if synthetic_camera:
        expected_source_K[:, 0, 0] = 17 / 1.5
        expected_source_K[:, 1, 1] = 17
        expected_source_K[:, 0, 2] = (6 + 0.5) / 1.5 - 0.5
        expected_source_K[:, 1, 2] = 3
        if task == "hand":
            expected_source_K[1, 0, 2] = (5 + 0.5) / 1.5 - 0.5
    torch.testing.assert_close(result.intrinsics, expected_source_K)
    homogeneous = result.joints_camera @ expected_source_K.transpose(-1, -2)
    expected_pixels = homogeneous[..., :2] / homogeneous[..., 2:]
    torch.testing.assert_close(result.keypoints_2d, expected_pixels)
    torch.testing.assert_close(
        result.keypoints_crop, (expected_pixels - boxes[:, None, :2]) / boxes[:, None, 2:]
    )
    assert result.joints_camera is predictor.latest.joints_camera
    assert result.vertices_camera is predictor.latest.vertices_camera
    assert result.camera_translation is predictor.latest.camera_translation
    assert result.projection_valid is predictor.latest.projection_valid


def test_square_pixel_identity_returns_unmodified_prediction():
    predictor = _KnownGeometryPredictor("hand")
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    K = torch.tensor([[8.0, 0, 3.5], [0, 8.001, 2.5], [0, 0, 1]])
    boxes = torch.tensor([[2.0, 3, 4]])
    right = torch.tensor([True])
    result = predict_square_pixels(predictor, image, boxes, K, right_hand=right)
    assert result is predictor.latest
    assert predictor.inputs[0] is image
    assert predictor.inputs[1] is boxes
    assert predictor.inputs[2] is K


@pytest.mark.parametrize("model", ["KB4", "FISHEYE624"])
def test_fisheye_crops_are_normalized_once_and_outputs_use_source_camera(model):
    predictor = _KnownGeometryPredictor("hand")
    yy, xx = np.indices((120, 160))
    image = np.stack([xx, yy, (xx + 2 * yy) % 256], axis=-1).astype(np.uint8)
    K = np.array([[95, 0, 79.5], [0, 98, 59.5], [0, 0, 1]])
    radial = [0.01, -0.001, 0.0001, 0] + ([0, 0] if model == "FISHEYE624" else [])
    offsets = [0.001, -0.002, 0.0002, 0, 0.0003, 0] if model == "FISHEYE624" else None
    camera = FisheyeCamera(K, radial, 160, 120, offsets, model)
    # The same source ROI makes the canonical-right crop's flip directly visible.
    boxes = torch.tensor([[105.0, 40, 32], [105.0, 40, 32]])
    right = torch.tensor([False, True])
    result, metadata, accepted, skipped = predict_perspective(
        predictor, image, boxes, right_hand=right, camera=camera
    )
    assert accepted == [0, 1] and skipped == {}
    crop, virtual_K, virtual_bbox, rotation = perspective_crop(image, boxes[0], camera, 16)
    normalized = (crop.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    expected_left = torch.from_numpy(normalized.transpose(2, 0, 1))
    torch.testing.assert_close(predictor.batch["img"][0], expected_left, atol=0, rtol=0)
    torch.testing.assert_close(predictor.batch["img"][1], expected_left.flip(-1), atol=0, rtol=0)
    torch.testing.assert_close(
        predictor.batch["original_K_fullimg"], torch.tensor(virtual_K).float().expand(2, -1, -1)
    )
    torch.testing.assert_close(
        predictor.batch["original_bbx_xys"], torch.tensor(virtual_bbox).float().expand(2, -1)
    )
    # Full perspective crops have center (W-1)/2. A legacy W-x mirror is off
    # by one even for these centered cameras and boxes.
    assert predictor.batch["K_fullimg"][1, 0, 2] == 7.5
    assert predictor.batch["bbx_xys"][1, 0] == 7.5
    torch.testing.assert_close(predictor.batch["img_size"], torch.full((2, 2), 16.0))

    R = torch.tensor(rotation).float()
    original = predictor.latest
    expected_joints = torch.einsum("ij,bnj->bni", R, original.joints_camera)
    expected_translation = torch.einsum("ij,bj->bi", R, original.camera_translation)
    torch.testing.assert_close(result.joints_camera, expected_joints)
    torch.testing.assert_close(result.vertices_camera, expected_joints)
    torch.testing.assert_close(result.camera_translation, expected_translation)
    torch.testing.assert_close(result.parameters["wrist_orient"], R.expand(2, 1, -1, -1))
    assert result.parameters["hand_pose"] is original.parameters["hand_pose"]
    assert result.head is original.head
    torch.testing.assert_close(
        original.head.parameters["wrist_orient"], torch.eye(3).expand(2, 1, 3, 3)
    )
    expected_pixels = torch.tensor(camera.project(expected_joints.numpy())).float()
    torch.testing.assert_close(result.keypoints_2d, expected_pixels)
    torch.testing.assert_close(
        result.keypoints_crop, (expected_pixels - boxes[:, None, :2]) / boxes[:, None, 2:]
    )
    assert result.projection_valid.all()
    assert result.intrinsics is None
    for row in metadata:
        torch.testing.assert_close(row["intrinsics"], torch.tensor(virtual_K).float())
        torch.testing.assert_close(row["bbox"], torch.tensor(virtual_bbox).float())
        torch.testing.assert_close(row["rotation_to_source"], R)


@pytest.mark.parametrize(
    "field,value", [("hand_mirror", "legacy_crop"), ("camera_mode", "perspective")]
)
def test_fisheye_rejects_uncalibrated_release_semantics(field, value):
    predictor = _KnownGeometryPredictor("hand")
    setattr(predictor.config, field, value)
    with pytest.raises(ValueError, match="calibrated full_image"):
        predict_perspective(predictor, None, None, right_hand=None, camera=None)


def test_frame_fisheye_metadata_keeps_original_raster_and_per_side_topology():
    predictor = _KnownGeometryPredictor("hand")
    camera = FisheyeCamera([[95, 0, 79.5], [0, 98, 59.5], [0, 0, 1]], [0, 0, 0, 0], 160, 120)
    record = {
        "intrinsics": camera.K.tolist(),
        "instances": [{"track_id": "right", "side": "right", "bbox": [105, 40, 32]}],
    }
    topology = {}
    result = predict_frame(
        predictor,
        np.zeros((120, 160, 3), dtype=np.uint8),
        3,
        record,
        task="hand",
        topology=topology,
        camera=camera,
    )
    assert result["frame_id"] == 3
    assert result["image_size"] == [160, 120]
    assert result["intrinsics"] is None
    assert result["camera"] == camera.to_dict()
    row = result["instances"][0]
    assert row["intrinsics"] is None and row["side"] == "right"
    assert row["track_id"] == "right" and row["topology"] == "right"
    torch.testing.assert_close(row["bbox"], torch.tensor([105.0, 40, 32]))
    assert row["perspective_crop"]["rotation_to_source"].shape == (3, 3)
    assert set(topology) == {"right"}


@pytest.mark.parametrize("adapter", ["square_pixels", "fisheye"])
def test_caller_autocast_preserves_fp32_camera_and_geometry(adapter):
    predictor = _KnownGeometryPredictor("hand")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    K = torch.tensor([[95.0, 0, 79.5], [0, 128.0, 59.5], [0, 0, 1]])
    boxes = torch.tensor([[105.0, 40, 32], [65.0, 85, 27]])
    right = torch.tensor([False, True])
    camera = FisheyeCamera(K.numpy(), [0.01, -0.001, 0.0001, 0], 160, 120)

    def call():
        if adapter == "square_pixels":
            return predict_square_pixels(predictor, image, boxes, K, right_hand=right)
        return predict_perspective(predictor, image, boxes, right_hand=right, camera=camera)[0]

    expected = call()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = call()
    for name in ("joints_camera", "vertices_camera", "camera_translation", "keypoints_2d"):
        value = getattr(actual, name)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, getattr(expected, name), atol=0, rtol=0)
    torch.testing.assert_close(
        actual.parameters["wrist_orient"], expected.parameters["wrist_orient"], atol=0, rtol=0
    )
    if adapter == "square_pixels":
        assert actual.intrinsics.dtype == torch.float32
        torch.testing.assert_close(actual.intrinsics, expected.intrinsics, atol=0, rtol=0)


@pytest.mark.parametrize("include_valid_right", [True, False])
def test_fisheye_invalid_roi_is_skipped_without_changing_valid_hand_identity(include_valid_right):
    predictor = _KnownGeometryPredictor("hand")
    camera = FisheyeCamera([[180, 0, 319.5], [0, 180, 239.5], [0, 0, 1]], [0, 0, 0, 0], 640, 480)
    # This corner lies outside the forward lens domain, while the center ROI
    # remains valid. The detector can produce both in the same frame.
    invalid = {"track_id": "left", "side": "left", "bbox": [30, 30, 40]}
    valid = {"track_id": "right", "side": "right", "bbox": [320, 240, 80]}
    instances = [invalid, valid] if include_valid_right else [invalid]
    topology = {}
    result = predict_frame(
        predictor,
        np.zeros((480, 640, 3), dtype=np.uint8),
        4,
        {"intrinsics": camera.K.tolist(), "instances": instances},
        task="hand",
        topology=topology,
        camera=camera,
    )
    assert len(result["skipped_instances"]) == 1
    skipped = result["skipped_instances"][0]
    assert skipped["track_id"] == "left" and skipped["side"] == "left"
    assert skipped["bbox"] == [30, 30, 40]
    assert "undistort" in skipped["reason"].lower()
    assert result["camera"] == camera.to_dict()
    if include_valid_right:
        assert len(result["instances"]) == 1
        row = result["instances"][0]
        assert row["track_id"] == "right" and row["side"] == "right"
        torch.testing.assert_close(row["bbox"], torch.tensor([320.0, 240, 80]))
        assert predictor.batch["right_hand"].tolist() == [True]
        assert len(predictor.batch["img"]) == 1
        assert set(topology) == {"right"}
    else:
        assert result["instances"] == []
        assert topology == {}
        assert not hasattr(predictor, "batch")
