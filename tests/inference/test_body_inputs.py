# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic Body inputs through real decoding, shared tracking and Pose I/O."""

import json

import cv2
import numpy as np
import pytest
import torch

from gem.inference.body import detect_body_annotations
from gem.inference.cli import main
from gem.inference.pose import run_pose
from tests.inference.test_entrypoints import TinyPredictor

CAMERA = [[123.0, 0, 43.0], [0, 147.0, 27.0], [0, 0, 1.0]]
SMALL_BOX = [2, 6, 12, 26]
LARGE_BOX = [30, 5, 70, 55]


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "body.png"
    rgb = np.full((64, 96, 3), [20, 170, 230], np.uint8)
    assert cv2.imwrite(str(path), rgb[..., ::-1])
    return path, rgb


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "bodies.avi"
    rgb = np.stack(
        [np.full((64, 96, 3), [20 + 20 * frame, 170, 230], np.uint8) for frame in range(9)]
    )
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (96, 64))
    assert writer.isOpened()
    for frame in rgb:
        writer.write(frame[..., ::-1])
    writer.release()
    return path, rgb


def install_detector(monkeypatch, scheduled):
    """Replace learned detections, retaining real ByteTrack and crop policies."""
    created = []

    class Detector:
        def __init__(self, checkpoint, *, device):
            self.checkpoint = str(checkpoint)
            self.device = device
            self.frames = []
            created.append(self)

        def detect(self, frame):
            boxes = scheduled[len(self.frames)]
            self.frames.append(frame.copy())
            return np.asarray(boxes, np.float32).reshape(-1, 4), np.full(len(boxes), 0.9)

    monkeypatch.setattr("gem.inference.body.YOLOXONNX", Detector)
    return created


def cli_arguments(command, task, source, output):
    return [
        command,
        "--task",
        task,
        "--input",
        str(source),
        "--checkpoint",
        "injected.ckpt",
        "--soma-assets",
        "injected-assets",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("detector_path", [None, "custom-person.onnx"])
def test_image_cli_automatically_detects_primary_person(
    image, tmp_path, monkeypatch, detector_path
):
    path, rgb = image
    detectors = install_detector(monkeypatch, [[SMALL_BOX, LARGE_BOX]])
    predictor = TinyPredictor()
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor", lambda *a, **kw: (predictor, {"task": "body"})
    )
    output = tmp_path / "image.pt"
    arguments = cli_arguments("pose-image", "body", path, output)
    camera = [[96, 0, 48], [0, 96, 32], [0, 0, 1]]
    if detector_path is not None:
        intrinsics = tmp_path / "camera.json"
        intrinsics.write_text(json.dumps({"intrinsics": CAMERA}))
        arguments += [
            "--body-detector",
            detector_path,
            "--detector-device",
            "cuda:2",
            "--intrinsics",
            str(intrinsics),
        ]
        camera = CAMERA
    main(arguments)
    result = torch.load(output, weights_only=False)
    detector = detectors[0]
    assert detector.checkpoint == (detector_path or "weights/yolox_humanart.onnx")
    assert detector.device == ("cpu" if detector_path is None else "cuda:2")
    np.testing.assert_array_equal(detector.frames[0], rgb[..., ::-1])
    np.testing.assert_array_equal(predictor.calls[0][0], rgb)
    np.testing.assert_allclose(predictor.calls[0][2], camera)
    assert predictor.calls[0][3] is None
    assert result["mode"] == "image" and result["source_frame_count"] == 1
    rows = result["frames"][0]["instances"]
    assert len(rows) == 1 and rows[0]["track_id"] == 1
    np.testing.assert_allclose(rows[0]["bbox"], [50, 30, 64], atol=1e-5)
    assert rows[0]["bbox_observed"] is True
    provider = result["box_provider"]
    assert provider["provider"] == "yolox" and provider["tracking"] == "primary"
    assert provider["annotation_coordinates"] == "source_image"
    assert provider["camera_source"] == (
        "pose_max_side_estimate" if detector_path is None else "provided"
    )


def test_video_keeps_source_timeline_and_primary_track_through_missing_detections(
    video, monkeypatch
):
    path, rgb = video
    observed = [False, True, True, False, True, False, False, True, False]
    scheduled = [[SMALL_BOX, LARGE_BOX] if present else [SMALL_BOX] for present in observed]
    detectors = install_detector(monkeypatch, scheduled)
    annotations, metadata = detect_body_annotations(path, "person.onnx", video=True)
    assert list(annotations) == list(range(9))
    rows = [record["instances"][0] for record in annotations.values()]
    assert [row["track_id"] for row in rows] == [1] * 9
    assert [row["bbox_observed"] for row in rows] == observed
    np.testing.assert_allclose([row["bbox"] for row in rows], [[50, 30, 64]] * 9, atol=1e-5)
    np.testing.assert_array_equal(np.stack(detectors[0].frames), rgb[..., ::-1])
    assert metadata["tracking"] == "primary"
    predictor = TinyPredictor()
    result = run_pose(predictor, path, annotations, task="body", video=True)
    assert result["source_fps"] == 25 and result["source_frame_count"] == 9
    assert [record["frame_id"] for record in result["frames"]] == list(range(9))
    np.testing.assert_array_equal(np.stack([call[0] for call in predictor.calls]), rgb)
    assert [record["instances"][0]["bbox_observed"] for record in result["frames"]] == observed


def test_supplied_video_boxes_bypass_detector_and_keep_ids_gaps_and_camera(
    video, tmp_path, monkeypatch
):
    path, _ = video
    monkeypatch.setattr(
        "gem.inference.body.YOLOXONNX",
        lambda *a, **kw: pytest.fail("Supplied boxes bypass detection"),
    )
    predictor = TinyPredictor()
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor", lambda *a, **kw: (predictor, {"task": "body"})
    )
    supplied = [
        {"track_id": "second", "bbox": [50, 30, 61]},
        {"track_id": 17, "bbox": [7, 16, 23]},
    ]
    boxes = tmp_path / "boxes.json"
    boxes.write_text(
        json.dumps(
            {
                "intrinsics": CAMERA,
                "frames": {"0": {"instances": []}, "2": {"instances": supplied}},
            }
        )
    )
    output = tmp_path / "supplied.pt"
    main(cli_arguments("pose-video", "body", path, output) + ["--boxes", str(boxes)])
    result = torch.load(output, weights_only=False)
    assert [record["frame_id"] for record in result["frames"]] == [0, 2]
    assert result["frames"][0]["instances"] == []
    assert "box_provider" not in result
    assert result["source_fps"] == 25
    rows = result["frames"][1]["instances"]
    assert [row["track_id"] for row in rows] == ["second", 17]
    np.testing.assert_array_equal(
        torch.stack([row["bbox"] for row in rows]), [[50, 30, 61], [7, 16, 23]]
    )
    assert len(predictor.calls) == 1
    np.testing.assert_array_equal(predictor.calls[0][2], CAMERA)


def test_no_person_uses_existing_tracking_error(image, monkeypatch):
    install_detector(monkeypatch, [[]])
    with pytest.raises(ValueError, match="No person track"):
        detect_body_annotations(image[0], "person.onnx")


@pytest.mark.parametrize(
    "task,flags,message",
    [
        ("body", ["--hand-detector", "hands.onnx"], "--hand-detector"),
        ("hand", ["--body-detector", "person.onnx"], "--body-detector"),
        ("hand", [], "--boxes"),
    ],
)
def test_detector_task_mismatches_and_missing_hand_input_fail_before_model_loading(
    task, flags, message, monkeypatch, capsys
):
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *a, **kw: pytest.fail("Invalid CLI loads no model"),
    )
    with pytest.raises(SystemExit) as error:
        main(cli_arguments("pose-image", task, "missing.png", "unused.pt") + flags)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
