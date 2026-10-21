# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.inference.cli import main
from gem.inference.inputs import frame_annotations, load_mapping
from gem.inference.pose import run_pose
from gem.inference.serialization import save_predictions

CAMERA = [[800, 0, 20], [0, 790, 10], [0, 0, 1]]


class TinyPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, image, boxes, intrinsics, *, right_hand):
        self.calls.append((image.copy(), boxes, intrinsics, right_hand))
        count = len(boxes)
        index = torch.arange(count, dtype=torch.float32)
        joints = index[:, None, None].expand(-1, 2, 3)
        faces = torch.tensor([[0, 1, 2]]).expand(count, -1, -1).clone()
        if right_hand is not None:
            faces[right_hand] = faces[right_hand].flip(-1)
        return SimpleNamespace(
            intrinsics=intrinsics.expand(count, -1, -1),
            camera_translation=index[:, None].expand(-1, 3),
            joints_camera=joints,
            vertices_camera=joints,
            keypoints_2d=joints[..., :2],
            keypoints_crop=joints[..., :2],
            projection_valid=torch.ones(count, 2, dtype=torch.bool),
            faces=faces,
            parameters={"identity_coeffs": index[:, None]},
            head=SimpleNamespace(token=index[:, None].expand(-1, 4)),
        )


def test_help_never_imports_model_image_or_training_runtime():
    code = """
import builtins
original = builtins.__import__
def lightweight(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'soma', 'cv2', 'lightning', 'hydra'}:
        raise AssertionError('Eager runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = lightweight
from gem.inference.cli import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ("pose-video", "motion-video", "motion-cache"))


def test_exact_frame_ids_and_hand_sides_define_association():
    document = {
        "intrinsics": CAMERA,
        "frames": {
            "2": {
                "instances": [
                    {"track_id": "p2", "side": "right", "bbox": [12, 9, 8]},
                    {"track_id": "p2", "side": "left", "bbox": [9, 10, 8]},
                ]
            },
            "0": {"instances": []},
        },
    }
    frames = frame_annotations(document, task="hand", video=True)
    assert list(frames) == [0, 2]
    assert frames[2]["instances"][0]["side"] == "right"
    document["frames"]["2"]["instances"][1]["side"] = "right"
    with pytest.raises(ValueError, match="Duplicate track/side"):
        frame_annotations(document, task="hand", video=True)
    document["frames"] = {"01": {"instances": []}}
    assert list(frame_annotations(document, task="hand", video=True)) == [1]
    document["frames"]["1"] = {"instances": []}
    with pytest.raises(ValueError, match="Duplicate frame ID"):
        frame_annotations(document, task="hand", video=True)
    document["frames"] = {"1.5": {"instances": []}}
    with pytest.raises(ValueError, match="integer frame IDs"):
        frame_annotations(document, task="hand", video=True)


def test_input_formats_and_camera_precedence(tmp_path):
    document = {
        "intrinsics": CAMERA,
        "frames": {"0": {"instances": [], "intrinsics": np.eye(3).tolist()}},
    }
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(document))
    assert load_mapping(path) == document
    import yaml

    path = tmp_path / "frames.yaml"
    path.write_text(yaml.safe_dump(document))
    assert load_mapping(path) == document
    frames = frame_annotations(document, task="body", video=True)
    assert frames[0]["intrinsics"] == np.eye(3).tolist()
    frames = frame_annotations(document, task="body", video=True, intrinsics=CAMERA)
    assert frames[0]["intrinsics"] == CAMERA
    with pytest.raises(ValueError, match="needs supplied intrinsics"):
        frame_annotations({"instances": []}, task="body", video=False)


def test_video_exact_decoded_frames_rgb_track_order_and_cpu_serialization(tmp_path):
    video = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (40, 24))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((24, 40, 3), (20 + index * 10, 40, 210), np.uint8))
    writer.release()
    document = {
        "intrinsics": CAMERA,
        "frames": {
            "0": {"instances": []},
            "2": {
                "instances": [
                    {"track_id": "second", "side": "right", "bbox": [12, 9, 8]},
                    {"track_id": "first", "side": "left", "bbox": [9, 10, 8]},
                ]
            },
        },
    }
    predictor = TinyPredictor()
    output = run_pose(
        predictor,
        video,
        frame_annotations(document, task="hand", video=True),
        task="hand",
        video=True,
    )
    assert [row["frame_id"] for row in output["frames"]] == [0, 2]
    assert output["frames"][0]["instances"] == []
    assert len(predictor.calls) == 1
    image, _, _, right = predictor.calls[0]
    assert abs(int(image[0, 0, 0]) - 210) <= 3
    assert abs(int(image[0, 0, 2]) - 40) <= 3
    assert right.tolist() == [True, False]
    rows = output["frames"][1]["instances"]
    assert [row["track_id"] for row in rows] == ["second", "first"]
    assert rows[1]["parameters"]["identity_coeffs"].item() == 1
    assert output["topology"]["right"].tolist() == [[2, 1, 0]]
    path = tmp_path / "output" / "predictions.pt"
    save_predictions(output, path)
    stored = torch.load(path, weights_only=False)
    assert stored["frames"][1]["instances"][0]["vertices_camera"].device.type == "cpu"
    assert stored["source_fps"] == 25
    missing = {"6": {"instances": []}}
    with pytest.raises(ValueError, match="annotated frames.*6"):
        run_pose(
            predictor,
            video,
            frame_annotations({"frames": missing, "intrinsics": CAMERA}, task="hand", video=True),
            task="hand",
            video=True,
        )


def test_image_cli_uses_public_predictor_and_writes_named_results(tmp_path, monkeypatch):
    predictor = TinyPredictor()
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor", lambda *args, **kwargs: (predictor, {"task": "body"})
    )
    image = tmp_path / "image.png"
    assert cv2.imwrite(str(image), np.full((24, 40, 3), (10, 50, 200), np.uint8))
    boxes = tmp_path / "boxes.json"
    boxes.write_text(
        json.dumps(
            {
                "intrinsics": CAMERA,
                "instances": [
                    {"track_id": 15, "bbox": [15, 12, 20]},
                ],
            }
        )
    )
    output = tmp_path / "result.pt"
    main(
        [
            "pose-image",
            "--task",
            "body",
            "--input",
            str(image),
            "--boxes",
            str(boxes),
            "--checkpoint",
            "injected.pt",
            "--soma-assets",
            "injected-assets",
            "--output",
            str(output),
        ]
    )
    result = torch.load(output, weights_only=False)
    assert result["mode"] == "image" and result["task"] == "body"
    row = result["frames"][0]["instances"][0]
    assert row["track_id"] == 15 and row["topology"] == "body"
    assert predictor.calls[0][0][0, 0].tolist() == [200, 50, 10]
    assert predictor.calls[0][3] is None
