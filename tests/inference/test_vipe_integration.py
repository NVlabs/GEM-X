# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.inference import body_video
from gem.inference.cli import main
from gem.inference.video import VideoFrames


def config(mode="vipe"):
    return {
        "pose": {"checkpoint": "body.ckpt", "release": "body", "batch_size": 2},
        "observations": {"checkpoint": "vitpose.pth", "release_id": "test-native77"},
        "detector": {"checkpoint": "detector.onnx", "device": "cpu"},
        "motion": {
            "checkpoint": "motion.ckpt",
            "condition_spec": {
                "pose_release_id": "gem-pose-body-s240000-v1:pose_tokens",
                "observation_source": "test-native77",
                "observation_joint_set": "soma77_native_v1",
            },
        },
        "camera": {"mode": mode},
    }


@dataclass
class Tracks:
    boxes_xys: np.ndarray
    boxes_xyxy: np.ndarray


def mock_models(monkeypatch, events, encoded_cameras):
    def detector(*args, **kwargs):
        events.append("person_detector")
        return object()

    monkeypatch.setattr(body_video, "YOLOXONNX", detector)
    monkeypatch.setattr(
        body_video,
        "detect_track",
        lambda frames, *_args, **_kw: Tracks(
            np.tile([32.0, 24.0, 40.0], (len(frames), 1)),
            np.tile([12.0, 4.0, 52.0, 44.0], (len(frames), 1)),
        ),
    )
    monkeypatch.setattr(
        body_video,
        "load_native77_detector",
        lambda *_a, **_k: SimpleNamespace(
            release_id="test-native77", predict=lambda frames, boxes: torch.ones(len(frames), 77, 3)
        ),
    )

    class Pose:
        def encode(self, batch):
            encoded_cameras.append(batch["K_fullimg"].clone())
            return SimpleNamespace(
                feature_version="gem-pose-body-s240000-v1:pose_tokens",
                pose_tokens=batch["K_fullimg"][:, 0, 0, None].expand(-1, 1024).clone(),
            )

    monkeypatch.setattr(body_video, "load_pose_checkpoint", lambda *_a, **_k: Pose())
    monkeypatch.setattr(body_video, "load_predictor", lambda *_a, **_k: object())
    monkeypatch.setattr(
        body_video, "run_motion", lambda model, sequence, **kwargs: {"conditions": sequence}
    )


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 48))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 30, np.uint8))
    writer.release()


@pytest.mark.parametrize("override", [False, True])
def test_vipe_uses_normalized_clip_and_paired_intrinsics(tmp_path, monkeypatch, override):
    source = tmp_path / "source.avi"
    make_video(source)
    events, encoded_cameras, camera_inputs = [], [], []
    mock_models(monkeypatch, events, encoded_cameras)

    def vipe(path, *, frame_count, **options):
        events.append("vipe")
        with VideoFrames(path) as frames:
            assert frames.fps == 30 and len(frames) == frame_count == 6
            camera_inputs.append(frames.path)
        K = torch.tensor([[500.0, 0, 32], [0, 510, 24], [0, 0, 1]]).repeat(frame_count, 1, 1)
        K[:, 0, 0] += torch.arange(frame_count)
        return {
            "R_w2c": torch.eye(3).repeat(frame_count, 1, 1),
            "cam_angvel": torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(frame_count, 1),
            "K_fullimg": K,
            "C2W": torch.eye(4).repeat(frame_count, 1, 1),
            "diagnostics": {"method": "vipe", "translation_used": False},
        }

    monkeypatch.setattr(body_video, "run_vipe_camera", vipe)
    cfg = config()
    if override:
        cfg["camera"]["intrinsics"] = [[900.0, 0, 32], [0, 900, 24], [0, 0, 1]]
    result = body_video.run_body_video(source, cfg)
    assert events == ["vipe", "person_detector"]
    expected_fx = torch.full((6,), 900.0) if override else 500.0 + torch.arange(6)
    actual_K = result["conditions"]["K_fullimg"][0]
    torch.testing.assert_close(actual_K, torch.cat(encoded_cameras))
    torch.testing.assert_close(actual_K[:, 0, 0], expected_fx)
    torch.testing.assert_close(result["conditions"]["f_imgseq"][0, :, 0], expected_fx)
    assert result["camera"]["diagnostics"]["intrinsics_source"] == (
        "configuration_override" if override else "vipe"
    )
    assert result["video"]["source"]["fps"] == 25
    assert result["video"]["normalized"]["fps"] == 30
    assert "path" not in result["video"]["normalized"]
    assert camera_inputs[0] != str(source)


def test_static_mode_never_runs_vipe(tmp_path, monkeypatch):
    source = tmp_path / "source.avi"
    make_video(source)
    events, cameras = [], []
    mock_models(monkeypatch, events, cameras)
    monkeypatch.setattr(
        body_video, "run_vipe_camera", lambda *_a, **_k: pytest.fail("ViPE in static mode")
    )
    result = body_video.run_body_video(source, config("static"))
    assert events == ["person_detector"]
    assert result["camera"]["diagnostics"]["method"] == "static"
    torch.testing.assert_close(result["camera"]["R_w2c"], torch.eye(3).repeat(6, 1, 1))


def test_vipe_failure_stops_before_loading_gem_models(tmp_path, monkeypatch):
    source = tmp_path / "source.avi"
    make_video(source)
    events = []
    mock_models(monkeypatch, events, [])

    def fail(*_args, **_kwargs):
        raise RuntimeError("camera inference failed")

    monkeypatch.setattr(body_video, "run_vipe_camera", fail)
    with pytest.raises(RuntimeError, match="camera inference failed"):
        body_video.run_body_video(source, config())
    assert not events


def test_cli_camera_options_override_config(tmp_path, monkeypatch):
    cfg = config("background")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    calls = []
    monkeypatch.setattr(
        body_video, "run_body_video", lambda source, config, **kwargs: calls.append(config) or {}
    )
    main(
        [
            "motion-video",
            "--task",
            "body",
            "--input",
            "input.mp4",
            "--config",
            str(path),
            "--camera-mode",
            "vipe",
            "--vipe-executable",
            "/some env/bin/vipe",
            "--vipe-pipeline",
            "custom-pinhole",
            "--output",
            str(tmp_path / "result.pt"),
        ]
    )
    assert calls[0]["camera"] == {
        "mode": "vipe",
        "vipe": {
            "executable": "/some env/bin/vipe",
            "pipeline": "custom-pinhole",
        },
    }


def test_relative_executable_in_config_resolves_from_config_directory(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"camera": {"vipe": {"executable": "env/bin/vipe"}}}))
    assert body_video.load_body_video_config(path)["camera"]["vipe"]["executable"] == str(
        tmp_path / "env/bin/vipe"
    )
