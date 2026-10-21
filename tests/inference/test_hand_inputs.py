# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand input providers through real video decode, tracking and Pose CLI I/O."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.common.camera_adapters import SquarePixelTransform
from gem.common.detectors.hand_boxes import BBoxSmoothingConfig, smooth_bbox_track
from gem.inference.cli import build_parser, main, resolve_pose_options
from gem.inference.hand import detect_hand_annotations, saved_hand_annotations
from gem.pose.predictor import PosePrediction

CAMERA = [[123.0, 0, 43.0], [0, 147.0, 27.0], [0, 0, 1.0]]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "hands.avi"
    rgb = np.stack(
        [np.full((64, 96, 3), [20 + 30 * frame, 170, 230], np.uint8) for frame in range(5)]
    )
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25, (96, 64))
    assert writer.isOpened()
    for frame in rgb:
        writer.write(frame[..., ::-1])
    writer.release()
    return path, rgb


def candidate(label, center_x, score=0.9):
    return (label, (center_x - 10, 20, center_x + 10, 40), score)


def install_session(monkeypatch, scheduled):
    """Supply learned detections only; use the actual public ONNX adapter."""
    sessions = []

    class Session:
        def __init__(self, path, *, sess_options, providers):
            self.path, self.providers, self.calls = path, providers, []
            sessions.append(self)

        def get_providers(self):
            return [item[0] if isinstance(item, tuple) else item for item in self.providers]

        def run(self, names, inputs):
            assert names == ["labels", "boxes", "scores"]
            self.calls.append({key: value.copy() for key, value in inputs.items()})
            # Lossless frame colors identify actual decoded input order. The
            # detector's normal RGB/PIL preprocessing is deliberately retained.
            red = np.rint(inputs["images"][:, 0, 100, 100] * 255).astype(int)
            indices = (red - 20) // 30
            width = max(1, max(len(scheduled[int(index)]) for index in indices))
            labels = np.zeros((len(indices), width), np.int64)
            boxes = np.zeros((len(indices), width, 4), np.float32)
            scores = np.zeros((len(indices), width), np.float32)
            for batch_index, frame_index in enumerate(indices):
                for item_index, (label, box, score) in enumerate(scheduled[int(frame_index)]):
                    labels[batch_index, item_index] = label
                    boxes[batch_index, item_index] = box
                    scores[batch_index, item_index] = score
            return labels, boxes, scores

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=Session,
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            get_available_providers=lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
        ),
    )
    return sessions


@pytest.fixture
def detections():
    # Anatomical labels deliberately disagree with screen order at bootstrap;
    # tracking both sides must precede a request to emit only the left side.
    return [
        [candidate(2, 14, 0.9), candidate(1, 80, 0.8), candidate(0, 48, 0.99)],
        [candidate(2, 16, 0.2)],
        [],
        [candidate(1, 26, 0.85), candidate(2, 75, 0.9)],
        [],
    ]


def test_detection_batches_preserve_rgb_tail_frame_grid_and_default_camera(
    video, detections, monkeypatch
):
    path, rgb = video
    sessions = install_session(monkeypatch, detections)
    records, metadata = detect_hand_annotations(path, "hand.onnx", video=True, batch_size=2)
    assert list(records) == list(range(5))
    assert [len(call["images"]) for call in sessions[0].calls] == [2, 2, 1]
    actual_colors = np.concatenate([call["images"][:, :, 100, 100] for call in sessions[0].calls])
    np.testing.assert_array_equal(actual_colors, rgb[:, 0, 0].astype(np.float32) / 255)
    for call in sessions[0].calls:
        np.testing.assert_array_equal(call["orig_target_sizes"], [[96, 64]] * len(call["images"]))
    assert records[0]["intrinsics"] == [[96, 0, 48], [0, 96, 32], [0, 0, 1]]
    assert metadata["camera_source"] == "pose_max_side_estimate"
    assert metadata["image_size"] == [96, 64]
    assert len(metadata["raw_candidates"]) == 5
    assert metadata["raw_candidates"][2] == []
    assert any(item["label_id"] == 0 for item in metadata["raw_candidates"][0])


def test_requested_side_keeps_two_hand_context_rescue_and_gap_flags(video, detections, monkeypatch):
    sessions = install_session(monkeypatch, detections)
    records, metadata = detect_hand_annotations(
        video[0],
        "hand.onnx",
        video=True,
        batch_size=3,
        hands=("left",),
        intrinsics=CAMERA,
        device="cuda:2",
    )
    assert sessions[0].providers[0] == ("CUDAExecutionProvider", {"device_id": 2})
    rows = [frame["instances"][0] for frame in records.values()]
    assert all(len(frame["instances"]) == 1 for frame in records.values())
    assert [row["side"] for row in rows] == ["left"] * 5
    np.testing.assert_array_equal(
        [row["bbox"] for row in rows], [[x, 30, 28] for x in [14, 16, 21, 26, 26]]
    )
    assert [row["bbox_observed"] for row in rows] == [True, True, False, True, False]
    assert metadata["tracking_stats"]["rescued"] == 1
    assert all(frame["intrinsics"] == CAMERA for frame in records.values())
    assert metadata["camera_source"] == "provided"


def test_bbox_smoothing_uses_video_fps_and_complete_clip_before_side_selection(
    video, detections, monkeypatch
):
    install_session(monkeypatch, detections)
    left_only, metadata = detect_hand_annotations(
        video[0], "hand.onnx", video=True, batch_size=1, hands=("left",), smooth_boxes=True
    )
    both, _ = detect_hand_annotations(
        video[0], "hand.onnx", video=True, batch_size=3, smooth_boxes=True
    )
    actual = np.array([record["instances"][0]["bbox"] for record in left_only.values()])
    native = np.array([[x, 30, 28] for x in (14, 16, 21, 26, 26)], dtype=np.float32)
    observed = np.array([True, True, False, True, False])
    expected = smooth_bbox_track(native, np.arange(5) / 25, observed, BBoxSmoothingConfig())
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual, [record["instances"][0]["bbox"] for record in both.values()]
    )
    assert not np.allclose(
        actual, smooth_bbox_track(native, np.arange(5) / 20, observed, BBoxSmoothingConfig())
    )
    assert [record["instances"][0]["bbox_observed"] for record in left_only.values()] == list(
        observed
    )
    assert metadata["tracking"] == "native"
    assert metadata["tracking_stats"]["rescued"] == 1
    assert metadata["bbox_smoothing"]["target_size_scale"] == 0.8


def test_bbox_smoothing_precedes_square_pixel_inverse(video, detections, monkeypatch):
    install_session(monkeypatch, detections)
    records, metadata = detect_hand_annotations(
        video[0],
        "hand.onnx",
        video=True,
        intrinsics=CAMERA,
        square_pixels=True,
        smooth_boxes=True,
        hands=("left",),
    )
    native = np.array([[x, 30, 28] for x in (14, 16, 21, 26, 26)], dtype=np.float32)
    filtered = smooth_bbox_track(
        native,
        np.arange(5) / 25,
        np.array([True, True, False, True, False]),
        BBoxSmoothingConfig(),
    )
    transform = SquarePixelTransform.from_intrinsics(CAMERA, 96, 64)
    np.testing.assert_array_equal(
        [record["instances"][0]["bbox"] for record in records.values()],
        transform.inverse_boxes(filtered),
    )
    assert metadata["annotation_coordinates"] == "source_image"
    assert metadata["detection_image_size"] == [115, 64]
    assert all(record["intrinsics"] == CAMERA for record in records.values())


def test_bbox_smoothing_keeps_entirely_missing_side_absent(video, monkeypatch):
    install_session(monkeypatch, [[], [candidate(2, 70)], [], [], []])
    records, _ = detect_hand_annotations(video[0], "hand.onnx", video=True, smooth_boxes=True)
    assert all(
        [item["side"] for item in record["instances"]] == ["right"] for record in records.values()
    )
    assert [record["instances"][0]["bbox_observed"] for record in records.values()] == [
        False,
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("video_input,tracker", [(False, "native"), (True, "none")])
def test_bbox_smoothing_requires_native_video_before_loading_detector(video_input, tracker):
    with pytest.raises(ValueError, match="requires native video tracking"):
        detect_hand_annotations(
            "unused", "unused.onnx", video=video_input, tracker=tracker, smooth_boxes=True
        )


@pytest.mark.parametrize("tracker", ["native", "none"])
def test_entirely_missing_detections_never_create_pose_instances(video, monkeypatch, tracker):
    install_session(monkeypatch, [[] for _ in range(5)])
    records, _ = detect_hand_annotations(video[0], "hand.onnx", video=True, tracker=tracker)
    assert list(records) == list(range(5))
    assert all(frame["instances"] == [] for frame in records.values())


def test_framewise_mode_uses_anatomical_top1_and_single_padding_without_fill(video, monkeypatch):
    schedule = [
        [candidate(1, 80, 0.8), candidate(1, 14, 0.7), candidate(2, 14, 0.9)],
        [candidate(2, 16, 0.2)],
        [],
        [candidate(1, 26, 0.85)],
        [],
    ]
    install_session(monkeypatch, schedule)
    records, metadata = detect_hand_annotations(video[0], "hand.onnx", video=True, tracker="none")
    first = records[0]["instances"]
    assert [(row["side"], row["bbox"]) for row in first] == [
        ("left", [80, 30, 26]),
        ("right", [14, 30, 26]),
    ]
    assert first[0]["detection_score"] == pytest.approx(0.8)
    assert records[1]["instances"] == records[2]["instances"] == records[4]["instances"] == []
    assert metadata["tracking"] == "framewise"


def test_image_detection_is_framewise_even_with_default_native_option(tmp_path, monkeypatch):
    path = tmp_path / "hand.png"
    assert cv2.imwrite(str(path), np.full((64, 96, 3), [230, 170, 20], np.uint8))
    install_session(monkeypatch, [[candidate(1, 80), candidate(2, 14)]])
    records, metadata = detect_hand_annotations(path, "hand.onnx")
    assert list(records) == [0]
    assert [row["bbox"][0] for row in records[0]["instances"]] == [80, 14]
    assert metadata["tracking"] == "framewise"


def write_saved_boxes(tmp_path, video_path, values, side="right"):
    root = tmp_path / "saved_boxes"
    folder = root / video_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{video_path.stem}_{side}_bbox.json"
    path.write_text(json.dumps(values))
    return root, path


def test_saved_legacy_boxes_keep_sparse_ids_absent_side_and_exact_outside_crops(video, tmp_path):
    supplied = {"000001": [-3.5, 31.25, 18.75], "000004": [99.5, 70, 23]}
    root, file = write_saved_boxes(tmp_path, video[0], supplied)
    records, metadata = saved_hand_annotations(video[0], root, video=True, intrinsics=CAMERA)
    assert list(records) == list(range(5))
    assert all(records[index]["instances"] == [] for index in (0, 2, 3))
    assert records[1]["instances"] == [
        {"side": "right", "track_id": "right", "bbox": supplied["000001"], "bbox_observed": True}
    ]
    assert records[4]["instances"][0]["bbox"] == supplied["000004"]
    assert all(record["intrinsics"] == CAMERA for record in records.values())
    assert metadata["tracking"] == "supplied"
    assert json.loads(file.read_text()) == supplied


@pytest.mark.parametrize(
    "values,message",
    [
        ({"000001": [1, 2, 3], "1": [4, 5, 6]}, "Duplicate saved hand frame"),
        ({"1.5": [1, 2, 3]}, "non-negative integers"),
        ({"000005": [1, 2, 3]}, "outside the input timeline"),
    ],
)
def test_saved_frame_ids_reject_aliases_and_invalid_timeline(video, tmp_path, values, message):
    root, _ = write_saved_boxes(tmp_path, video[0], values)
    with pytest.raises(ValueError, match=message):
        saved_hand_annotations(video[0], root, video=True)


class PoseFixture:
    def __init__(self):
        self.calls = []
        self.geometry = SimpleNamespace(
            _device_anchor=torch.empty(0), forward_physical=self.decode_physical
        )

    @staticmethod
    def decode_physical(parameters, *, right=None, transl=None):
        count = len(parameters["wrist_orient"])
        translation = torch.zeros(count, 3) if transl is None else transl
        joints = translation[:, None] + torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
        return {"joints": joints, "vertices": joints}

    def predict(self, image, boxes, intrinsics, *, right_hand):
        self.calls.append((image.copy(), boxes.clone(), intrinsics.clone(), right_hand.clone()))
        count = len(boxes)
        translation = torch.tensor([0.0, 0.0, 2.0]).expand(count, -1).clone()
        translation[:, 0] = float(image[0, 0, 0]) / 255
        joints = translation[:, None] + torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
        points = joints[..., :2] / joints[..., 2:] * intrinsics[:2, :2].diag() + intrinsics[:2, 2]
        crop = (points - boxes[:, None, :2]) / boxes[:, None, 2:]
        return PosePrediction(
            task="hand",
            pose_release_id=None,
            right_hand=right_hand,
            intrinsics=intrinsics.expand(count, -1, -1),
            camera_translation=translation,
            joints_camera=joints,
            vertices_camera=joints,
            keypoints_2d=points,
            keypoints_crop=crop,
            projection_valid=torch.ones(count, 2, dtype=torch.bool),
            faces=torch.tensor([[0, 1, 1]]).expand(count, -1, -1),
            parameters={
                "wrist_orient": torch.eye(3).expand(count, 1, 3, 3).clone(),
                "hand_pose": torch.eye(3).expand(count, 24, 3, 3).clone(),
                "identity_coeffs": torch.zeros(count, 5),
                "scale_params": torch.cat((torch.ones(count, 1), torch.zeros(count, 26)), dim=1),
            },
            head=SimpleNamespace(token=torch.zeros(count, 4)),
        )


def pose_arguments(path, output, *provider):
    return [
        "pose-video",
        "--task",
        "hand",
        "--input",
        str(path),
        "--checkpoint",
        "fixture.pt",
        "--soma-assets",
        "fixture-assets",
        "--output",
        str(output),
        *provider,
    ]


def install_predictor(monkeypatch):
    predictor = PoseFixture()
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *args, **kwargs: (predictor, {"task": "hand"}),
    )
    return predictor


def test_cli_ego_detection_saves_full_timeline_and_default_smoothing(
    video, detections, tmp_path, monkeypatch
):
    install_session(monkeypatch, detections)
    predictor = install_predictor(monkeypatch)
    output = tmp_path / "prediction.pt"
    main(
        pose_arguments(
            video[0],
            output,
            "--hand-detector",
            "hand.onnx",
            "--hands",
            "left",
            "--detector-batch-size",
            "2",
            "--hand-view",
            "ego",
        )
    )
    stored = torch.load(output, weights_only=False)
    assert stored["source_frame_count"] == 5 and stored["source_fps"] == 25
    assert [frame["frame_id"] for frame in stored["frames"]] == list(range(5))
    rows = [frame["instances"][0] for frame in stored["frames"]]
    assert [row["bbox_observed"] for row in rows] == [True, True, False, True, False]
    assert len(predictor.calls) == 5
    assert all(call[3].tolist() == [False] for call in predictor.calls)
    assert stored["box_provider"]["tracking"] == "native"
    assert stored["hand_view"] == "ego"
    assert stored["hand_smoothing"]["timeline_frame_count"] == 5
    assert "smoothed" in rows[0]
    assert rows[0]["camera_translation"][0] == pytest.approx(20 / 255)
    assert stored["hand_smoothing"]["rotations_filtered"] is True
    assert stored["box_provider"]["bbox_smoothing"]["target_size_scale"] == 0.8
    for row in rows:
        filtered = row["smoothed"]
        assert torch.isfinite(filtered["camera_translation"]).all()
        assert set(filtered["parameters"]) == set(row["parameters"])
        torch.testing.assert_close(filtered["joints_camera"][0], filtered["camera_translation"])


@pytest.mark.parametrize("is_video", [False, True])
def test_cli_framewise_image_and_video_save_scores_and_camera_override(
    video, detections, tmp_path, monkeypatch, is_video
):
    install_session(monkeypatch, detections)
    predictor = install_predictor(monkeypatch)
    source = video[0]
    if not is_video:
        source = tmp_path / "input.png"
        assert cv2.imwrite(str(source), video[1][0][..., ::-1])
    camera_file = tmp_path / "camera.json"
    camera_file.write_text(json.dumps({"intrinsics": CAMERA}))
    output = tmp_path / "framewise.pt"
    arguments = pose_arguments(
        source,
        output,
        "--hand-detector",
        "hand.onnx",
        "--hand-tracker",
        "none",
        "--hands",
        "right",
        "--intrinsics",
        str(camera_file),
    )
    arguments[0] = "pose-video" if is_video else "pose-image"
    main(arguments)
    stored = torch.load(output, weights_only=False)
    first = stored["frames"][0]["instances"][0]
    assert first["side"] == "right" and first["detection_score"] == pytest.approx(0.9)
    assert first["bbox"].tolist() == [14, 30, 26]
    assert first["bbox_observed"] is True
    assert stored["source_frame_count"] == (5 if is_video else 1)
    assert stored["box_provider"]["tracking"] == "framewise"
    assert stored["hand_view"] == "exo"
    assert "hand_smoothing" not in stored
    assert stored["box_provider"]["camera_source"] == "provided"
    assert all(call[2].tolist() == CAMERA for call in predictor.calls)
    assert all(call[3].tolist() == [True] for call in predictor.calls)
    if is_video:
        assert [len(frame["instances"]) for frame in stored["frames"]] == [1, 0, 0, 1, 0]


def test_cli_saved_boxes_skip_detector_and_keep_unobserved_frames_empty(
    video, tmp_path, monkeypatch
):
    root, _ = write_saved_boxes(tmp_path, video[0], {"000002": [-2.5, 30, 17.25]})
    monkeypatch.setattr(
        "gem.inference.hand.RTDETRHandONNX",
        lambda *a, **kw: pytest.fail("Detector must not run for saved boxes"),
    )
    predictor = install_predictor(monkeypatch)
    output = tmp_path / "saved.pt"
    main(pose_arguments(video[0], output, "--boxes", str(root)))
    stored = torch.load(output, weights_only=False)
    assert len(stored["frames"]) == 5
    assert [len(frame["instances"]) for frame in stored["frames"]] == [0, 0, 1, 0, 0]
    assert predictor.calls[0][1].tolist() == [[-2.5, 30, 17.25]]
    assert predictor.calls[0][3].tolist() == [True]
    assert len(predictor.calls) == 1
    assert "hand_smoothing" not in stored


def test_cli_unified_annotations_remain_caller_owned_despite_hand_provider_options(
    video, tmp_path, monkeypatch
):
    document = {
        "intrinsics": CAMERA,
        "frames": {
            "000002": {
                "instances": [
                    {"track_id": "person9", "side": "right", "bbox": [25, 35, 17]},
                    {"track_id": "person9", "side": "left", "bbox": [75, 35, 19]},
                ]
            }
        },
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(
        "gem.inference.hand.RTDETRHandONNX",
        lambda *a, **kw: pytest.fail("Detector must not run for annotations"),
    )
    predictor = install_predictor(monkeypatch)
    output = tmp_path / "external.pt"
    main(
        pose_arguments(
            video[0],
            output,
            "--boxes",
            str(path),
            "--hands",
            "left",
            "--hand-view",
            "ego",
            "--hand-tracker",
            "native",
            "--no-square-pixels",
            "--no-smooth-hand",
        )
    )
    stored = torch.load(output, weights_only=False)
    assert [frame["frame_id"] for frame in stored["frames"]] == [2]
    rows = stored["frames"][0]["instances"]
    assert [(row["track_id"], row["side"]) for row in rows] == [
        ("person9", "right"),
        ("person9", "left"),
    ]
    assert predictor.calls[0][1].tolist() == [[25, 35, 17], [75, 35, 19]]
    assert "box_provider" not in stored
    assert json.loads(path.read_text()) == document


@pytest.mark.parametrize("command", ["pose-image", "pose-video"])
def test_cli_body_cannot_select_hand_detector_before_loading_model(command, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *a, **kw: pytest.fail("No model loading on invalid CLI"),
    )
    arguments = pose_arguments("unused", tmp_path / "unused.pt", "--hand-detector", "hand.onnx")
    arguments[0] = command
    arguments[arguments.index("--task") + 1] = "body"
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_cli_requires_exactly_one_box_provider(tmp_path):
    arguments = pose_arguments("unused", tmp_path / "unused.pt")
    for extra in ([], ["--boxes", "boxes.json", "--hand-detector", "hand.onnx"]):
        with pytest.raises(SystemExit) as error:
            main(arguments + extra)
        assert error.value.code == 2


@pytest.mark.parametrize("tracker", ["native", "none"])
def test_square_pixel_detector_returns_original_camera_and_box_coordinates(
    video, detections, monkeypatch, tracker
):
    sessions = install_session(monkeypatch, detections)
    records, metadata = detect_hand_annotations(
        video[0],
        "hand.onnx",
        video=True,
        intrinsics=CAMERA,
        square_pixels=True,
        tracker=tracker,
    )
    for call in sessions[0].calls:
        np.testing.assert_array_equal(call["orig_target_sizes"], [[115, 64]] * len(call["images"]))
    first = records[0]["instances"][0]
    corrected_center = 14 if tracker == "native" else 80
    corrected_side = 28 if tracker == "native" else 26
    np.testing.assert_allclose(
        first["bbox"],
        [(corrected_center + 0.5) / (115 / 96) - 0.5, 30, corrected_side / (115 / 96)],
    )
    assert all(record["intrinsics"] == CAMERA for record in records.values())
    assert metadata["image_size"] == [96, 64]
    assert metadata["detection_image_size"] == [115, 64]
    assert metadata["annotation_coordinates"] == "source_image"
    assert metadata["raw_candidate_coordinates"] == "square_pixel_image"
    assert metadata["square_pixels"]["active"] is True


@pytest.mark.parametrize(
    "command,task,extra,expected",
    [
        ("pose-video", "hand", [], ("none", False, False)),
        ("pose-video", "hand", ["--hand-view", "ego"], ("native", True, True)),
        ("pose-image", "hand", ["--hand-view", "ego"], ("none", False, True)),
        (
            "pose-video",
            "hand",
            ["--hand-view", "ego", "--no-smooth-hand", "--no-square-pixels"],
            ("native", False, False),
        ),
        ("pose-video", "hand", ["--smooth-hand", "--square-pixels"], ("none", True, True)),
        ("pose-video", "body", ["--square-pixels"], ("none", False, True)),
        (
            "pose-video",
            "hand",
            ["--hand-view", "ego", "--fisheye-camera", "fisheye.json"],
            ("native", True, False),
        ),
    ],
)
def test_hand_view_profiles_resolve_before_runtime_imports(command, task, extra, expected):
    parser = build_parser()
    arguments = pose_arguments("unused.mp4", "unused.pt", "--boxes", "unused.json", *extra)
    arguments[0] = command
    arguments[arguments.index("--task") + 1] = task
    args = parser.parse_args(arguments)
    resolve_pose_options(args, parser)
    assert (args.hand_tracker, args.smooth_hand, args.square_pixels) == expected


@pytest.mark.parametrize(
    "extra",
    [
        ["--hand-tracker", "native"],
        ["--fisheye-camera", "fisheye.json"],
        ["--hand-view", "ego", "--fisheye-camera", "fisheye.json", "--square-pixels"],
        ["--hand-view", "ego", "--fisheye-camera", "fisheye.json", "--intrinsics", "camera.json"],
        ["--camera", "camera.json"],
        ["--hand-view", "ego", "--perspective-crop"],
        ["--hand-view", "ego", "--camera", "camera.json", "--intrinsics", "K.json"],
        ["--hand-view", "ego", "--camera", "camera.json", "--perspective-camera", "K.json"],
        ["--hand-view", "ego", "--perspective-camera", "camera.json", "--no-perspective-crop"],
        ["--hand-view", "ego", "--camera", "camera.json", "--perspective-crop", "--square-pixels"],
    ],
)
def test_invalid_hand_camera_profiles_fail_before_input_or_model_loading(extra, monkeypatch):
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *a, **kw: pytest.fail("Invalid profiles must not load the model"),
    )
    with pytest.raises(SystemExit) as error:
        main(pose_arguments("unused.mp4", "unused.pt", "--boxes", "unused.json", *extra))
    assert error.value.code == 2


@pytest.mark.parametrize("model", ["kb4", "pinhole", "rational_polynomial"])
@pytest.mark.parametrize("shortcut", [False, True])
def test_cli_perspective_camera_owns_intrinsics_and_preserves_sensor_image(
    video, detections, tmp_path, monkeypatch, model, shortcut
):
    sessions = install_session(monkeypatch, detections)
    install_predictor(monkeypatch)
    calibration = {
        "model": "kb4",
        "width": 96,
        "height": 64,
        "intrinsics": CAMERA,
        "radial": [0.01, -0.002, 0.0001, 0.0],
    }
    if model != "kb4":
        calibration["model"] = model
        calibration.pop("radial")
        calibration["distortion_coefficients"] = [] if model == "pinhole" else [0.0] * 14
    camera_file = tmp_path / "camera.json"
    camera_file.write_text(json.dumps(calibration))
    received = {}

    def capture(predictor, path, annotations, **kwargs):
        received.update(kwargs)
        received["annotations"] = annotations
        return {}

    monkeypatch.setattr("gem.inference.pose.run_pose", capture)
    output = tmp_path / "fish.pt"
    main(
        pose_arguments(
            video[0],
            output,
            "--hand-detector",
            "hand.onnx",
            "--hand-view",
            "ego",
            (
                ("--fisheye-camera" if model == "kb4" else "--perspective-camera")
                if shortcut
                else "--camera"
            ),
            str(camera_file),
            *([] if shortcut else ["--perspective-crop"]),
            "--no-smooth-hand",
        )
    )
    assert received["perspective_camera"] == calibration
    assert received["undistort_camera"] is None
    assert received["square_pixels"] is False
    assert all(record["intrinsics"] == CAMERA for record in received["annotations"].values())
    for call in sessions[0].calls:
        np.testing.assert_array_equal(call["orig_target_sizes"], [[96, 64]] * len(call["images"]))
    assert torch.load(output, weights_only=False)["hand_view"] == "ego"


def calibrated_source(model):
    value = {"model": model, "width": 96, "height": 64, "intrinsics": CAMERA}
    if model in ("kb4", "fisheye624"):
        value["radial"] = [0.03, -0.002, 0.0001, 0.0] + (
            [0.0, 0.0] if model == "fisheye624" else []
        )
    else:
        value["distortion_coefficients"] = [] if model == "pinhole" else [0.08, -0.002] + [0.0] * 12
    return value


@pytest.mark.parametrize("model", ["pinhole", "kb4", "fisheye624", "rational_polynomial"])
@pytest.mark.parametrize("square_pixels", [False, True])
def test_cli_camera_default_route_smooths_before_source_restoration(
    video, detections, tmp_path, monkeypatch, model, square_pixels
):
    from gem.common.camera_models import camera_from_dict
    from gem.common.undistortion import FullImageUndistortion, source_boxes
    from gem.pose.postprocess import smooth_hand_predictions

    install_session(monkeypatch, detections)
    predictor = install_predictor(monkeypatch)
    calibration = calibrated_source(model)
    camera_file = tmp_path / "camera.json"
    camera_file.write_text(json.dumps(calibration))
    camera = camera_from_dict(calibration)
    before_smoothing = []

    def smooth(result, *, geometry):
        assert geometry is predictor.geometry
        assert all("camera" not in frame for frame in result["frames"])
        assert result["camera_adapter"]["output_raster"] == (
            "source" if model == "pinhole" else "undistorted"
        )
        before_smoothing.append(result)
        return smooth_hand_predictions(result, geometry=geometry)

    monkeypatch.setattr("gem.pose.postprocess.smooth_hand_predictions", smooth)
    output = tmp_path / "prediction.pt"
    main(
        pose_arguments(
            video[0],
            output,
            "--hand-detector",
            "hand.onnx",
            "--hand-view",
            "ego",
            "--camera",
            str(camera_file),
            *([] if square_pixels else ["--no-square-pixels"]),
        )
    )
    stored = torch.load(output, weights_only=False)
    assert len(before_smoothing) == 1 and len(predictor.calls) == 5
    assert stored["camera_adapter"] == {
        "square_pixels": square_pixels,
        "perspective_crops": False,
        "full_image_undistortion": model != "pinhole",
        "output_raster": "source",
    }
    expected = video[1][0]
    if model != "pinhole":
        expected = FullImageUndistortion(camera).apply(expected)
    if square_pixels:
        expected = SquarePixelTransform.from_intrinsics(camera.K, 96, 64).resize_image(expected)
    np.testing.assert_array_equal(predictor.calls[0][0], expected)
    assert stored["box_provider"]["annotation_coordinates"] == "source_image"
    for frame, raw_frame in zip(stored["frames"], before_smoothing[0]["frames"]):
        if model == "pinhole":
            assert "camera" not in frame and frame["intrinsics"] is not None
        else:
            assert frame["camera"]["model"] == model and frame["intrinsics"] is None
        for row, raw in zip(frame["instances"], raw_frame["instances"]):
            torch.testing.assert_close(row["joints_camera"], raw["joints_camera"], atol=0, rtol=0)
            for name, value in raw["parameters"].items():
                torch.testing.assert_close(row["parameters"][name], value, atol=0, rtol=0)
            if model != "pinhole":
                inference_boxes = [raw["bbox"].numpy()]
                inference_K = np.asarray(raw_frame["intrinsics"])
                if square_pixels:
                    resize = SquarePixelTransform.from_intrinsics(inference_K, 96, 64)
                    inference_boxes = resize.transform_boxes(inference_boxes)
                    inference_K = resize.intrinsics
                expected_boxes, valid = source_boxes(inference_boxes, camera, inference_K)
                assert valid[0]
                np.testing.assert_array_equal(row["bbox"], expected_boxes[0])
                assert row["intrinsics"] is None
            for variant in (row, row["smoothed"]):
                expected_pixels = camera.project(variant["joints_camera"].numpy())
                np.testing.assert_allclose(
                    variant["keypoints_2d"], expected_pixels, atol=1e-5, rtol=1e-6
                )
                np.testing.assert_allclose(
                    variant["keypoints_crop"],
                    (expected_pixels - row["bbox"].numpy()[:2]) / row["bbox"].numpy()[2],
                    atol=1e-5,
                    rtol=1e-6,
                )


@pytest.mark.parametrize("model", ["kb4", "fisheye624", "rational_polynomial"])
def test_distorted_camera_with_saved_source_boxes_requires_perspective_crop(
    tmp_path, monkeypatch, model, capsys
):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(calibrated_source(model)))
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *a, **kw: pytest.fail("Invalid input coordinates must not load the model"),
    )
    with pytest.raises(SystemExit) as error:
        main(
            pose_arguments(
                "unused.mp4",
                "unused.pt",
                "--boxes",
                "unused.json",
                "--hand-view",
                "ego",
                "--camera",
                str(path),
            )
        )
    assert error.value.code == 2
    assert "use --perspective-crop" in capsys.readouterr().err


def test_direct_pinhole_camera_rejects_wrong_raster_before_model_loading(
    video, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({**calibrated_source("pinhole"), "width": 192}))
    monkeypatch.setattr(
        "gem.inference.pose.load_predictor",
        lambda *a, **kw: pytest.fail("Mismatched calibration must not load the model"),
    )
    with pytest.raises(SystemExit) as error:
        main(
            pose_arguments(
                video[0],
                "unused.pt",
                "--boxes",
                "unused.json",
                "--hand-view",
                "ego",
                "--camera",
                str(path),
            )
        )
    assert error.value.code == 2
    assert "dimensions must match" in capsys.readouterr().err
