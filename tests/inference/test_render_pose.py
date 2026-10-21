# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved Pose geometry reaches rendering unchanged, on the source frame grid."""

import builtins
import subprocess
import sys
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from gem.inference import render, render_cli, render_pose
from gem.inference.video import VideoFrames

CAMERA = torch.tensor([[141.0, 3.0, 43.75], [0.0, 179.0, 20.25], [0.0, 0.0, 1.0]])
LEFT = torch.tensor([[-0.2, -0.1, 2.0], [0.1, -0.1, 2.0], [-0.2, 0.2, 2.0]])
RIGHT = torch.tensor([[0.05, -0.1, 3.0], [0.25, -0.1, 3.0], [0.25, 0.15, 3.0], [0.05, 0.15, 3.0]])
MARKER = np.array([15, 220, 35], dtype=np.uint8)


def pose_result(source, *, mode="image", smoothed=False):
    rows = []
    for side, vertices in (("left", LEFT), ("right", RIGHT)):
        row = {
            "track_id": side,
            "side": side,
            "topology": side,
            "vertices_camera": vertices.clone(),
            # Deliberately nonzero: the renderer must not add this again.
            "camera_translation": torch.tensor([4.0, 5.0, 12.0]),
        }
        if smoothed:
            row["smoothed"] = {"vertices_camera": vertices + torch.tensor([0.1, 0.2, 0.3])}
        rows.append(row)
    result = {
        "format": "gem_pose_predictions_v1",
        "task": "hand",
        "mode": mode,
        "source": str(source),
        "source_frame_count": 1,
        "source_fps": None,
        "topology": {
            "left": torch.tensor([[0, 1, 2]]),
            "right": torch.tensor([[0, 2, 1], [0, 3, 2]]),
        },
        "frames": [
            {"frame_id": 0, "image_size": [96, 64], "intrinsics": CAMERA.clone(), "instances": rows}
        ],
    }
    if smoothed:
        result["hand_smoothing"] = {"method": "masked_gaussian_display_geometry"}
    return result


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "hands.png"
    rgb = np.full((64, 96, 3), [210, 60, 35], np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path, rgb


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "hands.avi"
    frames = np.stack(
        [np.full((64, 96, 3), [40 + 20 * index, 80, 200], np.uint8) for index in range(6)]
    )
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 20, (96, 64))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame[..., ::-1])
    writer.release()
    return path, frames


@pytest.fixture
def captured_renderer(monkeypatch):
    instances = []

    class Renderer:
        def __init__(self, width, height, *, backend):
            self.width, self.height, self.backend = width, height, backend
            self.calls = []
            self.closed = False
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def render(self, vertices, faces, intrinsics, extrinsic, *, background):
            self.calls.append(
                {
                    "vertices": vertices.copy(),
                    "faces": faces.copy(),
                    "intrinsics": intrinsics.copy(),
                    "extrinsic": extrinsic.copy(),
                    "background": background.copy(),
                }
            )
            rgb = background.copy()
            rgb[-16:, -24:] = MARKER
            return rgb

    monkeypatch.setattr(render_pose, "MotionMeshRenderer", Renderer)
    return instances


def test_two_hands_share_one_depth_scene_with_saved_winding_and_translation_once(
    image, tmp_path, captured_renderer
):
    result = pose_result(image[0])
    metadata = render_pose.render_pose(result, tmp_path / "image", width=64)
    renderer = captured_renderer[0]
    assert len(captured_renderer) == len(renderer.calls) == 1
    assert (renderer.width, renderer.height) == (64, 43)
    assert renderer.closed
    scene = renderer.calls[0]
    np.testing.assert_array_equal(scene["vertices"], torch.cat((LEFT, RIGHT)).numpy())
    np.testing.assert_array_equal(scene["faces"], [[0, 1, 2], [3, 5, 4], [3, 6, 5]])
    np.testing.assert_array_equal(scene["extrinsic"], np.eye(4))
    # Independently project source points, then apply OpenCV's resize mapping.
    source_projection = scene["vertices"] @ CAMERA.numpy().T
    source_pixels = source_projection[:, :2] / source_projection[:, 2:]
    resized_projection = scene["vertices"] @ scene["intrinsics"].T
    expected_pixels = (source_pixels.astype(np.float64) + 0.5) * [64 / 96, 43 / 64] - 0.5
    np.testing.assert_allclose(
        resized_projection[:, :2] / resized_projection[:, 2:], expected_pixels, atol=2e-6
    )
    assert scene["intrinsics"][0, 1] == 2  # full K retains skew as well as distinct fx/fy
    expected_background = cv2.resize(image[1], (64, 43), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(scene["background"], expected_background)
    saved = cv2.cvtColor(cv2.imread(metadata["images"]["overlay"]), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(saved[:20], expected_background[:20])
    np.testing.assert_array_equal(saved[-1, -1], MARKER)
    assert metadata["variant"] == "raw" and metadata["frame_count"] == 1


@pytest.mark.parametrize("variant", ["raw", "smoothed"])
def test_different_effective_cameras_preserve_each_projection_and_shared_depth_order(
    image, tmp_path, captured_renderer, variant
):
    result = pose_result(image[0], smoothed=True)
    cameras = [
        np.array([[230.0, 7.0, 39.5], [0.0, 175.0, 25.25], [0.0, 0.0, 1.0]]),
        np.array([[93.0, -4.0, 57.75], [0.0, 260.0, 31.25], [0.0, 0.0, 1.0]]),
    ]
    triangle_pixels = np.array([[30.0, 20.0], [65.0, 25.0], [40.0, 45.0]])
    pixels = [triangle_pixels, np.concatenate((triangle_pixels, [[65.0, 45.0]]))]
    depths = [np.array([2.0, 2.5, 3.0]), np.array([4.0, 5.0, 6.0, 5.4])]
    rows = result["frames"][0]["instances"]
    for row, camera, points_2d, depth in zip(rows, cameras, pixels, depths):
        # Construct physical rays analytically. The two triangles coincide in
        # image space, with the right one exactly twice as deep at each vertex.
        y = (points_2d[:, 1] - camera[1, 2]) * depth / camera[1, 1]
        x = ((points_2d[:, 0] - camera[0, 2]) * depth - camera[0, 1] * y) / camera[0, 0]
        points = np.stack((x, y, depth), axis=-1)
        row["intrinsics"] = torch.from_numpy(camera.copy())
        row["vertices_camera"] = torch.from_numpy(points.copy())
        row["smoothed"]["vertices_camera"] = torch.from_numpy(points * 1.1)
    original = deepcopy(result)
    render_pose.render_pose(result, tmp_path / variant, width=64, variant=variant)
    scene = captured_renderer[0].calls[0]
    projected = scene["vertices"] @ scene["intrinsics"].T
    displayed_pixels = projected[:, :2] / projected[:, 2:]
    expected_pixels = (np.concatenate(pixels) + 0.5) * [64 / 96, 43 / 64] - 0.5
    np.testing.assert_allclose(displayed_pixels, expected_pixels, atol=1e-12, rtol=0)
    depth_scale = 1.1 if variant == "smoothed" else 1.0
    np.testing.assert_array_equal(scene["vertices"][:, 2], np.concatenate(depths) * depth_scale)
    np.testing.assert_allclose(displayed_pixels[:3], displayed_pixels[3:6], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(scene["vertices"][3:6, 2], scene["vertices"][:3, 2] * 2)
    assert scene["intrinsics"][0, 0] == pytest.approx(cameras[0][0, 0] * 64 / 96)
    assert scene["intrinsics"][0, 0] != pytest.approx(float(CAMERA[0, 0]) * 64 / 96)
    for actual, before in zip(rows, original["frames"][0]["instances"]):
        for name in ("intrinsics", "vertices_camera", "camera_translation"):
            torch.testing.assert_close(actual[name], before[name], atol=0, rtol=0)
        torch.testing.assert_close(
            actual["smoothed"]["vertices_camera"],
            before["smoothed"]["vertices_camera"],
            atol=0,
            rtol=0,
        )


def test_identical_effective_cameras_preserve_all_physical_vertices_exactly(
    image, tmp_path, captured_renderer
):
    result = pose_result(image[0])
    effective = torch.tensor([[500.0, 0, 48.0], [0, 500.0, 32.0], [0, 0, 1.0]])
    for row in result["frames"][0]["instances"]:
        row["intrinsics"] = effective.clone()
    render_pose.render_pose(result, tmp_path / "shared_camera")
    scene = captured_renderer[0].calls[0]
    np.testing.assert_array_equal(scene["vertices"], torch.cat((LEFT, RIGHT)).numpy())
    np.testing.assert_array_equal(scene["intrinsics"], effective.numpy())


@pytest.mark.parametrize("variant", [None, "raw", "smoothed"])
def test_explicit_smoothing_default_and_raw_override_do_not_mutate_saved_geometry(
    image, tmp_path, captured_renderer, variant
):
    result = pose_result(image[0], smoothed=True)
    before = deepcopy(result)
    metadata = render_pose.render_pose(result, tmp_path / str(variant), variant=variant)
    selected = "raw" if variant == "raw" else "smoothed"
    expected = torch.cat((LEFT, RIGHT))
    if selected == "smoothed":
        expected += torch.tensor([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(captured_renderer[0].calls[0]["vertices"], expected.numpy())
    assert metadata["variant"] == selected
    for actual, original in zip(result["frames"][0]["instances"], before["frames"][0]["instances"]):
        torch.testing.assert_close(
            actual["vertices_camera"], original["vertices_camera"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["smoothed"]["vertices_camera"],
            original["smoothed"]["vertices_camera"],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "variant,message", [("smoothed", "no smoothed"), ("postprocessed", "raw or smoothed")]
)
def test_unavailable_or_motion_variant_fails_before_renderer(
    image, tmp_path, captured_renderer, variant, message
):
    with pytest.raises(ValueError, match=message):
        render_pose.render_pose(pose_result(image[0]), tmp_path / "bad", variant=variant)
    assert captured_renderer == []


def test_sparse_video_retains_native_fps_all_frames_and_plain_missing_backgrounds(
    video, tmp_path, captured_renderer
):
    result = pose_result(video[0], mode="framewise_video")
    result.update(source_frame_count=6, source_fps=20)
    first = result["frames"][0]
    empty = {**first, "frame_id": 2, "instances": []}
    later = {**deepcopy(first), "frame_id": 4}
    result["frames"] = [first, empty, later]
    metadata = render_pose.render_pose(result, tmp_path / "video")
    assert metadata["frame_count"] == 6 and metadata["fps"] == 20
    renderer = captured_renderer[0]
    assert renderer.closed and len(renderer.calls) == 2
    for call, index in zip(renderer.calls, (0, 4)):
        np.testing.assert_array_equal(call["background"], video[1][index])
    # Exercise the actual ffmpeg writer and decode its H.264 output. Sample
    # below the HUD: missing Pose frames must retain their own source image.
    with VideoFrames(metadata["videos"]["overlay"]) as exported:
        assert len(exported) == 6 and exported.fps == 20
        for index in range(6):
            expected = MARKER if index in (0, 4) else video[1][index, 55, 80]
            np.testing.assert_allclose(exported[index][55, 80], expected, atol=5)


def test_no_instances_image_returns_exact_source_without_geometry_call(
    image, tmp_path, captured_renderer
):
    result = pose_result(image[0])
    result["frames"][0]["instances"] = []
    metadata = render_pose.render_pose(result, tmp_path / "empty")
    saved = cv2.cvtColor(cv2.imread(metadata["images"]["overlay"]), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(saved, image[1])
    assert captured_renderer[0].calls == []


def test_replacement_image_requires_same_source_raster(image, tmp_path, captured_renderer):
    different = tmp_path / "different.png"
    assert cv2.imwrite(str(different), np.zeros((64, 98, 3), np.uint8))
    with pytest.raises(ValueError, match="raster dimensions"):
        render_pose.render_pose(pose_result(image[0]), tmp_path / "bad", source=different)
    assert captured_renderer[0].closed and captured_renderer[0].calls == []


@pytest.mark.parametrize("count,fps", [(5, 20), (6, 30)])
def test_replacement_video_requires_same_frame_count_and_fps(
    video, tmp_path, captured_renderer, count, fps
):
    result = pose_result(video[0], mode="framewise_video")
    result.update(source_frame_count=count, source_fps=fps)
    with pytest.raises(ValueError, match="timeline differs"):
        render_pose.render_pose(result, tmp_path / "bad")
    assert captured_renderer == []
    assert not (tmp_path / "bad/overlay.mp4").exists()


def test_legacy_bundle_cannot_silently_drop_saved_frames_beyond_replacement_video(
    video, tmp_path, captured_renderer
):
    result = pose_result("previous-location.avi", mode="framewise_video")
    result.pop("source_frame_count")
    result["source_fps"] = 20
    last = {**deepcopy(result["frames"][0]), "frame_id": 6}
    result["frames"].append(last)
    with pytest.raises(ValueError, match="ends before a saved Pose frame"):
        render_pose.render_pose(result, tmp_path / "shorter", source=video[0])
    assert captured_renderer == []
    assert not (tmp_path / "shorter/overlay.mp4").exists()


def test_render_help_stays_lightweight_and_lists_pose_options():
    code = """
import builtins
original = builtins.__import__
def lightweight(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','cv2','numpy','soma','open3d','transformers','lightning'}:
        raise AssertionError('Heavy help import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = lightweight
from gem.inference.render_cli import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ("hand", "smoothed", "--source", "--source-video"):
        assert option in result.stdout


@pytest.mark.parametrize("task", ["body", "hand"])
def test_pose_cli_renders_saved_geometry_without_assets_or_models(
    image, tmp_path, captured_renderer, monkeypatch, task
):
    result = pose_result("old-location.png")
    result["task"] = task
    if task == "body":
        row = result["frames"][0]["instances"][0]
        row.pop("side")
        row.update(track_id="subject", topology="body")
        result["frames"][0]["instances"] = [row]
        result["topology"] = {"body": result["topology"]["left"]}
    cache = tmp_path / "pose.pt"
    torch.save(result, cache)
    original_import = builtins.__import__

    def no_models(name, *args, **kwargs):
        if name.split(".")[0] in {"soma", "transformers", "onnxruntime", "lightning"} or name in {
            "gem.pose.api",
            "gem.motion.geometry",
        }:
            raise AssertionError(f"Unexpected model/asset import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_models)
    render_cli.main(
        [
            "--task",
            task,
            "--input",
            str(cache),
            "--output-dir",
            str(tmp_path / "cli"),
            "--source",
            str(image[0]),
            "--views",
            "overlay",
        ]
    )
    assert (tmp_path / "cli/overlay.png").is_file()
    assert len(captured_renderer[0].calls) == 1


@pytest.mark.parametrize("extra", [["--task", "body"], ["--task", "hand", "--views", "world"]])
def test_pose_cli_rejects_task_mismatch_and_world_view(image, tmp_path, captured_renderer, extra):
    cache = tmp_path / "pose.pt"
    torch.save(pose_result(image[0]), cache)
    with pytest.raises(SystemExit) as error:
        render_cli.main(["--input", str(cache), "--output-dir", str(tmp_path / "bad"), *extra])
    assert error.value.code == 2
    assert captured_renderer == []


def test_motion_cli_still_requires_body_assets_and_forwards_existing_options(tmp_path, monkeypatch):
    cache = tmp_path / "motion.pt"
    torch.save({"format": "gem_motion_predictions_v1"}, cache)
    calls = []

    def record_motion(result, output_dir, **kwargs):
        calls.append((result, output_dir, kwargs))
        return {"videos": {}}

    monkeypatch.setattr(render, "render_motion", record_motion)
    base = ["--input", str(cache), "--output-dir", str(tmp_path / "motion")]
    for options in (["--task", "body"], ["--task", "hand", "--soma-assets", "assets"]):
        with pytest.raises(SystemExit) as error:
            render_cli.main(base + options)
        assert error.value.code == 2
    assert calls == []
    render_cli.main(
        base
        + [
            "--task",
            "body",
            "--soma-assets",
            "assets",
            "--variant",
            "postprocessed",
            "--views",
            "world",
            "--source-video",
            "aligned.mp4",
            "--width",
            "128",
        ]
    )
    options = calls[0][2]
    assert options["soma_assets"] == "assets" and options["views"] == ["world"]
    assert options["variant"] == "postprocessed" and options["source_video"] == "aligned.mp4"
    assert options["width"] == 128


def fisheye_result(source, *, mode="image"):
    result = pose_result(source, mode=mode, smoothed=True)
    frame = result["frames"][0]
    frame["intrinsics"] = None
    frame["camera"] = {
        "model": "kb4",
        "width": 96,
        "height": 64,
        "intrinsics": [[65.0, 0, 48.0], [0, 65.0, 32.0], [0, 0, 1.0]],
        "radial": [0.1, -0.01, 0.001, 0.0],
    }
    pixels = torch.tensor([[18.25 + 12 * (i % 5), 18.5 + 6 * (i // 5)] for i in range(25)])
    valid = torch.ones(25, dtype=torch.bool)
    valid[7] = False
    pixels[7] = torch.nan
    for row in frame["instances"]:
        row.update(intrinsics=None, keypoints_2d=pixels.clone(), projection_valid=valid.clone())
        row["smoothed"].update(
            keypoints_2d=pixels + torch.tensor([6.0, -3.0]), projection_valid=valid.clone()
        )
    return result


@pytest.mark.parametrize("variant", ["raw", "smoothed"])
def test_fisheye_keypoints_use_source_raster_and_selected_geometry_without_mesh_context(
    image, tmp_path, captured_renderer, monkeypatch, variant
):
    result = fisheye_result(image[0])
    circles, lines = [], []
    original_circle, original_line = cv2.circle, cv2.line

    def circle(rgb, center, *args):
        circles.append(center)
        return original_circle(rgb, center, *args)

    def line(rgb, start, end, *args):
        lines.append((start, end))
        return original_line(rgb, start, end, *args)

    monkeypatch.setattr(cv2, "circle", circle)
    monkeypatch.setattr(cv2, "line", line)
    metadata = render_pose.render_pose(result, tmp_path / variant, width=64, variant=variant)
    assert captured_renderer == []
    assert metadata["overlay_type"] == "source_camera_keypoints"
    assert metadata["variant"] == variant
    assert len(circles) == 48 and len(lines) == 44
    row = result["frames"][0]["instances"][0]
    geometry = row["smoothed"] if variant == "smoothed" else row
    resized = (geometry["keypoints_2d"].numpy() + 0.5) * [64 / 96, 43 / 64] - 0.5
    expected = np.rint(resized[geometry["projection_valid"].numpy()]).astype(int)
    np.testing.assert_array_equal(circles[:24], expected)
    # Each finger starts at wrist 0; invalid joint 7 excludes both touching bones.
    integer_pixels = {
        index: tuple(np.rint(point).astype(int))
        for index, point in enumerate(resized)
        if np.isfinite(point).all()
    }
    assert lines[0] == (integer_pixels[0], integer_pixels[1])
    assert (integer_pixels[0], integer_pixels[5]) in lines
    assert (integer_pixels[0], integer_pixels[10]) in lines
    assert (integer_pixels[0], integer_pixels[15]) in lines
    assert (integer_pixels[0], integer_pixels[20]) in lines
    saved = cv2.cvtColor(cv2.imread(metadata["images"]["overlay"]), cv2.COLOR_BGR2RGB)
    assert saved.shape == (43, 64, 3)
    np.testing.assert_array_equal(
        saved[0], cv2.resize(image[1], (64, 43), interpolation=cv2.INTER_AREA)[0]
    )
    x, y = expected[0]
    np.testing.assert_array_equal(saved[y, x], [80, 175, 255])


def test_fisheye_video_retains_missing_frames_and_labels_keypoint_overlay(
    video, tmp_path, captured_renderer, monkeypatch
):
    result = fisheye_result(video[0], mode="framewise_video")
    result.update(source_frame_count=6, source_fps=20)
    first = result["frames"][0]
    result["frames"] += [
        {**first, "frame_id": 2, "instances": []},
        {**deepcopy(first), "frame_id": 4},
    ]
    titles = []

    def label(rgb, title, *_):
        titles.append(title)
        return rgb

    monkeypatch.setattr(render_pose, "_label", label)
    metadata = render_pose.render_pose(result, tmp_path / "fish-video")
    assert captured_renderer == []
    assert metadata["overlay_type"] == "source_camera_keypoints"
    assert titles == ["Hand Pose - smoothed - source-camera keypoints"] * 6
    with VideoFrames(metadata["videos"]["overlay"]) as exported:
        assert len(exported) == 6 and exported.fps == 20
        for index in (1, 2, 3, 5):
            np.testing.assert_allclose(exported[index][32, 48], video[1][index, 32, 48], atol=5)


def test_fisheye_replacement_image_still_requires_saved_raster_dimensions(
    image, tmp_path, captured_renderer
):
    different = tmp_path / "fish-different.png"
    assert cv2.imwrite(str(different), np.zeros((64, 98, 3), np.uint8))
    with pytest.raises(ValueError, match="raster dimensions"):
        render_pose.render_pose(fisheye_result(image[0]), tmp_path / "bad", source=different)
    assert captured_renderer == []
