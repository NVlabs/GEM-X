# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive filtering preserves physical parameters, FK and camera projection."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from gem.common.fisheye import FisheyeCamera
from gem.pose.geometry.hand import HandGeometry
from gem.pose.postprocess import _project, smooth_hand_predictions
from gem.pose.smoothing import (
    AdaptiveSmoothingConfig,
    smooth_rotations,
    smooth_translation,
    validate_frame_times,
)


class SyntheticHand(nn.Module):
    """Small side-dependent FK substitute with every saved parameter active."""

    def __init__(self, side):
        super().__init__()
        self.side = side
        self.register_buffer("faces", torch.tensor([[0, 1, 2]]))
        self.calls = 0

    def forward(
        self,
        poses,
        identity,
        *,
        pose2rot,
        global_translation,
        global_scale,
        scale_params,
        apply_correctives,
    ):
        assert not pose2rot and not apply_correctives and global_translation is None
        assert poses.dtype == identity.dtype == torch.float32
        self.calls += len(poses)
        index = torch.arange(25, device=poses.device, dtype=poses.dtype)
        template = torch.stack((index * 0.002 + 0.01, index * -0.001, index * 0.0005), -1)
        if self.side == "right":
            template[:, 0] *= -1
        joints = (poses @ template[None, :, :, None])[..., 0]
        joints = joints * global_scale[:, None, None]
        joints = joints + 0.002 * identity.mean(-1)[:, None, None]
        joints = joints + 0.001 * scale_params.mean(-1)[:, None, None]
        return {"joints": joints, "vertices": joints[:, :3] * 1.2}


def geometry():
    return HandGeometry(runtime_factory=lambda side, device: SyntheticHand(side).to(device))


def make_result(count=30, fps=30.0, sides=("left", "right")):
    frames = []
    rng = np.random.default_rng(17)
    for index in range(count):
        rows = []
        for side in sides:
            angles = np.full((25, 3), 0.01 * np.sin(index * 0.2))
            angles += rng.normal(0, 0.008, angles.shape)
            poses = torch.tensor(Rotation.from_rotvec(angles).as_matrix(), dtype=torch.float32)
            rows.append(
                {
                    "track_id": "wearer",
                    "side": side,
                    "topology": side,
                    "bbox": torch.tensor([80.0, 50.0, 90.0]),
                    "parameters": {
                        "wrist_orient": poses[:1],
                        "hand_pose": poses[1:],
                        "identity_coeffs": torch.full((5,), 0.1 + (index % 2) * 0.2),
                        "scale_params": torch.full((27,), 1.0 + (index % 2) * 0.02),
                    },
                    "pose_token": torch.tensor([float(index), 4.0]),
                    "camera_translation": torch.tensor([0.001 * (-1) ** index, 0.05, 0.8]),
                    # Intentionally unrelated cached geometry must be replaced by FK.
                    "joints_camera": torch.full((25, 3), 7.0),
                    "vertices_camera": torch.full((3, 3), 8.0),
                    "keypoints_2d": torch.zeros(25, 2),
                    "keypoints_crop": torch.zeros(25, 2),
                    "projection_valid": torch.ones(25, dtype=torch.bool),
                }
            )
        frames.append(
            {
                "frame_id": index,
                "intrinsics": torch.tensor([[410.0, 0, 80.0], [0, 390.0, 50.0], [0, 0, 1.0]]),
                "instances": rows,
            }
        )
    return {
        "task": "hand",
        "mode": "framewise_video",
        "source_fps": fps,
        "source_frame_count": count,
        "frames": frames,
        "release": {"inference": {"minimum_depth": 1e-4}},
    }


def test_filtered_parameters_decode_to_saved_geometry_and_raw_fields_are_preserved(monkeypatch):
    rig = geometry()
    monkeypatch.setattr(rig, "_right_parameters", lambda *args: pytest.fail("second right mirror"))
    result = make_result()
    original = deepcopy(result)
    actual = smooth_hand_predictions(result, geometry=rig)
    for frame, raw_frame in zip(actual["frames"], original["frames"]):
        for row, raw in zip(frame["instances"], raw_frame["instances"]):
            for name, value in raw.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(row[name], value, atol=0, rtol=0)
            for name, value in raw["parameters"].items():
                torch.testing.assert_close(row["parameters"][name], value, atol=0, rtol=0)
            filtered = row["smoothed"]
            decoded = rig.forward_physical(
                {name: value[None] for name, value in filtered["parameters"].items()},
                right=torch.tensor([row["side"] == "right"]),
                transl=filtered["camera_translation"][None],
            )
            torch.testing.assert_close(filtered["joints_camera"], decoded["joints"][0])
            torch.testing.assert_close(filtered["vertices_camera"], decoded["vertices"][0])
            points = filtered["joints_camera"].double() @ frame["intrinsics"].double().T
            pixels = points[:, :2] / points[:, 2:]
            torch.testing.assert_close(
                filtered["keypoints_2d"].double(), pixels, atol=1e-5, rtol=1e-6
            )
            torch.testing.assert_close(
                filtered["keypoints_crop"] * row["bbox"][2] + row["bbox"][:2],
                filtered["keypoints_2d"],
                atol=1e-5,
                rtol=1e-6,
            )
            assert set(filtered["parameters"]) == set(row["parameters"])
            assert filtered["shape_parameters"]["scale_params"].shape == (27,)
    assert all("smoothed" not in row for frame in result["frames"] for row in frame["instances"])
    assert actual["hand_smoothing"]["rotations_filtered"] is True
    assert actual["hand_smoothing"]["fk_parameters"] == "smoothed.parameters"
    assert actual["hand_smoothing"]["time_unit"] == "seconds"


def test_missing_frames_invalid_parameters_and_tracks_are_independent():
    result = make_result(9, sides=("left",))
    # Remove one entire decoded frame, leave another present but without a hand.
    del result["frames"][4]
    result["frames"][1]["instances"] = []
    result["frames"][5]["instances"][0]["parameters"]["hand_pose"][0, 0, 0] = float("nan")
    for frame in result["frames"]:
        for row in frame["instances"]:
            value = float(frame["frame_id"])
            row["parameters"]["identity_coeffs"].fill_(value)
            row["camera_translation"] = torch.tensor([value, 0, 1.0])
    other = deepcopy(result["frames"][0]["instances"][0])
    other["track_id"] = "other"
    other["parameters"]["identity_coeffs"].fill_(100)
    result["frames"][0]["instances"].append(other)
    rig = geometry()
    actual = smooth_hand_predictions(result, geometry=rig)
    assert [frame["frame_id"] for frame in actual["frames"]] == [0, 1, 2, 3, 5, 6, 7, 8]
    assert actual["frames"][1]["instances"] == []
    # Each valid segment's shape stays within its own endpoints, never a gap neighbor's.
    segments = {0: (0, 0), 2: (2, 3), 3: (2, 3), 5: (5, 5), 7: (7, 8), 8: (7, 8)}
    for frame in actual["frames"]:
        for row in frame["instances"]:
            filtered = row["smoothed"]
            if row["track_id"] == "other":
                assert filtered["parameters"]["identity_coeffs"].eq(100).all()
            elif frame["frame_id"] == 6:
                torch.testing.assert_close(
                    filtered["parameters"]["hand_pose"],
                    row["parameters"]["hand_pose"],
                    equal_nan=True,
                )
                torch.testing.assert_close(
                    filtered["joints_camera"], row["joints_camera"], atol=0, rtol=0
                )
            else:
                low, high = segments[frame["frame_id"]]
                shape = filtered["parameters"]["identity_coeffs"]
                assert ((shape >= low) & (shape <= high)).all()
    assert rig._layers["left"].module.calls == 7


def test_timestamp_gap_splits_even_adjacent_decoded_frames():
    result = make_result(3, fps=5, sides=("left",))
    actual = smooth_hand_predictions(result, geometry=geometry())
    for frame in actual["frames"]:
        row = frame["instances"][0]
        torch.testing.assert_close(
            row["smoothed"]["camera_translation"], row["camera_translation"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            row["smoothed"]["parameters"]["identity_coeffs"],
            row["parameters"]["identity_coeffs"],
            atol=0,
            rtol=0,
        )


def test_reprocessing_starts_from_raw_predictions():
    rig = geometry()
    first = smooth_hand_predictions(make_result(), geometry=rig)
    second = smooth_hand_predictions(first, geometry=rig)
    for a, b in zip(first["frames"], second["frames"]):
        for ar, br in zip(a["instances"], b["instances"]):
            for name in ("joints_camera", "camera_translation", "keypoints_2d"):
                torch.testing.assert_close(
                    ar["smoothed"][name], br["smoothed"][name], atol=0, rtol=0
                )


@pytest.mark.parametrize("count", [1, 7, 35])
def test_rotations_remain_proper_through_wraparound_and_inexact_inputs(count):
    angles = np.linspace(179.0, 181.0, count)
    rotations = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()[:, None]
    rotations[:, :, 0, 0] *= 1.0002
    filtered = smooth_rotations(rotations, np.arange(count) / 30, AdaptiveSmoothingConfig())
    np.testing.assert_allclose(
        filtered.swapaxes(-1, -2) @ filtered, np.broadcast_to(np.eye(3), filtered.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(filtered), 1.0, atol=1e-12)
    assert np.all(Rotation.from_matrix(filtered[:, 0]).magnitude() > 3.0)


def test_long_fit_reduces_stationary_noise_and_preserves_quadratic_motion():
    times = np.arange(80) / 60
    config = AdaptiveSmoothingConfig()
    rng = np.random.default_rng(4)
    noisy = rng.normal(0, 0.001, (len(times), 3))
    filtered = smooth_translation(noisy, times, config)
    assert np.std(filtered[10:-10]) < np.std(noisy[10:-10]) * 0.7
    trajectory = np.stack((0.1 * times**2, 0.2 * times, 1 + 0.02 * times**2), -1)
    np.testing.assert_allclose(
        smooth_translation(trajectory, times, config), trajectory, atol=1e-10
    )
    np.testing.assert_array_equal(filtered[[0, -1]], noisy[[0, -1]])


def test_video_rate_controls_physical_filter_width():
    result = make_result()
    faster = {**result, "source_fps": 60.0}
    low_rate = smooth_hand_predictions(result, geometry=geometry())
    high_rate = smooth_hand_predictions(faster, geometry=geometry())
    low_shape = low_rate["frames"][0]["instances"][0]["smoothed"]["parameters"]["identity_coeffs"]
    high_shape = high_rate["frames"][0]["instances"][0]["smoothed"]["parameters"]["identity_coeffs"]
    # The same physical window includes more of the alternating samples at 60 Hz.
    assert (high_shape > low_shape + 0.001).all()


@pytest.mark.parametrize("model", ["kb4", "fisheye624"])
def test_fisheye_and_effective_pinhole_projection(model):
    result = make_result(4)
    camera = {
        "model": model,
        "width": 160,
        "height": 100,
        "intrinsics": [[110.0, 0.0, 78.5], [0.0, 105.0, 48.75], [0.0, 0.0, 1.0]],
        "radial": [0.02, -0.001, 0.0001, 0.0] + ([0.0, 0.0] if model == "fisheye624" else []),
        "tangential_prism": [0.0] * 6,
    }
    lens = FisheyeCamera.from_dict(camera)
    for frame in result["frames"]:
        frame.update(camera=camera, intrinsics=None)
        for row in frame["instances"]:
            row["intrinsics"] = None
    actual = smooth_hand_predictions(result, geometry=geometry())
    for frame in actual["frames"]:
        for row in frame["instances"]:
            filtered = row["smoothed"]
            np.testing.assert_allclose(
                filtered["keypoints_2d"], lens.project(filtered["joints_camera"].numpy()), atol=1e-5
            )
    pinhole = make_result(4)
    effective = torch.tensor([[900.0, 3, 78.5], [0, 880, 45.5], [0, 0, 1.0]])
    for frame in pinhole["frames"]:
        frame["instances"][1]["intrinsics"] = effective
    actual = smooth_hand_predictions(pinhole, geometry=geometry())
    row = actual["frames"][0]["instances"][1]
    points = row["smoothed"]["joints_camera"].numpy().astype(np.float64)
    pixel = points @ effective.numpy().T
    np.testing.assert_allclose(
        row["smoothed"]["keypoints_2d"], pixel[:, :2] / pixel[:, 2:], atol=1e-5
    )


def test_invalid_projection_stays_nan_with_explicit_mask():
    points = np.array([[0, 0, 1], [1, 2, -1], [np.nan, 0, 1], [0, 0, 1e-4]])
    _, pixels, crop, valid = _project(points, np.zeros(3), np.eye(3), [0, 0, 1], 1e-4)
    np.testing.assert_array_equal(valid, [True, False, False, False])
    assert np.isnan(pixels[1:]).all() and np.isnan(crop[1:]).all()


def test_input_boundaries_and_empty_video():
    result = make_result(0)
    assert smooth_hand_predictions(result, geometry=geometry())["frames"] == []
    for change in ({"task": "body"}, {"mode": "image"}):
        with pytest.raises(ValueError, match="framewise hand video"):
            smooth_hand_predictions({**result, **change}, geometry=geometry())
    for fps in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="source_fps"):
            smooth_hand_predictions({**result, "source_fps": fps}, geometry=geometry())
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_frame_times([0, 0], 2)
    with pytest.raises(ValueError, match="timestamp"):
        validate_frame_times([0], 2)
    with pytest.raises(ValueError, match="positive"):
        AdaptiveSmoothingConfig(max_gap_seconds=0)


def test_physical_decode_replays_mixed_side_canonical_output_without_second_mirror(monkeypatch):
    rig = geometry()
    examples = make_result(3, sides=("left",))
    parameters = {
        name: torch.stack(
            [frame["instances"][0]["parameters"][name] for frame in examples["frames"]]
        )
        for name in examples["frames"][0]["instances"][0]["parameters"]
    }
    side = torch.tensor([True, False, True])
    translation = torch.tensor([[0.1, -0.2, 1.0], [-0.3, 0.1, 0.8], [0.2, 0.3, 1.2]])
    calls = []

    def mirror(poses, identity, global_scale, local_scale, transl):
        calls.append(len(poses))
        rotation = poses.new_tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        return rotation @ poses, transl * transl.new_tensor([-1, 1, 1])

    monkeypatch.setattr(rig, "_right_parameters", mirror)
    canonical = rig(parameters, right=side, transl=translation)
    assert calls == [2]
    monkeypatch.setattr(rig, "_right_parameters", lambda *args: pytest.fail("second mirror"))
    physical = rig.forward_physical(canonical["parameters"], right=side, transl=canonical["transl"])
    for name in ("joints", "vertices", "faces", "transl"):
        torch.testing.assert_close(physical[name], canonical[name], atol=0, rtol=0)


def test_fk_batches_preserve_complete_clip_filter_context(monkeypatch):
    from gem.pose import postprocess

    result = make_result(67, sides=("left",))
    expected = smooth_hand_predictions(result, geometry=geometry())
    monkeypatch.setattr(postprocess, "FK_BATCH_SIZE", 5)
    actual = smooth_hand_predictions(result, geometry=geometry())
    for a, b in zip(actual["frames"], expected["frames"]):
        actual_row, expected_row = a["instances"][0]["smoothed"], b["instances"][0]["smoothed"]
        for name in ("joints_camera", "vertices_camera", "camera_translation"):
            torch.testing.assert_close(actual_row[name], expected_row[name], atol=0, rtol=0)
