# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw benchmark and feature preparation without licensed models or weights."""

import pickle
from dataclasses import asdict
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.motion.api import MotionConditionSpec
from gem.motion.evaluation import prepare
from gem.pose.api import body_config


class TinySMPL:
    def __init__(self, offset=0):
        self.vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3.0]]) + offset
        self.J_regressor = torch.eye(4).repeat(6, 1)
        self.batch_sizes = []

    def __call__(self, global_orient, body_pose, betas, transl):
        self.batch_sizes.append(len(betas))
        return SimpleNamespace(vertices=self.vertices[None] + transl[:, None])


def save_raw(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def emdb_raw(root, name="P1_01_clip"):
    frames = 5
    cameras = np.tile(np.eye(4), (frames, 1, 1))
    cameras[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cameras[:, :3, 3] = [4, 5, 6]
    intrinsics = np.tile(np.eye(3), (frames, 1, 1))
    intrinsics[:, 0, 0] = np.arange(frames) + 100
    data = {
        "name": name,
        "gender": "female",
        "n_frames": frames,
        "emdb1": True,
        "emdb2": True,
        "good_frames_mask": np.array([False, True, False, True, False]),
        "camera": {"intrinsics": intrinsics, "extrinsics": cameras},
        "smpl": {
            "poses_root": np.zeros((frames, 3)),
            "poses_body": np.zeros((frames, 69)),
            "trans": np.arange(frames * 3).reshape(frames, 3),
            "betas": np.zeros(10),
        },
        "bboxes": {
            "bboxes": [[0, 0, 0, 0], [0, 0, 3, 4], [0, 0, 0, 0], [2, 2, 5, 6], [0, 0, 0, 0]],
            "invalid_idxs": [0, 2, 4],
        },
    }
    participant, sequence = name.split("_", 1)
    path = root / participant / sequence / f"{name}_data.pkl"
    save_raw(path, data)
    (path.parent / "images").mkdir()
    for i in range(frames):
        assert cv2.imwrite(
            str(path.parent / "images" / f"{i:05d}.jpg"), np.full((8, 8, 3), i * 20, np.uint8)
        )
    return data


def dpw_raw(root):
    detections = np.array([[1, 4], [2, 6], [1, 1]])
    data = {
        "sequence": "clip",
        "genders": ["m", "f"],
        "poses": np.zeros((2, 5, 72)),
        "trans": np.ones((2, 5, 3)),
        "betas": np.zeros((2, 10)),
        "poses2d": np.tile(detections, (2, 5, 1, 1)),
        "campose_valid": [[True, True, False, True, True], [True, False, True, True, True]],
        "cam_poses": np.tile(np.eye(4), (5, 1, 1)),
        "cam_intrinsics": np.eye(3),
    }
    data["poses2d"][0, 2, 2] = 0
    save_raw(root / "sequenceFiles/test/clip.pkl", data)
    return data


def models():
    result = {"neutral": TinySMPL(), "male": TinySMPL(1), "female": TinySMPL(10)}
    result["female"].J_regressor.zero_()  # GT must use the neutral vertex regressor.
    return result


def test_emdb_retains_invalid_frames_interpolates_boxes_and_regresses_world_mesh(tmp_path):
    data = emdb_raw(tmp_path / "raw")
    bodies = models()
    index = prepare.prepare_targets(
        "emdb2", tmp_path / "raw", tmp_path / "cache", bodies, batch_size=2, include_mesh=True
    )
    entry = index["sequences"][0]
    assert (entry["frame_count"], entry["valid_frame_count"]) == (5, 2)
    payload = torch.load(tmp_path / "cache" / entry["path"], weights_only=True)
    assert payload["frame_ids"].tolist() == [0, 1, 2, 3, 4]
    assert payload["mask"].tolist() == data["good_frames_mask"].tolist()
    assert payload["conditions"]["length"].tolist() == [5]
    np.testing.assert_allclose(
        payload["conditions"]["bbx_xys"][0],
        [[1.5, 2, 4.8], [1.5, 2, 4.8], [2.5, 3, 4.8], [3.5, 4, 4.8], [3.5, 4, 4.8]],
    )
    assert payload["box_observed"].tolist() == [False, True, False, True, False]
    expected_world = bodies["female"].vertices[None] + torch.tensor(data["smpl"]["trans"])[:, None]
    expected_joints = expected_world.repeat(1, 6, 1)
    torch.testing.assert_close(payload["eval_targets"]["global"], expected_joints.float())
    # A world x translation becomes camera y under this known 90-degree rotation.
    assert payload["eval_targets"]["incam"][0, 1].tolist() == [-7, 16, 18]
    torch.testing.assert_close(payload["eval_targets"]["verts_global"], expected_world.float())
    assert bodies["female"].batch_sizes == [2, 2, 1]
    assert payload["protocol"]["camera_source"] == "dataset_intrinsics_and_extrinsics"
    assert not payload["features_ready"]


def test_3dpw_mask_override_subset_and_optional_matched_boxes(tmp_path):
    dpw_raw(tmp_path / "raw")
    masks = {"clip_1": torch.tensor([True, False, False, True, False])}
    boxes = {"clip_1": torch.tensor([[3.0, 4, 8]] * 5)}
    index = prepare.prepare_targets(
        "3dpw",
        tmp_path / "raw",
        tmp_path / "cache",
        models(),
        joint_regressor=torch.eye(4)[torch.arange(14) % 4],
        valid_masks=masks,
        boxes=boxes,
        sequences=["clip_1"],
        batch_size=2,
    )
    assert len(index["sequences"]) == 1
    payload = torch.load(tmp_path / "cache" / index["sequences"][0]["path"], weights_only=True)
    assert payload["eval_targets"]["incam"].shape == (5, 14, 3)
    assert "global" not in payload["eval_targets"]
    torch.testing.assert_close(payload["conditions"]["bbx_xys"][0], boxes["clip_1"])
    assert payload["protocol"]["bbox_source"] == "user_supplied_bbx_xys"
    assert payload["mask_protocol"] == "3dpw_user_supplied_subset"
    for invalid, match in [
        (torch.ones(4, dtype=torch.bool), "bool"),
        (torch.ones(5, dtype=torch.bool), "without valid cameras"),
    ]:
        with pytest.raises(ValueError, match=match):
            list(
                prepare.raw_sequences(
                    "3dpw", tmp_path / "raw", valid_masks={"clip_1": invalid}, sequences=["clip_1"]
                )
            )


def test_raw_3dpw_camera_mask_preserves_full_person_timelines(tmp_path):
    data = dpw_raw(tmp_path)
    records = list(prepare.raw_sequences("3dpw", tmp_path))
    assert [record["sequence_id"] for record in records] == ["clip_0", "clip_1"]
    assert all(len(record["image_paths"]) == 5 for record in records)
    assert records[0]["mask"].tolist() == data["campose_valid"][0]
    assert records[0]["mask_protocol"] == "3dpw_campose_valid"
    assert not records[0]["box_observed"][2]
    assert np.isfinite(records[0]["bbx_xys"]).all()


def test_features_load_models_once_and_preserve_per_frame_intrinsics(tmp_path, monkeypatch):
    emdb_raw(tmp_path / "raw", "P1_01_first")
    emdb_raw(tmp_path / "raw", "P1_02_second")
    prepare.prepare_targets("emdb1", tmp_path / "raw", tmp_path / "cache", models(), batch_size=2)
    spec = MotionConditionSpec(
        f"{body_config().pose_release_id}:pose_tokens", "test-native77", "soma77_native_v1"
    )
    config = {
        "pose": {"release": "body", "checkpoint": "body.ckpt", "batch_size": 2},
        "observations": {
            "release_id": spec.observation_source,
            "checkpoint": "vitpose.pth",
            "batch_size": 2,
        },
        "motion": {"condition_spec": asdict(spec)},
    }
    calls = []

    class Detector:
        def predict(self, frames, boxes):
            assert len(frames) == len(boxes) == 5
            assert frames[0].shape == (8, 8, 3)
            return torch.ones(5, 77, 3)

    class Pose:
        def encode(self, batch):
            assert batch["K_fullimg"].shape[0] == batch["img"].shape[0]
            return SimpleNamespace(
                pose_tokens=batch["K_fullimg"][:, 0, 0, None].expand(-1, 1024),
                feature_version=spec.pose_release_id,
            )

    def detector_loader(*args, **kwargs):
        calls.append("detector")
        return Detector()

    def pose_loader(*args, **kwargs):
        calls.append("pose")
        return Pose()

    monkeypatch.setattr(prepare, "load_native77_detector", detector_loader)
    monkeypatch.setattr(prepare, "load_pose_checkpoint", pose_loader)
    index = prepare.extract_features(tmp_path / "cache", config)
    assert calls == ["detector", "pose"]
    for entry in index["sequences"]:
        payload = torch.load(tmp_path / "cache" / entry["path"], weights_only=True)
        conditions = payload["conditions"]
        assert payload["features_ready"] and entry["features_ready"]
        assert conditions["f_imgseq"].shape == (1, 5, 1024)
        assert conditions["kp2d_native77"].shape == (1, 5, 77, 3)
        torch.testing.assert_close(conditions["f_imgseq"][0, :, 0], torch.arange(100, 105).float())
        assert conditions["condition_spec"] == asdict(spec)
        assert len(payload["mask"]) == 5 and int(payload["mask"].sum()) == 2

    # A failed replacement must not present old Pose tokens with new observations.
    class FailedPose:
        def encode(self, batch):
            raise RuntimeError("feature extraction interrupted")

    monkeypatch.setattr(prepare, "load_pose_checkpoint", lambda *args, **kwargs: FailedPose())
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare.extract_features(tmp_path / "cache", config)
    payload = torch.load(tmp_path / "cache" / index["sequences"][0]["path"], weights_only=True)
    assert not payload["features_ready"] and "f_imgseq" not in payload["conditions"]


def test_cli_rejects_missing_target_inputs_without_loading_smpl(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        prepare.main(["--stage", "targets", "--output-dir", str(tmp_path)])
