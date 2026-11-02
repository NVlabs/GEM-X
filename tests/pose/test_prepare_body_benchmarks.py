# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable raw-annotation fixtures; no licensed models or datasets required."""

import pickle
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from gem.pose.data.benchmarks import BodyBenchmarkDataset
from gem.pose.evaluation.prepare_body import main, prepare_3dpw, prepare_emdb


class TinyBody:
    def __init__(self, offset=0.0):
        self.vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3.0]]) + offset
        self.J_regressor = torch.eye(4).repeat(6, 1)
        self.roots = []

    def __call__(self, global_orient, body_pose, betas, transl=None):
        self.roots.append(global_orient.clone())
        vertices = self.vertices.expand(len(betas), -1, -1).clone()
        if transl is not None:
            vertices += transl[:, None]
        return SimpleNamespace(vertices=vertices)


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def test_3dpw_raw_people_camera_mask_box_and_joint_layout(tmp_path):
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    cameras = np.tile(np.eye(4), (3, 1, 1))
    cameras[:, :3, :3] = rotation
    cameras[:, :3, 3] = [10, 20, 30]  # 3DPW root-aligned targets omit translation.
    detections = np.array([[2, 8, 1000], [1, 4, 1000], [1, 0.1, 0]])
    data = {
        "sequence": "clip",
        "genders": ["m", "f"],
        "campose_valid": [[True, False, True], [False, True, False]],
        "poses": np.zeros((2, 3, 72)),
        "poses2d": np.tile(detections, (2, 3, 1, 1)),
        "betas": np.zeros((2, 300)),
        "cam_poses": cameras,
        "cam_intrinsics": np.array([[100, 0, 6], [0, 110, 5], [0, 0, 1]]),
    }
    save_pickle(tmp_path / "sequenceFiles/test/clip.pkl", data)
    # A train file must never enter the test benchmark.
    save_pickle(tmp_path / "sequenceFiles/train/clip.pkl", data)
    models = {"male": TinyBody(), "female": TinyBody(10)}
    regressor = np.eye(4, dtype=np.float32)[np.arange(19) % 4]
    result = prepare_3dpw(tmp_path, models, regressor, batch_size=1)
    assert result["frame_id"].tolist() == [0, 2, 1]
    assert result["person_id"].tolist() == [0, 0, 1]
    assert result["imgname"].tolist() == [
        "imageFiles/clip/image_00000.jpg",
        "imageFiles/clip/image_00002.jpg",
        "imageFiles/clip/image_00001.jpg",
    ]
    np.testing.assert_allclose(result["center"], [[5, 2.5]] * 3)
    # width=6, height=3; enlarge 1.2, then aspect 192:256 => square side 9.6.
    np.testing.assert_allclose(result["box_size"], 9.6)
    assert result["keypoints_3d"].shape == (3, 44, 3)
    np.testing.assert_array_equal(result["keypoints_3d"][:, :25], 0)
    np.testing.assert_allclose(
        result["keypoints_3d"][0, 25:], regressor @ models["male"].vertices.numpy()
    )
    np.testing.assert_allclose(
        result["keypoints_3d"][2, 25:], regressor @ models["female"].vertices.numpy()
    )
    np.testing.assert_allclose(torch.cat(models["male"].roots), [[0, 0, np.pi / 2]] * 2)
    np.testing.assert_array_equal(result["intrinsics"], np.tile(data["cam_intrinsics"], (3, 1, 1)))


def test_emdb_raw_selection_neutral_regressor_gender_camera_and_crop(tmp_path):
    cameras = np.tile(np.eye(4), (3, 1, 1))
    cameras[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cameras[:, :3, 3] = [5, 6, 7]
    trans = np.array([[1, 2, 3], [10, 20, 30], [4, 5, 6.0]])
    data = {
        "name": "P1_01_clip",
        "gender": "female",
        "emdb1": True,
        "good_frames_mask": [True, False, True],
        "smpl": {
            "poses_root": np.zeros((3, 3)),
            "poses_body": np.zeros((3, 69)),
            "betas": np.zeros(10),
            "trans": trans,
        },
        "camera": {"extrinsics": cameras, "intrinsics": np.eye(3)},
        "bboxes": {"bboxes": np.array([[0, 0, 6, 12], [0, 0, 6, 12], [2, 3, 14, 9]])},
    }
    save_pickle(tmp_path / "P1/01_clip/P1_01_clip_data.pkl", data)
    save_pickle(tmp_path / "P1/02_other/P1_02_other_data.pkl", {**data, "emdb1": False})
    # Backup directories are not official participants.
    save_pickle(tmp_path / "P1-bak/01_clip/P1_01_clip_data.pkl", data)
    models = {"neutral": TinyBody(), "female": TinyBody(10)}
    models["female"].J_regressor.zero_()
    result = prepare_emdb(tmp_path, models, batch_size=1)
    assert result["frame_id"].tolist() == [0, 2]
    assert result["imgname"].tolist() == [
        "P1/01_clip/images/00000.jpg",
        "P1/01_clip/images/00002.jpg",
    ]
    np.testing.assert_array_equal(result["center"], [[3, 6], [8, 6]])
    np.testing.assert_allclose(result["box_size"], 16)
    vertices_world = models["female"].vertices.numpy()[None] + trans[[0, 2], None]
    vertices_camera = vertices_world @ cameras[0, :3, :3].T + cameras[0, :3, 3]
    expected = np.einsum("jv,bvc->bjc", models["neutral"].J_regressor, vertices_camera)
    np.testing.assert_allclose(result["keypoints_3d"], expected)


def test_prepared_dataset_rgb_crop_and_stable_person_frame_identity(tmp_path):
    image_path = tmp_path / "imageFiles/clip/image_00000.png"
    image_path.parent.mkdir(parents=True)
    bgr = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    assert cv2.imwrite(str(image_path), bgr)
    data = {
        "protocol": "3dpw",
        "imgname": [str(image_path.relative_to(tmp_path))] * 2,
        "center": [[2, 2]] * 2,
        "box_size": [4, 4],
        "intrinsics": np.tile(np.eye(3), (2, 1, 1)),
        "keypoints_3d": np.zeros((2, 44, 3), np.float32),
        "sequence_id": ["clip"] * 2,
        "person_id": [0, 1],
        "frame_id": [0, 0],
    }
    np.savez(tmp_path / "prepared.npz", **data)
    dataset = BodyBenchmarkDataset(tmp_path / "prepared.npz", tmp_path, image_size=4)
    first, second = dataset[0], dataset[1]
    assert first["sample_id"] == "3dpw:clip:0:0"
    assert second["sample_id"] == "3dpw:clip:1:0"
    expected = bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    for i, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
        expected[i] = (expected[i] - np.float64(255 * mean)) / np.float64(255 * std)
    np.testing.assert_array_equal(first["img"], expected)
    assert first["img_size"].tolist() == [4, 4]
    assert first["bbx_xys"].tolist() == [2, 2, 4]
    # Row reordering must not change the identity of an existing person-frame.
    data["person_id"] = [1, 0]
    np.savez(tmp_path / "reordered.npz", **data)
    reordered = BodyBenchmarkDataset(tmp_path / "reordered.npz", tmp_path)
    assert reordered[1]["sample_id"] == first["sample_id"]


def test_cli_requires_3dpw_regressor_before_loading_smpl(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--dataset",
                "3dpw",
                "--data-root",
                str(tmp_path),
                "--smpl-model-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "prepared.npz"),
            ]
        )
