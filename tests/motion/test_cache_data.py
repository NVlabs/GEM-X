# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completed-cache consumers, paired windows and retained-consumer equivalence."""

import ast
import importlib.util
import json
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from gem.motion.data.cache import (
    IMAGE_FEATURE_CACHE_SCHEMA,
    NATIVE_OBSERVATION_CACHE_SCHEMA,
    ImageFeatureCacheDataset,
    NativeVideoObservations,
    _NativeObservationCache,
)
from gem.motion.frontend.joint_sets import (
    SOMA29_NOFINGER,
    SOMA77_NATIVE,
    get_observation_schema,
    select_soma_observations,
    soma_observation_profile,
)


class OneSample(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_same(actual[name], expected[name])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


def image_cache(directory, source="custom_images", version="custom_feature_v1"):
    parts = []
    for number, (start, stop) in enumerate(((0, 2), (2, 5))):
        count = stop - start
        path = directory / f"part-{number}.pt"
        indices = torch.arange(start, stop)
        columns = {
            "source_index": indices,
            "source_name": [source] * count,
            "sample_id": [f"person-{index}" for index in indices.tolist()],
            "f_imgseq": indices[:, None].float().expand(-1, 1024).clone(),
            "bbx_xys": torch.tensor([50.0, 40.0, 30.0]).expand(count, 3).clone(),
            "K_fullimg": torch.eye(3).expand(count, 3, 3).clone(),
            "keypoints_2d_sapiens": torch.full((count, 308, 3), -0.25),
            "labels": {
                "has_shape": torch.tensor([False] * count),
                "missing_shape": torch.full((count, 128), torch.nan),
            },
            "metadata": [{"tokens": [str(index)]} for index in indices.tolist()],
        }
        torch.save(
            {
                "schema": IMAGE_FEATURE_CACHE_SCHEMA,
                "feature_version": version,
                "source": source,
                "start": start,
                "stop": stop,
                "columns": columns,
            },
            path,
        )
        parts.append({"file": path.name, "start": start, "stop": stop})
    manifest = directory / "images.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": IMAGE_FEATURE_CACHE_SCHEMA,
                "complete": True,
                "sources": {source: {"parts": parts}},
                "source_counts": {source: 5},
            }
        )
    )
    return manifest


def native_cache(directory):
    points = torch.arange(5 * 77 * 3).reshape(5, 77, 3).float()
    points[..., 2] = 0.9
    record = {
        "sequence_key": "sequence-camera",
        "person_id": "person0",
        "frame_ids": torch.tensor([3, 11, 15, 19, 27]),
        "kp2d": points,
        "detector_available": torch.tensor([False, True, False, True, False]),
        "bbx_xys": torch.tensor([50.0, 50.0, 100.0]).expand(5, 3).clone(),
    }
    torch.save(record, directory / "native.pt")
    manifest = directory / "native.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": NATIVE_OBSERVATION_CACHE_SCHEMA,
                "complete": True,
                "video_tracks": {"sequence-camera/person0": {"file": "native.pt"}},
            }
        )
    )
    sample = {
        "meta": {
            "sequence_id": "sequence-camera",
            "person_id": "person0",
            "frame_ids": [11, 15, 19],
            "anchor_frame_id": 15,
            "source_window": (1, 4),
            "data_name": "configured_video",
        },
        "length": 3,
        "kp2d": torch.zeros(5, 30, 3),
        "bbx_xys": torch.tensor([50.0, 50.0, 100.0]).expand(5, 3).clone(),
        "K_fullimg": torch.eye(3).expand(5, 3, 3).clone(),
        "f_imgseq": torch.arange(5 * 1024).reshape(5, 1024).float(),
        "soma_params_c": {"identity_coeffs": torch.arange(5)[:, None].expand(5, 128).float()},
        "motion_targets": {"transl_w": torch.randn(5, 3)},
        "mask": {
            "valid": torch.tensor([True, True, True, False, False]),
            "label_valid": torch.tensor([False, True, False, False, False]),
            "kp2d": torch.ones(5, dtype=torch.bool),
        },
    }
    return manifest, record, sample


def test_images_keep_arbitrary_feature_source_version_missing_labels_and_row_isolation(tmp_path):
    manifest = image_cache(tmp_path)
    reader = ImageFeatureCacheDataset(
        manifest, "custom_images", "custom_feature_v1", max_cached_parts=1
    )
    assert len(reader) == 5
    first, last = reader[0], reader[4]
    assert first["sample_id"] == "person-0" and last["sample_id"] == "person-4"
    assert last["f_imgseq"].shape == (1024,) and last["f_imgseq"].eq(4).all()
    assert last["labels"]["missing_shape"].isnan().all()
    last["f_imgseq"].fill_(100)
    last["labels"]["missing_shape"].fill_(100)
    last["metadata"]["tokens"].append("changed")
    unchanged = reader[4]
    assert unchanged["f_imgseq"].eq(4).all()
    assert unchanged["labels"]["missing_shape"].isnan().all()
    assert unchanged["metadata"]["tokens"] == ["4"]
    assert len(reader._cache) == 1
    with pytest.raises(IndexError):
        reader[-1]
    with pytest.raises(IndexError):
        reader[5]


def test_image_parts_reopen_after_worker_pickle(tmp_path):
    reader = ImageFeatureCacheDataset(image_cache(tmp_path), "custom_images", "custom_feature_v1")
    expected = reader[3]
    restored = pickle.loads(pickle.dumps(reader))
    assert not restored._cache
    assert_same(restored[3], expected)
    assert restored._cache


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_version", "other_release"),
        ("source", "other_source"),
        ("start", 1),
        ("stop", 9),
        ("schema", "other_schema"),
    ],
)
def test_image_part_header_mismatch_is_rejected_on_open(tmp_path, field, value):
    manifest = image_cache(tmp_path)
    part = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part[field] = value
    torch.save(part, tmp_path / "part-0.pt")
    reader = ImageFeatureCacheDataset(manifest, "custom_images", "custom_feature_v1")
    with pytest.raises(ValueError, match="header differs"):
        reader[0]


def test_incomplete_image_or_native_manifest_is_not_a_training_population(tmp_path):
    image = image_cache(tmp_path)
    native, _, sample = native_cache(tmp_path)
    for path in (image, native):
        contents = json.loads(path.read_text())
        contents["complete"] = False
        path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="complete collector"):
        ImageFeatureCacheDataset(image, "custom_images", "custom_feature_v1")
    with pytest.raises(ValueError, match="complete native77"):
        NativeVideoObservations(OneSample(sample), SOMA29_NOFINGER, native, detected_source=True)


def test_eval_fragment_requires_opt_in_and_reads_its_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    _, record, sample = native_cache(artifact_root)
    path = tmp_path / "evaluation-fragment.json"
    payload = {
        "schema": "gem_native77_eval_detection_fragment_v1",
        "fragment_complete": True,
        "artifact_root": str(artifact_root),
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete native77"):
        _NativeObservationCache(path, 2)
    with pytest.raises(ValueError, match="complete native77"):
        NativeVideoObservations(OneSample(sample), SOMA29_NOFINGER, path, detected_source=True)
    cache = _NativeObservationCache(path, 2, allow_evaluation_fragment=True)
    assert_same(cache.load({"file": "native.pt"}), record)
    payload["fragment_complete"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete native77"):
        _NativeObservationCache(path, 2, allow_evaluation_fragment=True)


@pytest.mark.parametrize("schema_name", [SOMA29_NOFINGER, SOMA77_NATIVE])
def test_native_video_join_selects_and_pads_only_observations(tmp_path, schema_name):
    manifest, record, sample = native_cache(tmp_path)
    original = deepcopy(sample)
    schema = get_observation_schema(schema_name)
    reader = NativeVideoObservations(OneSample(sample), schema, manifest, detected_source=True)
    actual = reader[0]
    expected = record["kp2d"][[1, 2, 3, 3, 3]][:, list(schema.active_indices)]
    assert_same(actual["kp2d"], expected)
    assert_same(actual["mask"]["kp2d"], torch.tensor([True, False, True, False, False]))
    for name in original:
        if name not in {"kp2d", "mask"}:
            assert_same(actual[name], original[name])
    assert_same(actual["mask"]["label_valid"], original["mask"]["label_valid"])
    assert_same(sample, original)
    actual["kp2d"].zero_()
    actual["mask"]["kp2d"].zero_()
    assert_same(reader[0]["kp2d"], expected)
    assert_same(sample, original)
    restored = pickle.loads(pickle.dumps(reader))
    assert not restored.native_cache._records
    assert_same(restored[0], reader[0])


def test_non_detected_native_video_does_not_open_cache_or_change_labels(tmp_path):
    _, _, sample = native_cache(tmp_path)
    original = deepcopy(sample)
    reader = NativeVideoObservations(OneSample(sample), SOMA29_NOFINGER, "unused_missing.json")
    assert reader.native_cache is None
    actual = reader[0]
    assert actual["kp2d"].shape == (5, 29, 3) and actual["kp2d"].eq(0).all()
    assert actual["mask"] is sample["mask"]
    assert_same(sample, original)


@pytest.mark.parametrize(
    "change,message",
    [
        ("identity", "identity differs"),
        ("frame_gap", "exact requested frame"),
        ("frame_after", "exact requested frame"),
        ("bbox", "bbox differs"),
        ("nonfinite", "nonfinite"),
    ],
)
def test_native_window_join_rejects_wrong_identity_frame_or_camera_pair(tmp_path, change, message):
    manifest, record, sample = native_cache(tmp_path)
    if change == "identity":
        record["person_id"] = "wrong_person"
    elif change == "frame_gap":
        sample["meta"]["frame_ids"][1] = 14
    elif change == "frame_after":
        sample["meta"]["frame_ids"][2] = 28
    elif change == "bbox":
        record["bbx_xys"][2, 0] += 1
    else:
        record["kp2d"][2, 0, 0] = torch.nan
    torch.save(record, tmp_path / "native.pt")
    reader = NativeVideoObservations(
        OneSample(sample), SOMA29_NOFINGER, manifest, detected_source=True
    )
    with pytest.raises(ValueError, match=message):
        reader[0]


def original_root():
    root = os.environ.get("GEM_NATIVE_REFERENCE_ROOT")
    if root is None:
        pytest.skip("Set GEM_NATIVE_REFERENCE_ROOT to compare completed original consumers")
    return Path(root)


def test_image_rows_match_actual_completed_source_consumer(tmp_path):
    path = original_root() / "gem/datasets/image_feature_dataset.py"
    spec = importlib.util.spec_from_file_location("original_feature_consumer", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    manifest = image_cache(tmp_path, source="cosmos", version=original.FEATURE_VERSION)
    left = original.ImageFeatureDataset(manifest, "cosmos", max_cached_parts=1)
    right = ImageFeatureCacheDataset(
        manifest, "cosmos", original.FEATURE_VERSION, max_cached_parts=1
    )
    for index in (0, 4, 1, 3):
        assert_same(right[index], left[index])


@pytest.mark.parametrize("schema_name", [SOMA29_NOFINGER, SOMA77_NATIVE])
@pytest.mark.parametrize("detected_source", [False, True])
def test_video_rows_match_original_source_wrapper(tmp_path, schema_name, detected_source):
    path = original_root() / "gem/datasets/soma_observations.py"
    # Execute the original consumer functions unchanged, supplying the already
    # parity-tested schema functions without importing its legacy DataModule.
    names = {
        "_NativeObservationCache",
        "_check_boxes",
        "_check_finite_points",
        "_frame_indices",
        "_repeat_last",
        "NativeSomaVideoDataset",
    }
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "torch": torch,
        "json": json,
        "Path": Path,
        "OrderedDict": OrderedDict,
        "Dataset": Dataset,
        "NATIVE_OBSERVATION_CACHE_SCHEMA": NATIVE_OBSERVATION_CACHE_SCHEMA,
        "select_soma_observations": select_soma_observations,
        "soma_observation_profile": soma_observation_profile,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    manifest, _, sample = native_cache(tmp_path)
    left = namespace["NativeSomaVideoDataset"](
        OneSample(deepcopy(sample)), schema_name, manifest, detected_source=detected_source
    )
    right = NativeVideoObservations(
        OneSample(deepcopy(sample)), schema_name, manifest, detected_source=detected_source
    )
    assert_same(right[0], left[0])
