# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumers for completed image-feature columns and native observation sidecars.

Collectors own population coverage and payload validation. These readers check
the loaded part's contract and join each requested video window to its paired
detector rows. Source selection, sampling, labels and feature versions belong
to the caller, without dataset-specific imports or a second data campaign.
"""

from __future__ import annotations

import json
import operator
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gem.motion.frontend.joint_sets import (
    SOMA_OBSERVATION_SCHEMAS,
    get_observation_schema,
    select_observation_joints,
)
from gem.pose.release_ids import canonical_pose_release_id

IMAGE_FEATURE_CACHE_SCHEMA = "gem_t1_image_feature_cache_v1"
NATIVE_OBSERVATION_CACHE_SCHEMA = "gem_native77_observation_cache_v1"


def _copy_row(columns, index):
    if isinstance(columns, torch.Tensor):
        return columns[index].clone()
    if isinstance(columns, Mapping):
        return {name: _copy_row(value, index) for name, value in columns.items()}
    return deepcopy(columns[index])


class ImageFeatureCacheDataset(Dataset):
    """Read one caller-selected feature source by its original logical ordinal.

    The completed manifest provides ordered contiguous part intervals. Opening
    a part checks its header once per LRU miss; the collector's full-coverage
    and numerical validation is reused. Returned rows retain all labels and
    metadata without interpreting missing-value or coordinate semantics.
    """

    def __init__(self, manifest_path, source, expected_feature_version, *, max_cached_parts=2):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if (
            self.manifest.get("schema") != IMAGE_FEATURE_CACHE_SCHEMA
            or self.manifest.get("complete") is not True
        ):
            raise ValueError("Image features require a complete collector manifest")
        if source not in self.manifest["sources"]:
            raise ValueError(f"Image source is absent from this cache: {source!r}")
        if max_cached_parts < 1:
            raise ValueError("max_cached_parts must be positive")
        self.source = source
        self.expected_feature_version = expected_feature_version
        self.max_cached_parts = max_cached_parts
        self.parts = self.manifest["sources"][source]["parts"]
        self._length = int(self.manifest["source_counts"][source])
        self._starts = tuple(part["start"] for part in self.parts)
        self._cache = OrderedDict()

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _load_part(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        part = self.parts[index]
        path = Path(part["file"])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected = (
            IMAGE_FEATURE_CACHE_SCHEMA,
            canonical_pose_release_id(self.expected_feature_version),
            self.source,
            part["start"],
            part["stop"],
        )
        actual = (
            record["schema"],
            canonical_pose_release_id(record["feature_version"]),
            record["source"],
            record["start"],
            record["stop"],
        )
        if actual != expected:
            raise ValueError(f"Feature part header differs from its completed manifest: {path}")
        self._cache[index] = record["columns"]
        if len(self._cache) > self.max_cached_parts:
            self._cache.popitem(last=False)
        return record["columns"]

    def __getitem__(self, index):
        index = operator.index(index)
        if not 0 <= index < self._length:
            raise IndexError(f"{self.source} source index out of range: {index}")
        part_index = bisect_right(self._starts, index) - 1
        return _copy_row(self._load_part(part_index), index - self._starts[part_index])


class _NativeObservationCache:
    def __init__(self, manifest_path, max_cached_parts, *, allow_evaluation_fragment=False):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        full_cache = (
            self.manifest.get("schema") == NATIVE_OBSERVATION_CACHE_SCHEMA
            and self.manifest.get("complete") is True
        )
        evaluation_fragment = (
            allow_evaluation_fragment
            and self.manifest.get("schema") == "gem_native77_eval_detection_fragment_v1"
            and self.manifest.get("fragment_complete") is True
        )
        if not (full_cache or evaluation_fragment):
            raise ValueError("Native observations require a complete native77 cache manifest")
        self.artifact_root = (
            Path(self.manifest["artifact_root"]) if evaluation_fragment else self.path.parent
        )
        if max_cached_parts < 1:
            raise ValueError("max_cached_parts must be positive")
        self.max_cached_parts = max_cached_parts
        self._records = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_records"] = OrderedDict()
        return state

    def load(self, entry):
        path = Path(entry["file"])
        if not path.is_absolute():
            path = self.artifact_root / path
        if path in self._records:
            self._records.move_to_end(path)
            return self._records[path]
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        points = record["kp2d"]
        count = len(points)
        if points.dtype != torch.float32 or tuple(points.shape) != (count, 77, 3):
            raise ValueError(f"Native point cache must contain float32[N,77,3]: {path}")
        available = record["detector_available"]
        if available.dtype != torch.bool or tuple(available.shape) != (count,):
            raise ValueError(f"Native detector availability must be bool[N]: {path}")
        if tuple(record["bbx_xys"].shape) != (count, 3):
            raise ValueError(f"Native detector boxes must be [N,3]: {path}")
        frames = record["frame_ids"]
        if frames.dtype != torch.long or frames.shape != (count,):
            raise ValueError(f"Native frame_ids must be int64[N]: {path}")
        if count == 0 or not bool((frames[1:] > frames[:-1]).all()):
            raise ValueError(f"Native frame_ids must be ordered and unique: {path}")
        self._records[path] = record
        if len(self._records) > self.max_cached_parts:
            self._records.popitem(last=False)
        return record


def _frame_indices(record, frame_ids, identity):
    cached = record["frame_ids"]
    requested = torch.as_tensor(frame_ids, dtype=torch.long)
    indices = torch.searchsorted(cached, requested)
    if bool((indices >= len(cached)).any()) or not torch.equal(cached[indices], requested):
        raise ValueError(f"Native detector cache lacks an exact requested frame: {identity}")
    return indices


def _repeat_last(value, length):
    if len(value) == length:
        return value.clone()
    return torch.cat((value, value[-1:].expand(length - len(value), *value.shape[1:])), dim=0)


class NativeVideoObservations(Dataset):
    """Join native77 detector rows to an existing caller-owned video sampler.

    Only ``kp2d`` and detector-availability ``mask.kp2d`` are replaced. Reliable
    labels, camera/frame grids, feature IDs and sampling remain owned by the
    base dataset. Non-detected sources receive zero observation placeholders;
    TrainingPreparation constructs their native GT observations after FK.
    """

    def __init__(
        self, base_dataset, schema, manifest_path, *, detected_source=False, max_cached_parts=2
    ):
        self.base_dataset = base_dataset
        self.schema = get_observation_schema(schema)
        if self.schema.name not in SOMA_OBSERVATION_SCHEMAS:
            raise ValueError("Native video observations require SOMA29 or SOMA77")
        self.detected_source = detected_source
        self.native_cache = (
            _NativeObservationCache(manifest_path, max_cached_parts) if detected_source else None
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        original = self.base_dataset[index]
        sample = dict(original)
        if not self.detected_source:
            sample["kp2d"] = original["kp2d"].new_zeros(
                (*original["kp2d"].shape[:-2], self.schema.num_joints, 3)
            )
            return sample
        meta = original["meta"]
        key = f"{meta['sequence_id']}/{meta['person_id']}"
        try:
            entry = self.native_cache.manifest["video_tracks"][key]
        except KeyError as error:
            raise ValueError(f"Native detector cache is missing video track {key}") from error
        record = self.native_cache.load(entry)
        if str(record["sequence_key"]) != str(meta["sequence_id"]) or str(
            record["person_id"]
        ) != str(meta["person_id"]):
            raise ValueError(f"Native detector track identity differs: {key}")
        length = int(original["length"])
        if len(meta["frame_ids"]) != length:
            raise ValueError(f"Video metadata frame_ids differs from the unpadded length: {key}")
        indices = _frame_indices(record, meta["frame_ids"], key)
        if not torch.equal(record["bbx_xys"][indices], original["bbx_xys"][:length]):
            raise ValueError(f"Native detector bbox differs from the paired feature bbox: {key}")
        points = record["kp2d"][indices]
        if not torch.isfinite(points).all():
            raise ValueError(f"Native detector points are nonfinite: {key}")
        selected = select_observation_joints(points, self.schema)
        padded_length = len(original["kp2d"])
        sample["kp2d"] = _repeat_last(selected, padded_length)
        available = _repeat_last(record["detector_available"][indices], padded_length)
        sample["mask"] = {**original["mask"], "kp2d": available & original["mask"]["valid"]}
        return sample
