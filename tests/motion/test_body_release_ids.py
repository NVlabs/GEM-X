# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body release naming compatibility at training and feature-cache boundaries."""

import json
from types import SimpleNamespace

import pytest
import torch

from gem.motion.data.cache import IMAGE_FEATURE_CACHE_SCHEMA, ImageFeatureCacheDataset
from gem.motion.model import MotionModel
from gem.pose.release_ids import BODY_RELEASE_ID, LEGACY_BODY_RELEASE_ID

PUBLIC_FEATURE = f"{BODY_RELEASE_ID}:pose_tokens"
LEGACY_FEATURE = f"{LEGACY_BODY_RELEASE_ID}:pose_tokens"


@pytest.mark.parametrize(
    "saved,configured", [(LEGACY_FEATURE, PUBLIC_FEATURE), (PUBLIC_FEATURE, LEGACY_FEATURE)]
)
def test_resume_accepts_the_known_body_name_without_rewriting_checkpoint(saved, configured):
    model = SimpleNamespace(
        schema=SimpleNamespace(name="soma77_native_v1"),
        pose_release_id=configured,
        state_dict=lambda: {},
        _fixed_asset_key=MotionModel._fixed_asset_key,
    )
    checkpoint = {
        "gem_motion_training": {
            "pose_release_id": saved,
            "observation_schema": "soma77_native_v1",
            "attempts_completed": 42,
            "branch_counts": {"video": 30, "image": 12},
        },
        "state_dict": {},
    }
    MotionModel.on_load_checkpoint(model, checkpoint)
    assert model.attempts_completed == 42 and model.branch_counts == {"video": 30, "image": 12}
    assert checkpoint["gem_motion_training"]["pose_release_id"] == saved
    checkpoint["gem_motion_training"]["pose_release_id"] = PUBLIC_FEATURE.replace(
        "s240000", "s230000"
    )
    with pytest.raises(ValueError, match="different pose feature release"):
        MotionModel.on_load_checkpoint(model, checkpoint)
    checkpoint["gem_motion_training"]["pose_release_id"] = saved
    checkpoint["gem_motion_training"]["observation_schema"] = "soma29_nofinger_v1"
    with pytest.raises(ValueError, match="different observation schema"):
        MotionModel.on_load_checkpoint(model, checkpoint)


@pytest.mark.parametrize(
    "saved,configured", [(LEGACY_FEATURE, PUBLIC_FEATURE), (PUBLIC_FEATURE, LEGACY_FEATURE)]
)
def test_feature_cache_accepts_known_body_name_and_retains_strict_header(
    tmp_path, saved, configured
):
    record = {
        "schema": IMAGE_FEATURE_CACHE_SCHEMA,
        "feature_version": saved,
        "source": "images",
        "start": 0,
        "stop": 2,
        "columns": {"f_imgseq": torch.arange(2048).reshape(2, 1024).float()},
    }
    part = tmp_path / "features.pt"
    torch.save(record, part)
    manifest = tmp_path / "features.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": IMAGE_FEATURE_CACHE_SCHEMA,
                "complete": True,
                "sources": {"images": {"parts": [{"file": part.name, "start": 0, "stop": 2}]}},
                "source_counts": {"images": 2},
            }
        )
    )
    dataset = ImageFeatureCacheDataset(manifest, "images", configured)
    torch.testing.assert_close(
        dataset[1]["f_imgseq"], record["columns"]["f_imgseq"][1], rtol=0, atol=0
    )
    assert dataset.expected_feature_version == configured
    assert torch.load(part, weights_only=True)["feature_version"] == saved
    # Names only identify this exact body feature release; other releases remain distinct.
    different = ImageFeatureCacheDataset(
        manifest, "images", PUBLIC_FEATURE.replace("s240000", "s230000")
    )
    with pytest.raises(ValueError, match="Feature part header differs"):
        different[0]
    record["source"] = "different-images"
    torch.save(record, part)
    with pytest.raises(ValueError, match="Feature part header differs"):
        ImageFeatureCacheDataset(manifest, "images", configured)[0]
