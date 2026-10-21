# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pytest

from gem.inference.body_video import body_release
from gem.inference.cli import build_parser
from gem.inference.pose import load_predictor
from gem.pose import api
from gem.pose.release_ids import (
    BODY_RELEASE_ID,
    LEGACY_BODY_RELEASE_ID,
    canonical_pose_release_id,
)


def test_public_body_preset_preserves_legacy_architecture():
    public = asdict(api.body_config())
    legacy = asdict(api.body_sg6_config())
    assert public.pop("pose_release_id") == BODY_RELEASE_ID
    assert legacy.pop("pose_release_id") == LEGACY_BODY_RELEASE_ID
    assert public == legacy


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (LEGACY_BODY_RELEASE_ID, BODY_RELEASE_ID),
        (f"{LEGACY_BODY_RELEASE_ID}:pose_tokens", f"{BODY_RELEASE_ID}:pose_tokens"),
        (BODY_RELEASE_ID, BODY_RELEASE_ID),
        (f"{BODY_RELEASE_ID}:pose_tokens", f"{BODY_RELEASE_ID}:pose_tokens"),
        (None, None),
        ("another-body-model", "another-body-model"),
        (f"{LEGACY_BODY_RELEASE_ID}:backbone", f"{LEGACY_BODY_RELEASE_ID}:backbone"),
        (f"{LEGACY_BODY_RELEASE_ID}-other", f"{LEGACY_BODY_RELEASE_ID}-other"),
    ],
)
def test_release_aliases_only_map_the_known_model_and_features(value, expected):
    assert canonical_pose_release_id(value) == expected


@pytest.mark.parametrize("command", ["pose-image", "pose-video"])
@pytest.mark.parametrize(
    ("release", "expected"),
    [(None, BODY_RELEASE_ID), ("body", BODY_RELEASE_ID), ("body_sg6", LEGACY_BODY_RELEASE_ID)],
)
def test_pose_cli_selects_public_or_legacy_body_preset(monkeypatch, command, release, expected):
    args = [
        command,
        "--task",
        "body",
        "--input",
        "input.png",
        "--boxes",
        "boxes.json",
        "--checkpoint",
        "body.ckpt",
        "--soma-assets",
        "assets",
        "--output",
        "body.pt",
    ]
    if release is not None:
        args.extend(["--release", release])
    options = build_parser().parse_args(args)
    monkeypatch.setattr(api, "checkpoint_pose_release", lambda path: None)
    loaded = []
    predictor = object()

    def fake_load(checkpoint, config, *, soma_assets, device):
        loaded.append((checkpoint, config.pose_release_id, soma_assets, device))
        return predictor

    monkeypatch.setattr(api, "load_pose_predictor", fake_load)
    actual, config = load_predictor(
        options.checkpoint,
        task=options.task,
        release=options.release,
        soma_assets=options.soma_assets,
        device=options.device,
    )
    assert actual is predictor
    assert config["pose_release_id"] == expected
    assert loaded == [("body.ckpt", expected, "assets", "cpu")]


@pytest.mark.parametrize("command", ["pose-image", "pose-video"])
def test_pose_cli_help_uses_public_preset_name(command, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "body, hand" in help_text
    assert "body_sg6" not in help_text


def test_default_pose_release_preserves_checkpoint_metadata(monkeypatch):
    saved = api.body_native_config(pose_release_id="existing-private-body-release")
    monkeypatch.setattr(api, "checkpoint_pose_release", lambda path: saved)
    monkeypatch.setattr(api, "load_pose_predictor", lambda path, config, **kwargs: config)
    predictor, config = load_predictor("body.ckpt", task="body", soma_assets="assets", device="cpu")
    assert predictor is saved
    assert config == asdict(saved)


@pytest.mark.parametrize(
    ("pose", "expected"),
    [
        ({}, BODY_RELEASE_ID),
        ({"release": "body"}, BODY_RELEASE_ID),
        ({"release": "body_sg6"}, LEGACY_BODY_RELEASE_ID),
    ],
)
def test_body_video_selects_public_or_legacy_preset(pose, expected):
    assert body_release({"pose": pose}).pose_release_id == expected


def test_body_video_preserves_explicit_release_mapping():
    release = api.body_native_config(pose_release_id="existing-private-body-release")
    assert body_release({"pose": {"release": asdict(release)}}) == release
