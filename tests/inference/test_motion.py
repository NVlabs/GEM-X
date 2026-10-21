# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict

import pytest
import torch
from torch import nn

from gem.common.geometry.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from gem.inference.cli import main
from gem.inference.motion import TimedMotionPipeline, load_predictor, run_motion
from gem.motion.api import MotionConditionSpec, MotionPredictor
from gem.motion.codec.soma_native_fixed_gs1 import SomaNativeFixedGs1EnDecoder
from gem.motion.frontend.time import ReferenceMotionTime
from gem.motion.network import R3MatchedGEMNetwork
from gem.motion.pipeline import MotionPipeline

SPEC = MotionConditionSpec("test:pose_tokens", "native-test", "soma29_nofinger_v1")
NETWORK = {"latent_dim": 32, "num_layers": 1, "num_heads": 4, "obs_num_joints": 29}


def sequence(spec=SPEC, network=NETWORK):
    return {
        "condition_spec": asdict(spec),
        "length": torch.tensor([3]),
        "obs": torch.randn(1, 3, network["obs_num_joints"], 3),
        "f_imgseq": torch.randn(1, 3, 1024),
        "bbx_xys": torch.tensor([20.0, 12.0, 18.0]).expand(1, 3, 3),
        "K_fullimg": torch.tensor([[800.0, 0, 20], [0, 800.0, 12], [0, 0, 1.0]]).expand(1, 3, 3, 3),
        "cam_angvel": matrix_to_rotation_6d(axis_angle_to_matrix(torch.randn(1, 3, 3) * 0.01)),
        "sampling_fps": 20.0,
    }


def checkpoint(tmp_path, network_options=NETWORK):
    torch.manual_seed(192)
    network = R3MatchedGEMNetwork(**network_options)
    codec = SomaNativeFixedGs1EnDecoder(stats_mean=torch.zeros(655), stats_std=torch.ones(655))
    # Non-default effective statistics must be loaded, not rebuilt from raw std.
    codec.std[10] = 1.5
    pipeline = MotionPipeline(network, codec).eval()
    path = tmp_path / "motion.ckpt"
    torch.save(
        {"state_dict": {"pipeline." + key: value for key, value in pipeline.state_dict().items()}},
        path,
    )
    return path, pipeline


@pytest.mark.parametrize(
    "num_joints, joint_set", [(29, "soma29_nofinger_v1"), (77, "soma77_native_v1")]
)
def test_native_checkpoint_codec_and_clock_match_direct_pipeline(tmp_path, num_joints, joint_set):
    network = {**NETWORK, "obs_num_joints": num_joints}
    spec = MotionConditionSpec(SPEC.pose_release_id, SPEC.observation_source, joint_set)
    path, expected = checkpoint(tmp_path, network)
    config = {"network": network, "condition_spec": asdict(spec), "time_options": {}}
    predictor = load_predictor(path, config)
    inputs = sequence(spec, network)
    timing = ReferenceMotionTime()
    prepared = timing.prepare_pipeline(inputs, train=False, postproc=False)
    with torch.inference_mode():
        original = expected({**prepared, "positions": timing.clock(prepared)})
    actual = run_motion(predictor, inputs)
    assert predictor.pipeline.pipeline.endecoder.std[10] == 1.5
    for key in ("pred_soma_params_incam", "pred_soma_params_global"):
        for field, value in original[key].items():
            torch.testing.assert_close(actual[key][field], value, atol=0, rtol=0)
    altered = {**inputs, "condition_spec": {**asdict(spec), "observation_source": "other"}}
    with pytest.raises(ValueError, match="do not match"):
        run_motion(predictor, altered)
    with pytest.raises(ValueError, match="omit --stats"):
        load_predictor(path, config, stats_path="unneeded.npz")


def test_cached_motion_cli_runs_with_native_stats(tmp_path):
    path, _ = checkpoint(tmp_path)
    config = tmp_path / "release.json"
    config.write_text(json.dumps({"network": NETWORK, "condition_spec": asdict(SPEC)}))
    data = tmp_path / "conditions.pt"
    torch.save(sequence(), data)
    output = tmp_path / "predictions.pt"
    main(
        [
            "motion-cache",
            "--task",
            "body",
            "--checkpoint",
            str(path),
            "--config",
            str(config),
            "--input",
            str(data),
            "--output",
            str(output),
        ]
    )
    result = torch.load(output, weights_only=False)
    assert result["format"] == "gem_motion_predictions_v1"
    assert result["sampling_fps"] == 20.0
    assert result["release"]["condition_spec"] == asdict(SPEC)
    assert result["pred_soma_params_global"]["transl"].shape == (1, 3, 3)
    state = torch.load(path, weights_only=False)["state_dict"]
    torch.save(
        {
            "state_dict": {
                key: value
                for key, value in state.items()
                if not key.startswith("pipeline.endecoder.")
            }
        },
        path,
    )
    with pytest.raises(ValueError, match="requires its training --stats"):
        load_predictor(path, {"network": NETWORK, "condition_spec": asdict(SPEC)})


def test_native_observation_preparation_precedes_physical_time():
    class RecordingPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.denoiser3d = type("Network", (), {"obs_num_joints": 29})()

        def forward(self, inputs):
            return inputs

    inputs = sequence()
    inputs.pop("obs")
    inputs["kp2d_native77"] = torch.ones(1, 3, 77, 3)
    inputs["kp2d_native77"][..., :2] *= 15
    predictor = MotionPredictor(TimedMotionPipeline(RecordingPipeline(), {}), SPEC)
    output = predictor.run(inputs)
    assert output["obs"].shape == (1, 3, 29, 3)
    torch.testing.assert_close(output["positions"], torch.tensor([[0, 1.5, 3.0]]))
    torch.testing.assert_close(output["motion_reference_step_scale"], torch.full((1, 3), 1.5))
    torch.testing.assert_close(output["cam_angvel_source_interval"], inputs["cam_angvel"])
    inputs["positions"] = torch.tensor([[0.0, 2.0, 5.0]])
    output = predictor.run(inputs)
    torch.testing.assert_close(output["positions"], inputs["positions"])
    torch.testing.assert_close(
        output["motion_reference_step_scale"], torch.tensor([[2.0, 3.0, 3.0]])
    )


@pytest.mark.parametrize("metadata_key", ["gem_motion_training", "hyper_parameters"])
@pytest.mark.parametrize("field", ["observation_schema", "pose_release_id"])
def test_native_checkpoint_semantics_reject_incompatible_release(tmp_path, metadata_key, field):
    path, _ = checkpoint(tmp_path)
    saved = torch.load(path, weights_only=False)
    metadata = {
        "observation_schema": SPEC.observation_joint_set,
        "pose_release_id": SPEC.pose_release_id,
    }
    saved[metadata_key] = metadata
    torch.save(saved, path)
    config = {"network": NETWORK, "condition_spec": asdict(SPEC)}
    load_predictor(path, config)
    metadata[field] = (
        "soma77_native_v1" if field == "observation_schema" else "different:pose_tokens"
    )
    torch.save(saved, path)
    with pytest.raises(ValueError, match="Checkpoint.*differs"):
        load_predictor(path, config)


def test_release_joint_count_rejected_before_weight_loading(tmp_path):
    with pytest.raises(ValueError, match="joint count differs"):
        load_predictor(
            tmp_path / "unread.ckpt",
            {"network": {**NETWORK, "obs_num_joints": 30}, "condition_spec": asdict(SPEC)},
        )


def test_compact30_release_rejected_before_weight_loading(tmp_path):
    spec = MotionConditionSpec("test:pose_tokens", "sapiens-test", "sapiens70_nofinger30_v1")
    with pytest.raises(ValueError, match="Unknown native SOMA observation schema"):
        load_predictor(
            tmp_path / "unread.ckpt",
            {"network": {**NETWORK, "obs_num_joints": 30}, "condition_spec": asdict(spec)},
        )


def test_release_requires_explicit_joint_count_before_weight_loading(tmp_path):
    network = {key: value for key, value in NETWORK.items() if key != "obs_num_joints"}
    with pytest.raises(KeyError, match="obs_num_joints"):
        load_predictor(
            tmp_path / "unread.ckpt", {"network": network, "condition_spec": asdict(SPEC)}
        )


@pytest.mark.parametrize("metadata_key", ["gem_motion_training", "hyper_parameters"])
@pytest.mark.parametrize("legacy_saved", [False, True])
def test_body_feature_alias_loads_existing_motion_checkpoint(tmp_path, metadata_key, legacy_saved):
    from gem.pose.release_ids import BODY_RELEASE_ID, LEGACY_BODY_RELEASE_ID

    public, legacy = f"{BODY_RELEASE_ID}:pose_tokens", f"{LEGACY_BODY_RELEASE_ID}:pose_tokens"
    saved_id, configured_id = (legacy, public) if legacy_saved else (public, legacy)
    path, expected = checkpoint(tmp_path)
    saved = torch.load(path, weights_only=False)
    saved[metadata_key] = {
        "pose_release_id": saved_id,
        "observation_schema": SPEC.observation_joint_set,
    }
    torch.save(saved, path)
    config = {
        "network": NETWORK,
        "condition_spec": {**asdict(SPEC), "pose_release_id": configured_id},
    }
    predictor = load_predictor(path, config)
    actual_state, expected_state = predictor.pipeline.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
    assert predictor.conditions.pose_release_id == public
    assert config["condition_spec"]["pose_release_id"] == configured_id
    assert torch.load(path, weights_only=False)[metadata_key]["pose_release_id"] == saved_id
    saved[metadata_key]["pose_release_id"] = public.replace("s240000", "s230000")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="pose feature release differs"):
        load_predictor(path, config)
