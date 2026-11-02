# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch

from gem.motion.evaluation.runner import evaluate_index, load_config, summarize


class Predictor:
    def __init__(self):
        self.lengths = []

    def run(self, conditions):
        length = int(conditions["length"].item())
        self.lengths.append(length)
        params = {"global_orient": torch.zeros(1, length, 3)}
        return {
            "pred_soma_params_incam": params,
            "pred_soma_params_global": params,
            "static_conf_logits": torch.zeros(1, length, 6),
        }


class Evaluator:
    vertex_projector = None

    def evaluate(self, outputs, batch):
        value = float(outputs["pred_soma_params_global"]["global_orient"][0, 0])
        valid = int(batch["mask"].sum())
        return {"metrics": {"mpjpe": np.full(valid, value), "fs": np.array([])}}


class Postprocessor:
    def __init__(self):
        self.lengths = []

    def __call__(self, camera, world, logits):
        self.lengths.append(world["global_orient"].shape[1])
        return {
            "pred_soma_params_incam": {k: v + 1 for k, v in camera.items()},
            "pred_soma_params_global": {k: v + 1 for k, v in world.items()},
        }


def make_index(tmp_path, *, features=True, rate=30):
    conditions = {
        "length": torch.tensor([7]),
        "frame_times": torch.arange(7, dtype=torch.float64)[None] / rate,
    }
    if features:
        conditions.update(f_imgseq=torch.zeros(1, 7, 1024), kp2d_native77=torch.zeros(1, 7, 77, 3))
    record = {
        "dataset_id": "EMDB_1",
        "sequence_id": "example",
        "conditions": conditions,
        "mask": torch.tensor([1, 0, 1, 1, 0, 1, 1], dtype=torch.bool),
        "eval_targets": {"incam": torch.zeros(7, 24, 3)},
        "protocol": {"mask_protocol": "official", "box_protocol": "supplied"},
    }
    torch.save(record, tmp_path / "sequence.pt")
    entry = {"dataset_id": "EMDB_1", "sequence_id": "example", "path": "sequence.pt"}
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"format": "gem_motion_benchmark_index_v1", "sequences": [entry, entry]})
    )
    return path


def test_full_timeline_raw_pp_dedup_and_protocol_reporting(tmp_path):
    predictor, processor = Predictor(), Postprocessor()
    report = evaluate_index(predictor, Evaluator(), make_index(tmp_path), postprocessor=processor)
    assert predictor.lengths == processor.lengths == [7]
    assert report["sequence_count"] == 1 and report["valid_frames"] == 5
    assert report["results"]["raw"]["metrics"]["EMDB_1"]["mpjpe"] == 0
    assert report["results"]["pp"]["metrics"]["EMDB_1"]["mpjpe"] == 1
    assert report["results"]["pp"]["metric_sample_counts"]["EMDB_1"]["fs"] == 0
    assert report["results"]["pp"]["metrics"]["EMDB_1"]["fs"] is None
    assert report["sequences"][0]["input_protocol"]["box_protocol"] == "supplied"


def test_metric_samples_are_pooled_instead_of_sequence_means():
    metrics, counts = summarize(
        {
            "EMDB_2": {
                "short": {"mpjpe": np.array([10.0]), "jitter": np.array([2.0])},
                "long": {"mpjpe": np.array([0.0, 0.0, 0.0]), "jitter": np.array([])},
            }
        }
    )
    assert metrics["EMDB_2"] == {"mpjpe": 2.5, "jitter": 2.0}
    assert counts["EMDB_2"] == {"mpjpe": 4, "jitter": 1}


def test_unprepared_features_and_wrong_clock_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="stage features"):
        evaluate_index(
            Predictor(), Evaluator(), make_index(tmp_path, features=False), postprocess="raw"
        )
    with pytest.raises(ValueError, match="30 Hz"):
        evaluate_index(Predictor(), Evaluator(), make_index(tmp_path, rate=25), postprocess="raw")


def test_sequence_selection_requires_an_actual_match(tmp_path):
    path = make_index(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        evaluate_index(Predictor(), Evaluator(), path, postprocess="raw", sequences=["missing"])
    report = evaluate_index(
        Predictor(), Evaluator(), path, postprocess="raw", sequences=["EMDB_1/example"]
    )
    assert report["sequence_count"] == 1


def test_config_resolves_environment_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_TEST_SOMA", str(tmp_path / "soma"))
    path = tmp_path / "config.yaml"
    path.write_text(
        "motion:\n  checkpoint: weights.ckpt\nevaluation:\n  soma_assets: ${oc.env:BENCHMARK_TEST_SOMA}\n  mesh_projector: null\n  joint_projectors:\n    smpl24: smpl24.pt\n"
    )
    config = load_config(path)
    assert config["motion"]["checkpoint"] == str(tmp_path / "weights.ckpt")
    assert config["evaluation"]["joint_projectors"]["smpl24"] == str(tmp_path / "smpl24.pt")
    assert config["evaluation"]["mesh_projector"] is None


def test_gem_eval_dispatches_motion_without_loading_pose(monkeypatch):
    from gem.motion.evaluation import cli as motion_cli
    from gem.pose.evaluation import cli

    calls = []
    monkeypatch.setattr(motion_cli, "main", lambda argv: calls.append(argv))
    cli.main(["motion", "--help"])
    assert calls == [["--help"]]
