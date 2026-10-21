# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark aggregation and callback isolation across batches and sampler padding."""

import copy
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gem.common.evaluation import mean_joint_error
from gem.pose.data.evaluation import PreparedPoseEvaluationDataset
from gem.pose.evaluation.runner import evaluate_dataset, merge_sample_records


class PreparedEvaluator:
    """Minimal scalar metric for five caller-defined landmarks."""

    def evaluate(self, prediction, batch):
        return {"mpjpe_mm": mean_joint_error(prediction, batch["target_joints"]) * 1000}


class PreparedPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self):
        return self.anchor.device

    def predict_batch(self, batch):
        return batch["prepared_prediction"]


def benchmark_rows():
    generator = torch.Generator().manual_seed(93)
    rows = []
    for index in range(6):
        target = torch.randn(5, 3, generator=generator)
        rows.append(
            {
                "sample_id": f"sample{index}",
                "img": torch.zeros(3, 4, 4),
                "bbx_xys": torch.tensor([320.0, 240.0, 200.0]),
                "K_fullimg": torch.eye(3),
                "target_joints": target,
                "prepared_prediction": target + torch.tensor([index / 1000, 0.0, 0.0]),
            }
        )
    return rows


def test_batch_boundaries_and_padding_do_not_change_mean_error():
    rows = benchmark_rows()
    predictor = PreparedPrediction().train()
    one = evaluate_dataset(
        predictor, PreparedEvaluator(), PreparedPoseEvaluationDataset(rows), batch_size=1
    )
    padded = evaluate_dataset(
        predictor,
        PreparedEvaluator(),
        PreparedPoseEvaluationDataset([*rows, copy.deepcopy(rows[1])]),
        batch_size=3,
    )
    assert predictor.training
    assert one == padded
    assert one["sample_count"] == one["metric_sample_counts"]["mpjpe_mm"] == 6
    assert one["metrics"]["mpjpe_mm"] == pytest.approx(2.5, abs=1e-4)


def test_rank_padding_merge_uses_sample_identity():
    first = {"a": {"metrics": {"mpjpe_mm": 2.0}}, "b": {"metrics": {"mpjpe_mm": 4.0}}}
    second = {"c": {"metrics": {"mpjpe_mm": 8.0}}, "a": first["a"]}
    merged = merge_sample_records([first, second])
    assert list(merged) == ["a", "b", "c"]


def test_prepared_benchmark_rejects_training_crop_rotation():
    row = benchmark_rows()[0]
    row["aug_rot"] = torch.tensor(15.0)
    with pytest.raises(ValueError, match="training crop rotation"):
        PreparedPoseEvaluationDataset([row])[0]


@pytest.mark.parametrize("fail", [False, True])
def test_online_callback_restores_rng_and_exact_module_modes(monkeypatch, fail):
    from gem.pose.evaluation import callback
    from gem.pose.training.recovery import _capture_rng, _restore_rng

    owner = nn.Module()
    owner.pose_model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.3)).train()
    owner.pose_model[0].eval()
    owner.objective = nn.Module()
    owner.objective.geometry = nn.Identity().eval()
    owner.release = SimpleNamespace(conditioning={}, inference={})
    logged = []
    modes = [module.training for module in owner.modules()]

    def make_predictor(model, geometry, config):
        model.eval()
        return SimpleNamespace(model=model)

    def run(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(10)
        if fail:
            raise RuntimeError("benchmark data failure")
        return {"metrics": {"mpjpe_mm": 3.0}, "sample_count": 2}

    monkeypatch.setattr(callback, "PosePredictor", make_predictor)
    monkeypatch.setattr(callback, "evaluate_dataset", run)
    hook = callback.PoseEvaluationCallback([None, None], object(), name="example", every_n_steps=1)
    logger = SimpleNamespace(
        log_metrics=lambda values, **kwargs: logged.append(values), save=lambda: None
    )
    trainer = SimpleNamespace(global_step=2, is_global_zero=True, loggers=[logger])
    before = _capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(2))
    _restore_rng(before)
    if fail:
        with pytest.raises(RuntimeError, match="benchmark data failure"):
            hook.on_train_batch_end(trainer, owner, None, {}, 0)
        assert hook.last_step is None
    else:
        hook.on_train_batch_end(trainer, owner, None, {}, 0)
        assert hook.last_step == 2
        assert logged == [
            {
                "eval/example/mpjpe_mm": 3.0,
                "eval/example/eval_samples": 2,
                "eval/example/eval_total_samples": 2,
                "eval/example/eval_sample_coverage": 1.0,
            }
        ]
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert [module.training for module in owner.modules()] == modes


def test_evaluation_help_does_not_import_runtime():
    code = """
import builtins
original = builtins.__import__
def lightweight(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'soma', 'lightning', 'hydra'}:
        raise AssertionError('Eager runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = lightweight
from gem.pose.evaluation.cli import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_callback_explicit_phase_schedule_and_natural_completion(monkeypatch):
    from gem.pose.evaluation.callback import PoseEvaluationCallback

    steps = [*range(10001, 200000, 10000), *range(200000, 240001, 10000)]
    hook = PoseEvaluationCallback([None], object(), name="body", steps=steps)
    seen = []
    monkeypatch.setattr(hook, "_evaluate", lambda trainer, model: seen.append(trainer.global_step))
    trainer = SimpleNamespace(global_step=0, max_steps=240000, max_epochs=-1, current_epoch=0)
    for step in [1, 10000, 10001, 20000, 20001, 190001, 200000, 200001, 210000, 240000]:
        trainer.global_step = step
        hook.on_train_batch_end(trainer, None, None, None, 0)
    assert seen == [10001, 20001, 190001, 200000, 210000, 240000]
    seen.clear()
    trainer.global_step = 18000
    hook.on_train_end(trainer, None)
    assert seen == []
    trainer.global_step = 240000
    hook.on_train_end(trainer, None)
    assert seen == [240000]


def test_resume_evaluation_waits_for_training_start_and_persists_coverage(monkeypatch, tmp_path):
    import json

    from gem.pose.evaluation import callback

    owner = nn.Module()
    owner.pose_model = nn.Linear(2, 2)
    owner.objective = SimpleNamespace(geometry=nn.Identity())
    owner.release = SimpleNamespace(conditioning={}, inference={})
    monkeypatch.setattr(callback, "PosePredictor", lambda *args: object())
    monkeypatch.setattr(
        callback,
        "evaluate_dataset",
        lambda *args, **kwargs: {
            "sample_count": 2,
            "metric_sample_counts": {"mpjpe": 2},
            "metrics": {
                "mpjpe": 3.0,
                "camera_root_translation_requested": 1,
                "camera_root_translation_valid_samples": 1,
            },
        },
    )
    hook = callback.PoseEvaluationCallback(
        [None, None], object(), name="prepared", evaluate_on_resume=True, output_dir=tmp_path
    )
    from lightning.pytorch.loggers import CSVLogger

    logger = CSVLogger(tmp_path, name="logs", version="resume", flush_logs_every_n_steps=100)
    trainer = SimpleNamespace(
        global_step=18000, ckpt_path=None, is_global_zero=True, loggers=[logger]
    )
    hook.on_train_start(trainer, owner)
    assert hook.last_step is None
    trainer.ckpt_path = "step-018000.ckpt"
    hook.on_train_start(trainer, owner)
    saved = json.loads((tmp_path / "prepared/step-018000.json").read_text())
    assert saved["step"] == 18000
    assert saved["metrics"]["eval_samples"] == saved["metrics"]["eval_total_samples"] == 2
    assert saved["metrics"]["eval_sample_coverage"] == 1
    assert saved["metrics"]["camera_root_translation_full_dataset"] == 1
    assert saved["metrics"]["camera_root_translation_reportable"] == 0
    import csv

    with (tmp_path / "logs/resume/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["step"] == "18000"
    assert rows[0]["eval/prepared/mpjpe"] == "3.0"
    # The callback state prevents a duplicate at the same restored step.
    monkeypatch.setattr(
        callback, "evaluate_dataset", lambda *args, **kwargs: pytest.fail("duplicate")
    )
    hook.on_train_start(trainer, owner)
