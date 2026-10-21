# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-independent distributed validation collection."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gem.motion.evaluation import MotionEvaluationCallback
from gem.motion.evaluation.callback import (
    aggregate_sequence_records,
    gather_sequence_records,
    merge_sequence_records,
)
from gem.motion.training.recovery import capture_rng_state, restore_rng_state


def test_aggregation_weights_frames_and_deduplicates_rank_padding():
    first = {
        "camera_fixture": {
            "short": {"mpjpe": np.array([1.0, 3.0])},
            "duplicate": {"mpjpe": np.array([100.0])},
        }
    }
    second = {
        "camera_fixture": {
            "long": {"mpjpe": np.array([5.0, 7.0, 9.0, 11.0])},
            "duplicate": {"mpjpe": np.array([2.0])},
        }
    }
    summary, counts = aggregate_sequence_records(merge_sequence_records([first, second]))
    assert counts == {"camera_fixture": 3}
    assert summary == {"val_metric_camera_fixture/mpjpe": 38 / 7}


def test_gather_disables_inference_mode_for_nccl_compatible_object_buffers(monkeypatch):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather(destination, records):
        assert not torch.is_inference_mode_enabled()
        destination[:] = [records, {"prepared": {"other": {"mpjpe": np.array([3.0])}}}]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with torch.inference_mode():
        actual = gather_sequence_records({"prepared": {"one": {"mpjpe": np.array([2.0])}}})
    assert set(actual["prepared"]) == {"one", "other"}


def test_callback_exposes_totals_resets_records_and_restores_training_rng():
    class Evaluator:
        def evaluate(self, outputs, batch):
            random.random()
            np.random.random()
            torch.rand(8)
            return {
                "dataset_id": "prepared",
                "vid": batch["vid"],
                "metrics": {"mpjpe": np.array([2.0, 4.0])},
            }

    callback = MotionEvaluationCallback(Evaluator())
    trainer = SimpleNamespace(is_global_zero=True)
    logged = []
    module = SimpleNamespace(log_dict=lambda values, **kwargs: logged.append((values, kwargs)))
    before = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(7))
    restore_rng_state(before)
    callback.on_validation_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, {}, {"vid": "clip"}, 0)
    callback.on_validation_batch_end(trainer, module, {}, {"vid": "clip"}, 1)
    callback.on_validation_epoch_end(trainer, module)
    callback.on_validation_end(trainer, module)
    actual = (random.random(), np.random.random(), torch.rand(7))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    assert callback.last_sequence_counts == {"prepared": 1}
    assert callback.last_metrics == {"val_metric_prepared/mpjpe": 3.0}
    assert callback.records == {}
    assert logged[0][1] == {"sync_dist": False, "rank_zero_only": True}
    callback.on_validation_start(trainer, module)
    assert callback.last_metrics == {} and callback.last_sequence_counts == {}
    callback.on_validation_end(trainer, module)


@pytest.mark.parametrize("foot_samples", [[[], [2.0, 4.0], [8.0]], [[], [], []]])
def test_callback_pools_available_foot_contacts_and_omits_unavailable_metric(foot_samples):
    callback = MotionEvaluationCallback(SimpleNamespace(evaluate=lambda outputs, batch: outputs))
    trainer = SimpleNamespace(is_global_zero=True)
    logged = []
    module = SimpleNamespace(log_dict=lambda values, **kwargs: logged.append(values))
    callback.on_validation_start(trainer, module)
    for index, samples in enumerate(foot_samples):
        callback.on_validation_batch_end(
            trainer,
            module,
            {
                "dataset_id": "world_fixture",
                "vid": f"clip-{index}",
                "metrics": {"mpjpe": np.array([index + 2.0]), "fs": np.array(samples)},
            },
            {},
            index,
        )
    callback.on_validation_epoch_end(trainer, module)
    callback.on_validation_end(trainer, module)
    expected = {"val_metric_world_fixture/mpjpe": 3.0}
    if any(foot_samples):
        expected["val_metric_world_fixture/fs"] = 14 / 3
    assert callback.last_sequence_counts == {"world_fixture": 3}
    assert callback.last_metrics == expected
    assert logged == [expected]


@pytest.mark.parametrize("metrics", [{"mpjpe": np.array([])}, {"fs": np.array([np.nan])}])
def test_callback_still_rejects_empty_required_metrics_and_nonfinite_foot_sliding(metrics):
    callback = MotionEvaluationCallback(SimpleNamespace(evaluate=lambda outputs, batch: outputs))
    with pytest.raises(ValueError, match="empty or non-finite"):
        callback.on_validation_batch_end(
            None,
            None,
            {"dataset_id": "world_fixture", "vid": "clip", "metrics": metrics},
            {},
            0,
        )


def _distributed_metrics_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2
    )
    records = {
        "camera_fixture": {
            "duplicate": {"mpjpe": np.array([10.0, 12.0]) if rank == 0 else np.array([30.0, 32.0])},
            f"unique{rank}": {"mpjpe": np.array([2.0]) if rank == 0 else np.array([4.0, 6.0, 8.0])},
        }
    }
    with torch.inference_mode():
        summary, counts = aggregate_sequence_records(gather_sequence_records(records))
    (Path(root) / f"rank{rank}.json").write_text(json.dumps({"metrics": summary, "counts": counts}))
    dist.destroy_process_group()


def test_two_process_actual_gather_deduplicates_and_publishes_same_metrics(tmp_path):
    mp.spawn(_distributed_metrics_worker, args=(str(tmp_path),), nprocs=2, join=True)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert records[0] == records[1]
    assert records[0]["counts"] == {"camera_fixture": 3}
    assert records[0]["metrics"]["val_metric_camera_fixture/mpjpe"] == pytest.approx(82 / 6)
