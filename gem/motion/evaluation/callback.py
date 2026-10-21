# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-deduplicated distributed metric collection for Lightning."""

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist

from gem.motion.training.recovery import capture_rng_state, restore_rng_state


def merge_sequence_records(rank_records):
    """Match the original rank-order update rule for sampler padding copies."""
    merged = {}
    for records in rank_records:
        for dataset, sequences in records.items():
            merged.setdefault(dataset, {}).update(sequences)
    return merged


def aggregate_sequence_records(records):
    """Concatenate per-frame arrays before averaging, separately per metric.

    Jitter has T-3 samples; RTE/MPJPE have T samples. Averaging sequence means
    would therefore change the established benchmark weighting.
    Foot sliding pools individual contact observations. If every sequence has
    no contact observations, that metric is unavailable and is not logged.
    """
    summary, counts = {}, {}
    for dataset, sequences in records.items():
        counts[dataset] = len(sequences)
        if not sequences:
            continue
        keys = next(iter(sequences.values())).keys()
        for name in keys:
            values = np.concatenate([metrics[name] for metrics in sequences.values()])
            if name == "fs" and not values.size:
                continue
            summary[f"val_metric_{dataset}/{name}"] = float(values.mean())
    return summary, counts


def gather_sequence_records(records):
    if not dist.is_available() or not dist.is_initialized():
        return records
    gathered = [None] * dist.get_world_size()
    # NCCL uses the current rank's CUDA device for object serialization.
    # Its receive buffers must be normal writable tensors, even when the
    # trainer evaluates under inference_mode.
    with torch.inference_mode(False):
        dist.all_gather_object(gathered, records)
    return merge_sequence_records(gathered)


class MotionEvaluationCallback(pl.Callback):
    """Collect raw validation metrics and isolate lazy eval runtime RNG use."""

    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator
        self.records = {}
        self.last_metrics = {}
        self.last_sequence_counts = {}
        self._training_rng = None

    def on_validation_start(self, trainer, pl_module):
        self._training_rng = capture_rng_state()
        self.records = {}
        self.last_metrics = {}
        self.last_sequence_counts = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        record = self.evaluator.evaluate(outputs, batch)
        metrics = {name: np.atleast_1d(value) for name, value in record["metrics"].items()}
        for name, values in metrics.items():
            if (not values.size and name != "fs") or not np.isfinite(values).all():
                raise ValueError(
                    f"Evaluation metric {record['dataset_id']}/{name} is empty or non-finite"
                )
        self.records.setdefault(record["dataset_id"], {})[record["vid"]] = metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        records = gather_sequence_records(self.records)
        self.last_metrics, self.last_sequence_counts = aggregate_sequence_records(records)
        if trainer.is_global_zero and self.last_metrics:
            pl_module.log_dict(self.last_metrics, sync_dist=False, rank_zero_only=True)
        self.records = {}

    def on_validation_end(self, trainer, pl_module):
        restore_rng_state(self._training_rng)
        self._training_rng = None

    on_test_start = on_validation_start
    on_test_batch_end = on_validation_batch_end
    on_test_epoch_end = on_validation_epoch_end
    on_test_end = on_validation_end
