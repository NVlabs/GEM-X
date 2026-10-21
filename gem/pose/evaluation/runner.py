# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-deduplicated Pose evaluation with protocol-owned aggregation."""

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, default_collate

from gem.pose.api import PoseReleaseConfig, checkpoint_pose_release, load_pose_predictor


def load_evaluation_predictor(checkpoint, *, soma_assets, release=None, device="cpu"):
    """Load native metadata or an explicitly configured historical release."""
    if release is None:
        release = checkpoint_pose_release(checkpoint)
        if release is None:
            raise ValueError("Historical evaluation checkpoints require an explicit release")
    elif not isinstance(release, PoseReleaseConfig):
        release = PoseReleaseConfig(**release)
    return load_pose_predictor(checkpoint, release, soma_assets=soma_assets, device=device)


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {name: _to_device(item, device) for name, item in value.items()}
    return value


def _row(value, index):
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu()
    if isinstance(value, dict):
        return {name: _row(item, index) for name, item in value.items()}
    return value[index]


def merge_sample_records(rank_records):
    """Discard DistributedSampler padding using stable producer sample IDs."""
    merged = {}
    for records in rank_records:
        for sample_id, record in records.items():
            merged.setdefault(sample_id, record)
    return merged


def summarize_records(records, evaluator):
    names = sorted({name for row in records.values() for name in row["metrics"]})
    metrics, counts = {}, {}
    for name in names:
        values = [row["metrics"][name] for row in records.values() if name in row["metrics"]]
        values = [value for value in values if not math.isnan(value)]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Benchmark metric {name} contains infinite values")
        counts[name] = len(values)
        metrics[name] = sum(values) / len(values) if values else None
    if hasattr(evaluator, "summarize") and records:
        payload = default_collate([row["protocol"] for row in records.values()])
        for name, value in evaluator.summarize(payload).items():
            value = float(value)
            metrics[name] = value if math.isfinite(value) else None
    return {"metrics": metrics, "metric_sample_counts": counts, "sample_count": len(records)}


def evaluate_dataset(
    predictor,
    evaluator,
    dataset,
    *,
    batch_size=16,
    num_workers=0,
    precision="32-true",
):
    """Run prepared benchmark samples, retaining cross-batch protocol state.

    Evaluators return scalar-per-sample tensors. Optional ``protocol_values``
    and ``summarize`` retain pooled joint/sequence statistics across batch and
    rank boundaries. No batch-mean approximation or training loss is substituted.
    """
    if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
        raise ValueError("Evaluation precision must be 32-true, 16-mixed or bf16-mixed")
    distributed = dist.is_available() and dist.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    device = predictor.device
    if hasattr(evaluator, "to"):
        evaluator.to(device)
    states = [(module, module.training) for module in predictor.modules()]
    predictor.eval()
    records = {}
    try:
        for batch in loader:
            batch = _to_device(batch, device)
            context = (
                nullcontext()
                if precision == "32-true"
                else torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if precision == "16-mixed" else torch.bfloat16,
                )
            )
            with context:
                prediction = predictor.predict_batch(batch)
            # A protocol adapter may fit a differentiable model to frozen
            # predictions, so it owns its own gradient context.
            values = evaluator.evaluate(prediction, batch)
            payload = (
                evaluator.protocol_values(prediction, batch)
                if hasattr(evaluator, "protocol_values")
                else {}
            )
            sample_ids = batch["sample_id"]
            if any(tuple(value.shape) != (len(sample_ids),) for value in values.values()):
                raise ValueError("Pose evaluator metrics must have one scalar per sample")
            for index, sample_id in enumerate(sample_ids):
                records.setdefault(
                    sample_id,
                    {
                        "metrics": {
                            name: float(value[index].detach()) for name, value in values.items()
                        },
                        "protocol": _row(payload, index),
                    },
                )
    finally:
        for module, training in states:
            module.training = training
    if distributed:
        gathered = [None] * dist.get_world_size()
        with torch.inference_mode(False):
            dist.all_gather_object(gathered, records)
        records = merge_sample_records(gathered)
    return summarize_records(records, evaluator)
