# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Pose checkpoint with an explicit prepared benchmark protocol."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("pose",))
    parser.add_argument("--task", choices=("body", "hand"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Override predictor checkpoint in config")
    parser.add_argument("--device", help="Override configured device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from .runner import evaluate_dataset

    config = OmegaConf.load(args.config)
    if args.checkpoint is not None:
        config.predictor.checkpoint = str(args.checkpoint)
    device = args.device or config.get("device", "cuda")
    predictor = instantiate(config.predictor, device=device)
    if predictor.model.task.name != args.task:
        raise ValueError("Evaluation task does not match checkpoint release")
    evaluator, dataset = instantiate(config.evaluator), instantiate(config.data)
    loader_options = (
        OmegaConf.to_container(config.loader, resolve=True) if "loader" in config else {}
    )
    result = evaluate_dataset(
        predictor,
        evaluator,
        dataset,
        precision=config.get("precision", "32-true"),
        **loader_options,
    )
    result.update(
        format="gem_pose_evaluation_v1",
        task=args.task,
        dataset=config.get("name", "prepared"),
        protocol=evaluator.protocol,
        pose_release_id=predictor.model.pose_release_id,
        input_rows=len(dataset),
        precision=config.get("precision", "32-true"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
