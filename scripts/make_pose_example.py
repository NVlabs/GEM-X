# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create clearly synthetic prepared Pose-body samples for training plumbing.

No model, data download, annotation or fitting is performed. Parameter/camera
labels are artificial legal values; keypoint confidence is zero because no SOMA
projection is evaluated by this generator. This is not a real training dataset.
"""

import argparse
from pathlib import Path

import torch


def make_samples(count=8, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator().manual_seed(seed)
    axis = torch.linspace(0, 1, 256)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    samples = []
    for index in range(count):
        color = torch.rand(3, 1, 1, generator=generator) * 0.3
        rgb = (torch.stack((xx, yy, (xx + yy) / 2)) * 0.6 + color).clamp(0, 1)
        samples.append(
            {
                "img": (rgb - mean) / std,
                "bbx_xys": torch.tensor([128.0, 128.0, 256.0]),
                "K_fullimg": torch.tensor([[256.0, 0, 128.0], [0, 256.0, 128.0], [0, 0, 1.0]]),
                "img_size": torch.tensor([256.0, 256.0]),
                "soma_body_pose": torch.randn(76, 3, generator=generator) * 0.02,
                "soma_global_orient": torch.zeros(3),
                "soma_identity_coeffs": torch.zeros(128),
                "soma_scale_params": torch.ones(56),
                "soma_global_scale": torch.tensor(1.0),
                "soma_transl_m": torch.tensor([0.0, 0.0, 2.0]),
                "pred_cam_gt": torch.tensor([1.0, 0.0, 0.0]),
                "keypoints_2d": torch.zeros(77, 3),
                "keypoints_3d": torch.zeros(77, 4),
                "landmarks_2d": torch.zeros(308, 3),
                "has_soma_params": torch.tensor(1.0),
                "has_cam_gt": torch.tensor(1.0),
                "aug_rot": torch.tensor(0.0),
                "sample_id": f"synthetic-example-{index}",
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = make_samples(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "gem_pose_prepared_body_v1",
            "synthetic": True,
            "description": "Synthetic RGB gradients and artificial finite native SOMA parameter/camera targets; zero keypoint confidence. Not real training or accuracy evidence.",
            "seed": args.seed,
            "samples": samples,
        },
        args.output,
    )
    print(f"Saved {len(samples)} SYNTHETIC prepared Pose-body samples to {args.output}")


if __name__ == "__main__":
    main()
