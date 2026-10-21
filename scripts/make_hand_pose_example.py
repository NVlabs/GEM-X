# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create SYNTHETIC canonical-left hand samples for training plumbing.

The RGB gradients and legal MHR-hand targets are artificial, not annotated hand
images or accuracy evidence. The training objective reconstructs 25-joint target
coordinates with FK; this generator does not load a model, fit or download data.
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
        angles = torch.randn(25, generator=generator) * 0.02
        poses = torch.eye(3).repeat(25, 1, 1)
        poses[:, 0, 0] = poses[:, 1, 1] = angles.cos()
        poses[:, 0, 1], poses[:, 1, 0] = -angles.sin(), angles.sin()
        kp2d, kp3d = torch.zeros(25, 3), torch.zeros(25, 4)
        kp2d[:, 2], kp3d[:, 3] = 1, 1
        samples.append(
            {
                "img": (rgb - mean) / std,
                "bbx_xys": torch.tensor([128.0, 128.0, 256.0]),
                "K_fullimg": torch.tensor([[256.0, 0, 128.0], [0, 256.0, 128.0], [0, 0, 1.0]]),
                "img_size": torch.tensor([256.0, 256.0]),
                "hand_poses": poses,
                "hand_identity_coeffs": torch.zeros(5),
                "hand_global_scale": torch.ones(1),
                "hand_scale_params": torch.zeros(26),
                "pred_cam_gt": torch.tensor([1.0, 0.0, 0.0]),
                "keypoints_2d": kp2d,
                "keypoints_3d": kp3d,
                "has_hand_params": torch.tensor(1.0),
                "has_cam_gt": torch.tensor(1.0),
                "aug_rot": torch.tensor(0.0),
                "sample_id": f"synthetic-hand-example-{index}",
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
            "schema": "gem_pose_prepared_hand_v1",
            "synthetic": True,
            "description": "Synthetic RGB gradients and artificial canonical-left MHR-hand targets. Keypoint coordinates are FK placeholders. Not real training or accuracy evidence.",
            "seed": args.seed,
            "samples": samples,
        },
        args.output,
    )
    print(f"Saved {len(samples)} SYNTHETIC prepared Pose-hand samples to {args.output}")


if __name__ == "__main__":
    main()
