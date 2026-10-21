# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named tensor results saved as a single ordinary PyTorch file."""

from pathlib import Path

import torch


def tensor_tree(value, *, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tensor_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree(item, device=device) for item in value)
    return value


def save_predictions(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_tree(result), path)
