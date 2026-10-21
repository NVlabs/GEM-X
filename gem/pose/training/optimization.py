# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizers and the optional constant-then-cosine Body Pose schedule."""

import math

import torch


class ConstantThenCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """Keep group rates constant, then decay each to the same absolute floor.

    The body training recipe uses 200,000 constant-rate updates followed by 40,000
    cosine updates. Rates stay at the floor after the decay; there is no restart.
    Standard Lightning checkpoints retain the scheduler together with Adam.
    """

    def __init__(self, optimizer, *, constant_steps, decay_steps, eta_min):
        if constant_steps < 0 or decay_steps <= 0:
            raise ValueError("constant_steps must be nonnegative and decay_steps positive")
        if not 0 <= eta_min < min(group["lr"] for group in optimizer.param_groups):
            raise ValueError("eta_min must be nonnegative and below each initial learning rate")
        self.constant_steps = int(constant_steps)
        self.decay_steps = int(decay_steps)
        self.eta_min = float(eta_min)
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.constant_steps:
            return list(self.base_lrs)
        progress = min(max(self.last_epoch - self.constant_steps, 0) / self.decay_steps, 1)
        weight = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + (lr - self.eta_min) * weight for lr in self.base_lrs]


def build_pose_optimizer(
    model,
    *,
    lr=1e-5,
    backbone_lr=5e-6,
    weight_decay=1e-4,
    fused=True,
    additional_head_parameters=(),
):
    """Optimize Pose backbones, prediction heads and supplied learned loss state.

    Differential groups follow the source order: backbone, then head. An
    explicit ``backbone_lr=None`` retains the source single-group head-before-
    backbone order. Biases/norms use the same weight decay as other parameters.
    Existing frozen parameters are excluded; this function does not freeze or
    unfreeze modules. Direct 2D and learned-density parameters join the head
    group; SOMA geometry is never added to the optimizer.
    """
    lr, weight_decay = float(lr), float(weight_decay)
    backbone_lr = None if backbone_lr is None else float(backbone_lr)
    head = [parameter for parameter in model.task.head.parameters() if parameter.requires_grad]
    extra_head = []
    if getattr(model, "kp2d_head", None) is not None:
        extra_head.extend(
            parameter for parameter in model.kp2d_head.parameters() if parameter.requires_grad
        )
    extra_head.extend(
        parameter for parameter in additional_head_parameters if parameter.requires_grad
    )
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if backbone_lr is None:
        groups = [{"params": head + backbone + extra_head, "lr": lr}]
    else:
        groups = [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": head + extra_head, "lr": lr, "name": "head"},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay, fused=fused)
    if fused:
        # Match GEP: Lightning/GradScaler owns unscaling and overflow omission,
        # while AdamW's parameter groups still select the fused update kernel.
        optimizer._step_supports_amp_scaling = False
    return optimizer


# Keep the original body entry point and its defaults for existing configurations.
build_body_optimizer = build_pose_optimizer
