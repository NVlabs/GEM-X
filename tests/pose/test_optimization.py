# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from gem.pose.training.optimization import ConstantThenCosineLR, build_body_optimizer


def model():
    backbone = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    head = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
    backbone[0].bias.requires_grad_(False)
    head[1].bias.requires_grad_(False)
    return SimpleNamespace(backbone=backbone, task=SimpleNamespace(head=head))


def parameters(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def test_differential_groups_order_and_frozen_parameters_match_source():
    pose = model()
    optimizer = build_body_optimizer(pose, fused=False)
    groups = optimizer.param_groups
    assert [group["name"] for group in groups] == ["backbone", "head"]
    assert [group["lr"] for group in groups] == [5e-6, 1e-5]
    assert [id(parameter) for parameter in groups[0]["params"]] == [
        id(parameter) for parameter in parameters(pose.backbone)
    ]
    assert [id(parameter) for parameter in groups[1]["params"]] == [
        id(parameter) for parameter in parameters(pose.task.head)
    ]
    assert all(group["weight_decay"] == 1e-4 and group["fused"] is False for group in groups)
    assert all(group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8 for group in groups)
    # LayerNorm weights and trainable biases belong to the same decay group.
    assert any(parameter is pose.backbone[1].weight for parameter in groups[0]["params"])
    assert any(parameter is pose.task.head[0].bias for parameter in groups[1]["params"])
    assert not pose.backbone[0].bias.requires_grad and not pose.task.head[1].bias.requires_grad


def test_explicit_single_group_keeps_head_then_backbone_order():
    pose = model()
    optimizer = build_body_optimizer(
        pose, lr=2e-5, backbone_lr=None, weight_decay=0.02, fused=False
    )
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    expected = parameters(pose.task.head) + parameters(pose.backbone)
    assert [id(parameter) for parameter in group["params"]] == [
        id(parameter) for parameter in expected
    ]
    assert group["lr"] == 2e-5 and group["weight_decay"] == 0.02


def test_fused_backend_is_explicit_and_amp_unscaling_is_left_to_lightning(monkeypatch):
    class AdamW:
        def __init__(self, groups, **kwargs):
            self.groups, self.options = groups, kwargs
            self._step_supports_amp_scaling = True

    monkeypatch.setattr(torch.optim, "AdamW", AdamW)
    optimizer = build_body_optimizer(model())
    assert optimizer.options == {"weight_decay": 1e-4, "fused": True}
    assert optimizer._step_supports_amp_scaling is False


def test_cpu_updates_and_optimizer_resume_match_source_adamw_construction():
    torch.manual_seed(39)
    pose = model()
    reference = deepcopy(pose)
    actual = build_body_optimizer(pose, fused=False)
    expected = torch.optim.AdamW(
        [
            {"params": parameters(reference.backbone), "lr": 5e-6, "name": "backbone"},
            {"params": parameters(reference.task.head), "lr": 1e-5, "name": "head"},
        ],
        weight_decay=1e-4,
        fused=False,
    )
    inputs = torch.tensor([[0.5, -0.2, 0.8], [0.1, 0.7, -0.4]])

    def update(network, optimizer):
        optimizer.zero_grad()
        loss = network.task.head(network.backbone(inputs)).square().mean()
        loss.backward()
        optimizer.step()

    update(pose, actual)
    update(reference, expected)
    # Resume native Adam moments and group metadata into another identical model.
    resumed_model = deepcopy(pose)
    resumed = build_body_optimizer(resumed_model, fused=False)
    resumed.load_state_dict(deepcopy(actual.state_dict()))
    for network, optimizer in ((pose, actual), (reference, expected), (resumed_model, resumed)):
        update(network, optimizer)
    for group, source_group, resumed_group in zip(
        actual.param_groups, expected.param_groups, resumed.param_groups
    ):
        for parameter, source, restored in zip(
            group["params"], source_group["params"], resumed_group["params"]
        ):
            torch.testing.assert_close(parameter, source, atol=0, rtol=0)
            torch.testing.assert_close(parameter, restored, atol=0, rtol=0)
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual.state[parameter][name], expected.state[source][name], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    actual.state[parameter][name], resumed.state[restored][name], atol=0, rtol=0
                )


def test_body_schedule_preserves_initial_rates_and_reaches_shared_absolute_floor():
    optimizer = build_body_optimizer(model(), fused=False)
    scheduler = ConstantThenCosineLR(
        optimizer, constant_steps=200000, decay_steps=40000, eta_min=1e-7
    )
    for step, expected in (
        (0, [5e-6, 1e-5]),
        (200000, [5e-6, 1e-5]),
        (220000, [2.55e-6, 5.05e-6]),
        (240000, [1e-7, 1e-7]),
        (280000, [1e-7, 1e-7]),
    ):
        scheduler.last_epoch = step
        torch.testing.assert_close(
            torch.tensor(scheduler.get_lr(), dtype=torch.float64),
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-14,
            atol=1e-20,
        )
