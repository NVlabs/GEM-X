# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark protocols for image Pose, separate from training objectives."""

from importlib import import_module

_EXPORTS = {
    "HandEvaluator": ".hand",
    "PoseEvaluationCallback": ".callback",
    "PoseVisualizationCallback": ".visualization",
    "evaluate_dataset": ".runner",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
