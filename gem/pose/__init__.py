# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEM-Pose image-to-parameter models for body and hand."""

from importlib import import_module

_EXPORTS = {
    "PoseModel": ".model",
    "BodyTask": ".tasks",
    "HandTask": ".tasks",
    "PoseInferenceConfig": ".predictor",
    "PosePrediction": ".predictor",
    "PosePredictor": ".predictor",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
