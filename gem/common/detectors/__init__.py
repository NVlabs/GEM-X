# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional image detectors; tracking dependencies load only when requested."""

from importlib import import_module

_EXPORTS = {
    "ByteTracker": ".bytetrack",
    "TrackResult": ".tracks",
    "detect_track": ".tracks",
    "smooth_boxes": ".tracks",
    "square_crops": ".tracks",
    "YOLOXONNX": ".yolox",
    "HandDetection": ".hand",
    "RTDETRHandONNX": ".hand",
    "HandTrackResult": ".hand_tracking",
    "track_hand_detections": ".hand_tracking",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
