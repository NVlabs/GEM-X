# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB RT-DETRv2 hand detection from an explicit local ONNX model.

Only NumPy, Pillow and ONNXRuntime are needed. Class1 is anatomical left and
class2 anatomical right; class IDs are not screen-position assignments. The
original detector has no NMS. Tracking and framewise top-hand policy belong
to the caller.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

HAND_SIDES = {1: "left", 2: "right"}
INPUT_SIZE = 640


@dataclass
class HandDetection:
    label_id: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def side(self) -> str | None:
        """Unknown/unused classes have no anatomical hand side."""
        return HAND_SIDES.get(self.label_id)

    @property
    def cxcys(self) -> tuple[float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)


def refine_box(xyxy, img_w, img_h, pad=0.0, square=False, min_size=0.0, clip=True):
    """Apply the source pad→square→minimum→shift/clip order.

    ``pad`` grows each side, so side lengths multiply by ``1 + 2*pad``.
    Clipping first shifts a box inside the raster, preserving its size where
    possible. A box larger than the image is then shrunk to the image bounds.
    """
    x1, y1, x2, y2 = xyxy
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if pad:
        width, height = width * (1 + 2 * pad), height * (1 + 2 * pad)
    if square:
        width = height = max(width, height)
    if min_size:
        width, height = max(width, min_size), max(height, min_size)
    x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
    if clip:
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if x2 > img_w:
            x1 -= x2 - img_w
            x2 = float(img_w)
        if y2 > img_h:
            y1 -= y2 - img_h
            y2 = float(img_h)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    return x1, y1, x2, y2


class RTDETRHandONNX:
    """Per-image hand candidates, sorted by confidence with original box rules.

    Inputs are HWC RGB uint8 arrays. PIL bilinear resizing directly to640×640
    reproduces the trained image path; there is no letterbox or ImageNet
    normalization. ONNX receives original raster sizes as int64 ``[W,H]`` and
    returns boxes already in full-image pixels.

    Defaults retain rectangular candidates and the original clipping rule.
    A framewise square crop can request ``top1_per_class=True, pad=.15,
    square=True, clip=False`` for the GEP side×1.3 policy. Tracking uses its own
    candidate threshold and box rules; do not apply padding twice.
    """

    def __init__(
        self,
        onnx_path,
        *,
        device="cpu",
        score_thr=0.5,
        top1_per_class=False,
        pad=0.0,
        square=False,
        min_size=0.0,
        clip=True,
    ):
        import onnxruntime as ort

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda" or device.startswith("cuda:"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA detection requires ONNXRuntime with CUDAExecutionProvider")
            device_id = 0 if device == "cuda" else int(device.split(":")[1])
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError("Hand detector device must be cpu, cuda, or cuda:N")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=providers
        )
        self.execution_providers = self.session.get_providers()
        if device != "cpu" and "CUDAExecutionProvider" not in self.execution_providers:
            raise RuntimeError("ONNXRuntime did not activate the requested CUDA provider")
        self.score_thr = score_thr
        self.top1_per_class = top1_per_class
        self.pad, self.square, self.min_size, self.clip = pad, square, min_size, clip

    @staticmethod
    def _preprocess(image_rgb):
        image = Image.fromarray(image_rgb).convert("RGB")
        resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        return pixels.transpose(2, 0, 1)[None]

    def _run(self, batch, sizes):
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {"images": batch, "orig_target_sizes": sizes},
        )
        output = []
        for label, box, score, (width, height) in zip(labels, boxes, scores, sizes):
            keep = score > self.score_thr
            detections = [
                HandDetection(
                    int(label_id), float(confidence), tuple(float(value) for value in bounds)
                )
                for label_id, bounds, confidence in zip(label[keep], box[keep], score[keep])
            ]
            detections.sort(key=lambda detection: -detection.score)
            if self.top1_per_class:
                seen, selected = set(), []
                for detection in detections:
                    if detection.label_id not in seen:
                        seen.add(detection.label_id)
                        selected.append(detection)
                detections = selected
            if self.pad or self.square or self.min_size or self.clip:
                for detection in detections:
                    detection.bbox_xyxy = refine_box(
                        detection.bbox_xyxy,
                        width,
                        height,
                        self.pad,
                        self.square,
                        self.min_size,
                        self.clip,
                    )
            output.append(detections)
        return output

    def detect(self, image_rgb):
        """Return candidates for one RGB image; side labels are anatomical."""
        height, width = image_rgb.shape[:2]
        return self._run(self._preprocess(image_rgb), np.array([[width, height]], dtype=np.int64))[
            0
        ]

    def detect_batch(self, images_rgb):
        """Detect one non-empty image batch, retaining its frame order."""
        batch = np.concatenate([self._preprocess(image) for image in images_rgb], axis=0)
        sizes = np.array([[image.shape[1], image.shape[0]] for image in images_rgb], dtype=np.int64)
        return self._run(batch, sizes)
