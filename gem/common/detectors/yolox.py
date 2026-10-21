# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit-file YOLOX person detection with optional ONNXRuntime."""

from __future__ import annotations

import cv2
import numpy as np


class YOLOXONNX:
    """YOLOX person detector using ONNX Runtime.

    Parameters
    ----------
    onnx_path : str
        Path to an existing YOLOX ONNX model. No model files are fetched.
    input_size : tuple
        Model input resolution (H, W).  Default (640, 640).
    score_thr : float
        Minimum person confidence threshold.
    nms_thr : float
        NMS IoU threshold.
    device : str
        "cuda:N", "cuda" or "cpu".
    """

    def __init__(self, onnx_path, input_size=(640, 640), score_thr=0.5, nms_thr=0.45, device="cpu"):
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
            raise ValueError("YOLOX device must be cpu, cuda, or cuda:N")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
        self.execution_providers = self.sess.get_providers()
        if device != "cpu" and "CUDAExecutionProvider" not in self.execution_providers:
            raise RuntimeError("ONNXRuntime did not activate the requested CUDA provider")
        self.input_size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    # -- public API ---------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect persons in a single BGR image.

        Returns
        -------
        boxes : ndarray (N, 4)  — xyxy in original image coords
        scores : ndarray (N,)
        """
        padded, ratio = self._preprocess(img_bgr)
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        ort_in = {self.sess.get_inputs()[0].name: inp}
        outputs = self.sess.run(None, ort_in)
        return self._postprocess(outputs[0], ratio)

    # -- internal -----------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        ratio = min(self.input_size[0] / h, self.input_size[1] / w)
        resized = cv2.resize(
            img,
            (int(w * ratio), int(h * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return padded, ratio

    def _postprocess(self, outputs: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        if outputs.shape[-1] == 5:
            # ONNX with NMS baked in: shape (1, N, 5) = [x1, y1, x2, y2, score]
            boxes = outputs[0, :, :4] / ratio
            scores = outputs[0, :, 4]
            mask = scores > self.score_thr
            return boxes[mask], scores[mask]

        # ONNX without NMS: shape (1, N, 85) for COCO
        # Decode grid predictions
        strides = [8, 16, 32]
        grids, exp_strides = [], []
        for s in strides:
            hs, ws = self.input_size[0] // s, self.input_size[1] // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            exp_strides.append(np.full((*grid.shape[:2], 1), s))
        grids = np.concatenate(grids, 1)
        exp_strides = np.concatenate(exp_strides, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * exp_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * exp_strides

        preds = outputs[0]
        # Convert cxcywh → xyxy
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2) / ratio
        y1 = (cy - bh / 2) / ratio
        x2 = (cx + bw / 2) / ratio
        y2 = (cy + bh / 2) / ratio
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        obj_conf = preds[:, 4]
        cls_scores = preds[:, 5:]  # 80 COCO classes
        person_score = obj_conf * cls_scores[:, 0]  # class 0 = person

        mask = person_score > self.score_thr
        boxes_f = boxes_xyxy[mask]
        scores_f = person_score[mask]

        # NMS
        keep = _nms(boxes_f, scores_f, self.nms_thr)
        return boxes_f[keep], scores_f[keep]


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Standard greedy NMS on (N, 4) xyxy boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep
