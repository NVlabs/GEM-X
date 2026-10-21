# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SOMA77 image observations from the DINOv3 heatmap detector.

The model consumes the central 192 columns of an affine 256-square crop. Its
confidence is the raw heatmap maximum, with no sigmoid or score threshold.
"""

from collections.abc import Sequence
from os import PathLike

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from gem.common.backbones import (
    DinoV3Backbone,
    convert_dinov3_native_state_dict,
    dinov3_vith16plus_config,
)

SOMA77_FLIP = list(range(77))
for _left, _right in [
    (9, 10),
    *[(i, i + 28) for i in range(11, 39)],
    *[(i, i + 5) for i in range(67, 72)],
]:
    SOMA77_FLIP[_left], SOMA77_FLIP[_right] = _right, _left
SOMA77_FLIP = tuple(SOMA77_FLIP)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_native77_inputs(frames_rgb: Sequence[np.ndarray], bboxes: Tensor) -> Tensor:
    """Crop RGB uint8 frames using full-image ``(cx, cy, square_size)`` boxes.

    Return normalized CPU tensors of shape ``(T, 3, 256, 192)``. The 255-edge
    affine convention is part of this detector's training/inference contract.
    """
    output = np.empty((len(frames_rgb), 3, 256, 192), dtype=np.float32)
    destination = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
    for i, (frame, box) in enumerate(zip(frames_rgb, bboxes)):
        cx, cy, size = (float(value) for value in box)
        half = size / 2
        source = np.array(
            [[cx - half, cy - half], [cx + half, cy - half], [cx, cy]], dtype=np.float32
        )
        transform = cv2.getAffineTransform(source, destination)
        crop = cv2.warpAffine(frame, transform, (256, 256), flags=cv2.INTER_LINEAR)
        normalized = (crop.astype(np.float32) / 255.0 - _MEAN) / _STD
        output[i] = normalized[:, 32:224].transpose(2, 0, 1)
    return torch.from_numpy(output)


def flip_native77_heatmaps(heatmaps: Tensor) -> Tensor:
    """Undo horizontal augmentation and swap semantic left/right joint IDs."""
    return heatmaps[:, SOMA77_FLIP].flip(-1)


def decode_native77_heatmaps(heatmaps: Tensor, bboxes: Tensor) -> Tensor:
    """UDP edge mapping and quarter-pixel refinement; return CPU ``(B,77,3)``."""
    maps = heatmaps.detach().float().cpu().numpy()
    boxes = bboxes.detach().float().cpu()
    batch, joints, height, width = maps.shape
    flat = maps.reshape(batch, joints, -1)
    indices = flat.argmax(-1)
    x, y = (indices % width).astype(np.float32), (indices // width).astype(np.float32)
    for n in range(batch):
        for k in range(joints):
            ix, iy = int(x[n, k]), int(y[n, k])
            if 1 < ix < width - 1:
                x[n, k] += np.sign(maps[n, k, iy, ix + 1] - maps[n, k, iy, ix - 1]) * 0.25
            if 1 < iy < height - 1:
                y[n, k] += np.sign(maps[n, k, iy + 1, ix] - maps[n, k, iy - 1, ix]) * 0.25
    points = np.stack((x, y), axis=-1)
    center = boxes[:, :2].numpy()
    scale = (torch.cat((boxes[:, [2]] * 24 / 32, boxes[:, [2]]), dim=1) / 200).numpy()
    points[..., 0] = (
        points[..., 0] / (width - 1) * (scale[:, [0]] * 200) + center[:, [0]] - scale[:, [0]] * 100
    )
    points[..., 1] = (
        points[..., 1] / (height - 1) * (scale[:, [1]] * 200) + center[:, [1]] - scale[:, [1]] * 100
    )
    return torch.from_numpy(np.concatenate((points, flat.max(-1, keepdims=True)), axis=-1))


class Native77HeatmapHead(nn.Module):
    """Two learned 2x upsamplers, preserving the released checkpoint keys."""

    def __init__(self, in_channels=1280):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, 77, 1)

    def forward(self, features):
        return self.final_layer(self.deconv_layers(features))


class Native77Model(nn.Module):
    """DINOv3 ViT-H/16+ plus the native77 deconvolution head."""

    def __init__(self, backbone: DinoV3Backbone | None = None):
        super().__init__()
        if backbone is None:
            config = dinov3_vith16plus_config()
            config.layer_norm_eps = 1e-6
            config.pos_embed_rescale = None
            backbone = DinoV3Backbone(config=config, rotary_precision="bf16")
        self.backbone = backbone
        self.keypoint_head = Native77HeatmapHead(backbone.embed_dim)

    def forward(self, images):
        return self.keypoint_head(self.backbone(images))


class Native77Pose2D(nn.Module):
    """Batched RGB-to-keypoints inference with explicit detector release identity.

    ``predict`` accepts one square ``(cx,cy,size)`` box per frame and returns
    full-image pixels and raw confidence, in native SOMA77 joint order.
    """

    def __init__(self, model: nn.Module, *, release_id: str, batch_size: int = 16):
        super().__init__()
        if not release_id:
            raise ValueError("release_id must identify the caller-selected detector weights")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model.eval()
        self.release_id = release_id
        self.batch_size = batch_size
        self.eval()

    @torch.inference_mode()
    def predict(self, frames_rgb: Sequence[np.ndarray], bboxes: Tensor) -> Tensor:
        boxes = torch.as_tensor(bboxes, dtype=torch.float32, device="cpu")
        if boxes.shape != (len(frames_rgb), 3):
            raise ValueError("bboxes must have shape (number of RGB frames, 3), with cx, cy, size")
        if not torch.isfinite(boxes).all() or (boxes[:, 2] <= 0).any():
            raise ValueError("bboxes must be finite and have positive square sizes")
        if not len(frames_rgb):
            return torch.empty((0, 77, 3), dtype=torch.float32)
        device = next(self.model.parameters()).device
        results = []
        for start in range(0, len(frames_rgb), self.batch_size):
            batch_boxes = boxes[start : start + self.batch_size]
            inputs = crop_native77_inputs(frames_rgb[start : start + self.batch_size], batch_boxes)
            inputs = inputs.to(device)
            # The released detector uses FP32 inference, including its heatmap
            # head, with an intentional BF16 rotary roundtrip inside attention.
            with torch.autocast(device.type, enabled=False):
                original, flipped = self.model(torch.cat((inputs, inputs.flip(-1)))).chunk(2)
                heatmaps = (original + flip_native77_heatmaps(flipped)) * 0.5
            results.append(decode_native77_heatmaps(heatmaps, batch_boxes))
        return torch.cat(results)


def load_native77_detector(
    checkpoint: str | PathLike[str], *, release_id: str, device="cpu", batch_size: int = 16
) -> Native77Pose2D:
    """Load an existing native77 checkpoint; no model download is performed."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    converted = {
        f"backbone.encoder.{key}": value
        for key, value in convert_dinov3_native_state_dict(
            state, prefix="backbone.", rotary_precision="bf16"
        ).items()
    }
    converted.update(
        {key: value for key, value in state.items() if key.startswith("keypoint_head.")}
    )
    unknown = state.keys() - {
        key for key in state if key.startswith(("backbone.", "keypoint_head."))
    }
    if unknown:
        raise ValueError(f"Unknown detector checkpoint keys: {sorted(unknown)}")
    with torch.device("meta"):
        model = Native77Model()
    model.load_state_dict(converted, strict=True, assign=True)
    return Native77Pose2D(model.to(device), release_id=release_id, batch_size=batch_size)
