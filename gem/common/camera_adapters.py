# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-plane camera adapters shared by body and hand Pose inference."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SquarePixelTransform:
    """Resize a calibrated image toward equal horizontal and vertical focal lengths.

    This changes image coordinates only; camera-frame 3D points stay unchanged.
    All mappings follow OpenCV's half-pixel resize convention and use the actual
    integer output dimensions, including their rounding.
    """

    width: int
    height: int
    scale_x: float
    scale_y: float
    intrinsics: np.ndarray
    source_size: tuple[int, int]

    @classmethod
    def from_intrinsics(cls, intrinsics, width: int, height: int, tol: float = 1e-3):
        """Compute a resize with target focal ``max(fx, fy)``.

        Differences ``abs(fy / fx - 1) <= tol`` leave the image untouched.
        The zero-skew 3x3 calibration must describe the decoded source image.
        """
        K = np.asarray(intrinsics, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("Square-pixel correction requires finite 3x3 intrinsics")
        fx, fy = K[0, 0], K[1, 1]
        if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
            raise ValueError("Square-pixel correction requires positive focals and image size")
        if K[0, 1] != 0 or K[1, 0] != 0 or not np.array_equal(K[2], [0, 0, 1]):
            raise ValueError("Square-pixel correction requires zero-skew camera intrinsics")
        source_size = (int(width), int(height))
        if abs(fy / fx - 1) <= tol:
            return cls(width, height, 1.0, 1.0, K.astype(np.float32), source_size)
        focal = max(fx, fy)
        new_width = max(1, round(width * focal / fx))
        new_height = max(1, round(height * focal / fy))
        scale_x, scale_y = new_width / width, new_height / height
        corrected_K = np.eye(3, dtype=np.float32)
        corrected_K[0, 0] = scale_x * fx
        corrected_K[1, 1] = scale_y * fy
        corrected_K[0, 2] = scale_x * (K[0, 2] + 0.5) - 0.5
        corrected_K[1, 2] = scale_y * (K[1, 2] + 0.5) - 0.5
        return cls(new_width, new_height, scale_x, scale_y, corrected_K, source_size)

    @property
    def is_identity(self) -> bool:
        return (self.width, self.height) == self.source_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize RGB pixels while retaining their dtype and channel order."""
        if self.is_identity:
            return image
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def transform_points(self, points) -> np.ndarray:
        """Map source pixel coordinates ``(..., 2)`` to the corrected image."""
        points = np.asarray(points, dtype=np.float64)
        return ((points + 0.5) * [self.scale_x, self.scale_y] - 0.5).astype(np.float32)

    def inverse_points(self, points) -> np.ndarray:
        """Map corrected pixel coordinates ``(..., 2)`` to the source image."""
        points = np.asarray(points, dtype=np.float64)
        return ((points + 0.5) / [self.scale_x, self.scale_y] - 0.5).astype(np.float32)

    def transform_boxes(self, boxes) -> np.ndarray:
        """Map source ``(..., 3)`` center-x/center-y/side boxes without losing coverage.

        The side uses the larger scale, retaining the full source square within
        the new square after the anisotropic image resize.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        corrected = np.empty_like(boxes, dtype=np.float32)
        corrected[..., :2] = self.transform_points(boxes[..., :2])
        corrected[..., 2] = boxes[..., 2] * max(self.scale_x, self.scale_y)
        return corrected

    def inverse_boxes(self, boxes) -> np.ndarray:
        """Invert the center/side mapping of :meth:`transform_boxes`."""
        boxes = np.asarray(boxes, dtype=np.float64)
        original = np.empty_like(boxes, dtype=np.float32)
        original[..., :2] = self.inverse_points(boxes[..., :2])
        original[..., 2] = boxes[..., 2] / max(self.scale_x, self.scale_y)
        return original
