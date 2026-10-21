# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Full-image pinhole rectification and source-raster hand box restoration."""

import cv2
import numpy as np


class FullImageUndistortion:
    """Retain raster dimensions and K while rectifying a calibrated source image.

    Output pixels describe a pinhole camera with the source camera's intrinsics.
    Its field of view can exclude wide-angle source content. Camera axes and 3D
    geometry do not change.
    """

    def __init__(self, camera):
        self.camera = camera
        self.width, self.height = camera.width, camera.height
        self.K = camera.K.astype(np.float32)
        map_xy = np.empty((self.height, self.width, 2), dtype=np.float32)
        for top in range(0, self.height, 128):
            yy, xx = np.mgrid[top : min(top + 128, self.height), : self.width]
            rays = np.stack(
                (
                    (xx - self.K[0, 2]) / self.K[0, 0],
                    (yy - self.K[1, 2]) / self.K[1, 1],
                    np.ones_like(xx),
                ),
                axis=-1,
            )
            pixels = camera.project(rays)
            map_xy[top : top + 128] = np.where(
                np.isfinite(pixels).all(axis=-1, keepdims=True), pixels, -1
            )
        self._maps = cv2.convertMaps(map_xy, None, cv2.CV_16SC2)

    def apply(self, image):
        """Remap one RGB image without changing its shape, dtype or channel order."""
        if image.shape[:2] != (self.height, self.width):
            raise ValueError("Camera calibration dimensions must match the decoded source raster")
        return cv2.remap(image, *self._maps, interpolation=cv2.INTER_LINEAR)

    def source_boxes(self, boxes, intrinsics=None):
        """Map rectified square boxes back to source-image square bounds."""
        return source_boxes(boxes, self.camera, self.K if intrinsics is None else intrinsics)


def source_boxes(boxes, camera, intrinsics=None):
    """Map (N, 3) pinhole boxes through a 17-by-17 interior sampling grid.

    Returns float32 source boxes and a boolean mask of boxes with at least one
    finite projected sample. Unmappable boxes are NaN. No raster clipping is
    applied; square bounds enclose every finite projected sample.
    """
    boxes = np.asarray(boxes)
    K = camera.K.astype(np.float32) if intrinsics is None else np.asarray(intrinsics)
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 17), np.linspace(-0.5, 0.5, 17))
    offsets = np.stack((gx, gy), axis=-1).reshape(-1, 2)
    mapped = np.full((len(boxes), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(boxes), dtype=bool)
    for index, bbox in enumerate(boxes):
        pixels = np.asarray(bbox[:2]) + offsets * bbox[2]
        rays = np.c_[(pixels - K[:2, 2]) / K.diagonal()[:2], np.ones(len(pixels))]
        projected = camera.project(rays)
        projected = projected[np.isfinite(projected).all(axis=-1)]
        if len(projected):
            low, high = projected.min(axis=0), projected.max(axis=0)
            mapped[index] = np.r_[(low + high) / 2, np.max(high - low)].astype(np.float32)
            valid[index] = True
    return mapped, valid
