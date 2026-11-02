# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body benchmark crops from locally prepared official dataset annotations."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _benchmark_crop(image, center, size, image_size):
    """Preserve the release benchmark's affine crop and channel rounding."""
    center = np.asarray(center, dtype=np.float64)
    half = np.float32(size * 0.5)
    source = np.asarray([center, center + [0, half], center + [half, 0]], dtype=np.float32)
    midpoint = image_size * 0.5
    destination = np.asarray(
        [[midpoint, midpoint], [midpoint, image_size], [image_size, midpoint]],
        dtype=np.float32,
    )
    transform = cv2.getAffineTransform(source, destination)
    image = cv2.warpAffine(
        image,
        transform,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    crop = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    std = 255.0 * np.array([0.229, 0.224, 0.225])
    for channel in range(3):
        crop[channel] = (crop[channel] - mean[channel]) / std[channel]
    return torch.from_numpy(crop)


class BodyBenchmarkDataset(Dataset):
    """Read an NPZ produced by ``gem.pose.evaluation.prepare_body``.

    ``dataset_root`` is the original 3DPW or EMDB download directory. The NPZ
    contains image paths relative to that root, final square crop sizes, camera
    intrinsics and benchmark joints in metres. No private annotation archive,
    temporal sidecar, model files or runtime SMPL dependency is needed here.
    """

    def __init__(self, dataset_file, dataset_root, *, image_size=256):
        self.dataset_root = Path(dataset_root)
        self.image_size = int(image_size)
        with np.load(dataset_file, allow_pickle=False) as data:
            self.protocol = str(data["protocol"].item())
            self.data = {
                name: data[name]
                for name in (
                    "imgname",
                    "center",
                    "box_size",
                    "intrinsics",
                    "keypoints_3d",
                    "sequence_id",
                    "person_id",
                    "frame_id",
                )
            }
        if self.protocol not in {"3dpw", "emdb"}:
            raise ValueError("Body benchmark protocol must be '3dpw' or 'emdb'")

    def __len__(self):
        return len(self.data["imgname"])

    def __getitem__(self, index):
        image_path = self.dataset_root / str(self.data["imgname"][index])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise OSError(f"Cannot read benchmark image {image_path}")
        height, width = image.shape[:2]
        center = self.data["center"][index]
        size = self.data["box_size"][index]
        sequence = str(self.data["sequence_id"][index])
        person, frame = (int(self.data[key][index]) for key in ("person_id", "frame_id"))
        return {
            "sample_id": f"{self.protocol}:{sequence}:{person}:{frame}",
            "img": _benchmark_crop(image, center, size, self.image_size),
            "bbx_xys": torch.tensor([*center, size], dtype=torch.float32),
            "K_fullimg": torch.tensor(self.data["intrinsics"][index], dtype=torch.float32),
            "img_size": torch.tensor([width, height], dtype=torch.float32),
            "keypoints_3d": torch.tensor(self.data["keypoints_3d"][index], dtype=torch.float32),
            "sequence_id": sequence,
            "person_id": person,
            "frame_id": frame,
            "source_idx": index,
            "imgname": str(image_path),
        }
