# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB square crops with the pixel conventions used by body/hand releases."""

import cv2
import numpy as np
import torch

from gem.pose.tasks.hand import mirror_hand_camera

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_rgb(image, bbox, image_size=256, *, crop_convention="pose"):
    """Crop RGB uint8 pixels using the retained OpenCV affine sampling grid."""
    cx, cy, size = (float(value) for value in bbox)
    if crop_convention == "pose":
        transform = np.array(
            [
                [image_size / size, 0, image_size / 2 - cx * image_size / size],
                [0, image_size / size, image_size / 2 - cy * image_size / size],
            ],
            dtype=np.float32,
        )
    elif crop_convention == "motion_training":
        # The same nominal grid has different subpixel rounding when its
        # matrix is solved from FP32 points, as used for cached Body Pose features.
        source = np.asarray(
            ((cx - size / 2, cy - size / 2), (cx + size / 2, cy - size / 2), (cx, cy + size / 2)),
            dtype=np.float32,
        )
        destination = np.asarray(
            ((0, 0), (image_size, 0), (image_size / 2, image_size)), dtype=np.float32
        )
        transform = cv2.getAffineTransform(source, destination)
    else:
        raise ValueError(f"Unknown crop convention {crop_convention!r}")
    crop = cv2.warpAffine(image, transform, (image_size, image_size), flags=cv2.INTER_LINEAR)
    normalized = (crop.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy())


def prepare_pose_batch(
    images,
    bboxes,
    intrinsics=None,
    *,
    right_hand=None,
    image_size=256,
    crop_convention="pose",
    hand_mirror="full_image",
    device="cpu",
):
    """Prepare one crop per square ``[cx, cy, size]`` box in supplied RGB images.

    ``images`` is one HWC uint8 image shared by the boxes, or one image per box.
    Inputs describe an undistorted pinhole image; missing intrinsics use the
    retained Pose max(width,height) focal estimate. Right hands become canonical left.
    ``full_image`` mirrors pixels about W-1 before cropping; ``legacy_crop``
    mirrors the crop and uses the historical W-x metadata convention.
    Both original and canonical camera fields are retained in the batch.
    """
    if hand_mirror not in {"full_image", "legacy_crop"}:
        raise ValueError(f"Unknown hand crop convention {hand_mirror!r}")
    boxes = torch.as_tensor(
        bboxes if isinstance(bboxes, torch.Tensor) else np.asarray(bboxes),
        dtype=torch.float32,
        device="cpu",
    )
    if boxes.ndim != 2 or boxes.shape[-1] != 3 or len(boxes) == 0:
        raise ValueError("Provide at least one square bbox as (B,3) [cx,cy,size]")
    if not bool((boxes[:, 2] > 0).all()):
        raise ValueError("Crop sizes must be positive")
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images] * len(boxes)
    if len(images) != len(boxes):
        raise ValueError("Provide one RGB image per bbox, or one image shared by all boxes")
    for image in images:
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Images must be HWC RGB uint8 arrays")
    sizes = torch.tensor([[image.shape[1], image.shape[0]] for image in images]).float()
    if intrinsics is None:
        # Pose deployment uses max-side focal; Motion's independent fallback
        # uses the diagonal. Sharing that estimator would change old weights'
        # CLIFF inputs and recovered camera depths for rectangular images.
        cameras = torch.eye(3).expand(len(boxes), 3, 3).clone()
        focal = sizes.max(dim=-1).values
        cameras[:, 0, 0] = cameras[:, 1, 1] = focal
        cameras[:, :2, 2] = sizes / 2
    else:
        cameras = torch.as_tensor(intrinsics, dtype=torch.float32, device="cpu")
        if cameras.ndim == 2:
            cameras = cameras.expand(len(boxes), 3, 3)
        if cameras.shape != (len(boxes), 3, 3):
            raise ValueError("Intrinsics must have shape (3,3) or (B,3,3)")
    right = (
        torch.zeros(len(boxes), dtype=torch.bool)
        if right_hand is None
        else torch.as_tensor(right_hand, dtype=torch.bool, device="cpu")
    )
    if right.shape != (len(boxes),):
        raise ValueError("right_hand must contain one boolean per bbox")
    model_boxes, model_cameras = boxes.clone(), cameras.clone()
    crops = []
    for index, image in enumerate(images):
        if right[index]:
            model_boxes[index], model_cameras[index] = mirror_hand_camera(
                boxes[index],
                cameras[index],
                sizes[index, 0],
                calibrated=hand_mirror == "full_image",
            )
            if hand_mirror == "full_image":
                crop = crop_rgb(
                    np.ascontiguousarray(image[:, ::-1]),
                    model_boxes[index],
                    image_size,
                    crop_convention=crop_convention,
                )
            else:
                crop = crop_rgb(
                    image, boxes[index], image_size, crop_convention=crop_convention
                ).flip(-1)
        else:
            crop = crop_rgb(image, boxes[index], image_size, crop_convention=crop_convention)
        crops.append(crop)
    return {
        name: value.to(device)
        for name, value in {
            "img": torch.stack(crops),
            "bbx_xys": model_boxes,
            "K_fullimg": model_cameras,
            "img_size": sizes,
            "right_hand": right,
            "original_bbx_xys": boxes,
            "original_K_fullimg": cameras,
        }.items()
    }
