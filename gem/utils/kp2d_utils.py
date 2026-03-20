# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared 2D keypoint visualisation utilities."""

import functools
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gem.utils.pylogger import Log
from gem.utils.video_io_utils import get_video_lwh, get_video_reader

# PyTorch >=2.6 defaults weights_only=True, but our saved files contain numpy arrays.
_torch_load = functools.partial(torch.load, weights_only=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARENTS_77 = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
]

COCO_SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

# ---------------------------------------------------------------------------
# color palette (BGR for OpenCV)
# ---------------------------------------------------------------------------

_PART_COLORS_77 = {
    "torso": (0, 215, 255),  # gold
    "head": (180, 130, 255),  # light purple
    "left_arm": (0, 255, 100),  # green
    "left_hand": (130, 255, 130),  # light green
    "right_arm": (50, 130, 255),  # orange
    "right_hand": (130, 180, 255),  # light orange/salmon
    "left_leg": (255, 190, 0),  # cyan-blue
    "right_leg": (255, 0, 170),  # magenta
}

# Map each of the 77 joints to its body-part group
_JOINT_GROUP_77 = [""] * 77
for _j in range(0, 4):
    _JOINT_GROUP_77[_j] = "torso"
for _j in range(4, 11):
    _JOINT_GROUP_77[_j] = "head"
for _j in range(11, 14):
    _JOINT_GROUP_77[_j] = "left_arm"
for _j in range(14, 39):
    _JOINT_GROUP_77[_j] = "left_hand"
for _j in range(39, 42):
    _JOINT_GROUP_77[_j] = "right_arm"
for _j in range(42, 67):
    _JOINT_GROUP_77[_j] = "right_hand"
for _j in range(67, 72):
    _JOINT_GROUP_77[_j] = "left_leg"
for _j in range(72, 77):
    _JOINT_GROUP_77[_j] = "right_leg"

# Bone stickwidth (ellipse minor-axis): body/limb = 4, hands = 2, rest = 3
_BONE_STICKWIDTH_77 = [3] * 77
for _j in range(0, 4):
    _BONE_STICKWIDTH_77[_j] = 4  # torso
for _j in range(67, 77):
    _BONE_STICKWIDTH_77[_j] = 4  # legs
for _j in range(14, 39):
    _BONE_STICKWIDTH_77[_j] = 2  # left hand
for _j in range(42, 67):
    _BONE_STICKWIDTH_77[_j] = 2  # right hand

# Joint radius: body/limbs = 4, hands = 2
_JOINT_RADIUS_77 = [4] * 77
for _j in range(14, 39):
    _JOINT_RADIUS_77[_j] = 2
for _j in range(42, 67):
    _JOINT_RADIUS_77[_j] = 2

# Per-bone colors for the 19 COCO skeleton bones (BGR)
_COCO_BONE_COLORS = [
    (255, 190, 0),  # 0: 15-13 left leg
    (255, 190, 0),  # 1: 13-11 left leg
    (255, 0, 170),  # 2: 16-14 right leg
    (255, 0, 170),  # 3: 14-12 right leg
    (0, 215, 255),  # 4: 11-12 hip
    (0, 215, 255),  # 5: 5-11  left torso
    (0, 215, 255),  # 6: 6-12  right torso
    (0, 215, 255),  # 7: 5-6   shoulders
    (0, 255, 100),  # 8: 5-7   left upper arm
    (50, 130, 255),  # 9: 6-8   right upper arm
    (0, 255, 100),  # 10: 7-9  left forearm
    (50, 130, 255),  # 11: 8-10 right forearm
    (180, 130, 255),  # 12: 1-2  eyes
    (180, 130, 255),  # 13: 0-1  nose-left eye
    (180, 130, 255),  # 14: 0-2  nose-right eye
    (180, 130, 255),  # 15: 1-3  left ear
    (180, 130, 255),  # 16: 2-4  right ear
    (0, 255, 100),  # 17: 3-5  left ear-shoulder
    (50, 130, 255),  # 18: 4-6  right ear-shoulder
]

# Per-joint colors for 17 COCO joints
_COCO_JOINT_COLORS = [
    (180, 130, 255),  # 0: nose
    (180, 130, 255),  # 1: left eye
    (180, 130, 255),  # 2: right eye
    (180, 130, 255),  # 3: left ear
    (180, 130, 255),  # 4: right ear
    (0, 255, 100),  # 5: left shoulder
    (50, 130, 255),  # 6: right shoulder
    (0, 255, 100),  # 7: left elbow
    (50, 130, 255),  # 8: right elbow
    (0, 255, 100),  # 9: left wrist
    (50, 130, 255),  # 10: right wrist
    (255, 190, 0),  # 11: left hip
    (255, 0, 170),  # 12: right hip
    (255, 190, 0),  # 13: left knee
    (255, 0, 170),  # 14: right knee
    (255, 190, 0),  # 15: left ankle
    (255, 0, 170),  # 16: right ankle
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_bbx_xyxy(bbx_xyxy, window=5):
    """Apply moving-average smoothing to a (L, 4) bounding-box sequence."""
    if bbx_xyxy.shape[0] <= window:
        return bbx_xyxy
    kernel = torch.ones(1, 1, window, dtype=bbx_xyxy.dtype) / window
    # (L, 4) -> (4, 1, L) for conv1d, then back
    padded = bbx_xyxy.T.unsqueeze(1)  # (4, 1, L)
    pad_size = window // 2
    padded = torch.nn.functional.pad(padded, (pad_size, pad_size), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, kernel).squeeze(1).T  # (L, 4)
    return smoothed


def _open_cv2_writer(path, width, height, fps):
    """Open an OpenCV VideoWriter for mp4v output."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))


def _draw_ellipse_bone(canvas, pt1, pt2, color, stickwidth):
    """Draw a bone as a filled ellipse between two joints."""
    x1, y1 = pt1
    x2, y2 = pt2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    length = math.hypot(x1 - x2, y1 - y2)
    if length < 1:
        return
    angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
    polygon = cv2.ellipse2Poly(
        (int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
    )
    cv2.fillConvexPoly(canvas, polygon, color, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def render_2d_keypoints(video_path, vitpose_path, bbx_path, output_path, fps=30, conf_thr=0.5):
    """Render 2D keypoint overlay on a video.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    vitpose_path : str or Path
        Path to the vitpose ``.pt`` file (tensor of shape ``(L, J, 2/3)``).
    bbx_path : str or Path
        Path to the bounding-box ``.pt`` file.
    output_path : str or Path
        Destination path for the rendered overlay video.
    fps : int
        Frames per second for the output video.
    conf_thr : float
        Confidence threshold below which joints/bones are hidden.
    """
    if not Path(vitpose_path).exists():
        Log.info("[2D KP] Missing vitpose results. Skipping 2D keypoint render.")
        return

    vitpose = _torch_load(vitpose_path)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
    if isinstance(vitpose, np.ndarray):
        vitpose = torch.from_numpy(vitpose)
    assert vitpose.ndim == 3 and vitpose.shape[-1] >= 2, "vitpose expected (L, J, 2/3)"

    bbx = _torch_load(bbx_path)
    bbx.get("detected_bbx_xys", bbx.get("bbx_xys", None))

    reader = get_video_reader(video_path)
    writer = _open_cv2_writer(
        output_path, get_video_lwh(video_path)[1], get_video_lwh(video_path)[2], fps
    )
    for i, img_raw in tqdm(
        enumerate(reader), total=get_video_lwh(video_path)[0], desc="Render 2D KP"
    ):
        if i >= vitpose.shape[0]:
            break
        img = img_raw.copy()
        keypoints = vitpose[i].cpu().numpy()
        use_conf = keypoints.shape[1] == 3
        num_joints = keypoints.shape[0]

        if num_joints == 77:
            # Draw bones as filled ellipses
            for child_idx, parent_idx in enumerate(PARENTS_77):
                if parent_idx < 0:
                    continue
                if use_conf and (
                    keypoints[parent_idx][2] <= conf_thr or keypoints[child_idx][2] <= conf_thr
                ):
                    continue
                pt1 = keypoints[parent_idx][:2].tolist()
                pt2 = keypoints[child_idx][:2].tolist()
                group = _JOINT_GROUP_77[child_idx]
                color = _PART_COLORS_77[group]
                stickwidth = _BONE_STICKWIDTH_77[child_idx]
                cur_canvas = img.copy()
                _draw_ellipse_bone(cur_canvas, pt1, pt2, color, stickwidth)
                img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
            # Draw joints with dark outline + colored fill
            for j in range(num_joints):
                if use_conf and keypoints[j][2] <= conf_thr:
                    continue
                x, y = keypoints[j][:2].astype(int)
                group = _JOINT_GROUP_77[j]
                color = _PART_COLORS_77[group]
                radius = _JOINT_RADIUS_77[j]
                cv2.circle(img, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), max(radius - 1, 1), color, -1, cv2.LINE_AA)
        elif num_joints == 17:
            # Draw bones as filled ellipses
            for bone_idx, (a, b) in enumerate(COCO_SKELETON):
                if use_conf and (keypoints[a][2] <= conf_thr or keypoints[b][2] <= conf_thr):
                    continue
                pt1 = keypoints[a][:2].tolist()
                pt2 = keypoints[b][:2].tolist()
                cur_canvas = img.copy()
                _draw_ellipse_bone(cur_canvas, pt1, pt2, _COCO_BONE_COLORS[bone_idx], 4)
                img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
            # Draw joints with dark outline + colored fill
            for j in range(num_joints):
                if use_conf and keypoints[j][2] <= conf_thr:
                    continue
                x, y = keypoints[j][:2].astype(int)
                cv2.circle(img, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 3, _COCO_JOINT_COLORS[j], -1, cv2.LINE_AA)

        # # Draw bounding box
        # if bbx_xys is not None and i < len(bbx_xys):
        #     cx, cy, s = bbx_xys[i].detach().cpu().numpy().tolist()
        #     half = 0.5 * float(s)
        #     bx0, by0 = int(round(cx - half)), int(round(cy - half))
        #     bx1, by1 = int(round(cx + half)), int(round(cy + half))
        #     cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(img[..., ::-1])
    writer.release()
    reader.close()
    Log.info(f"[2D KP] Saved overlay to {output_path}")
