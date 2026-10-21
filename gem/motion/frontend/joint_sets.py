# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SOMA29/77 and historical compact30 observation contracts.

Native skeleton joints and surface landmarks are distinct observation sources.
The native noise associations transfer profile strengths, not coordinates.
Metadata is available without importing Torch, body assets, or a trainer.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ObservationSchema:
    """Joint order, flip correspondence and augmentation semantics."""

    name: str
    source_num_joints: int
    active_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    flip_permutation: tuple[int, ...]
    coco17_aug_sources: tuple[tuple[int, ...], ...]
    visibility_parents: tuple[tuple[int, ...], ...]
    lower_body_indices: tuple[int, ...]
    hand_endpoint_pairs: tuple[tuple[int, int], ...]
    leg_endpoint_pairs: tuple[tuple[int, int], ...]

    @property
    def num_joints(self):
        return len(self.active_indices)


SAPIENS70_NOFINGER30_SOURCE_INDICES = (
    *range(21),
    41,
    62,
    *range(63, 70),
)

SAPIENS70_NOFINGER30_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe_tip",
    "left_small_toe_tip",
    "left_heel",
    "right_big_toe_tip",
    "right_small_toe_tip",
    "right_heel",
    "right_wrist",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
)

SAPIENS70_NOFINGER30_V1 = ObservationSchema(
    name="sapiens70_nofinger30_v1",
    source_num_joints=70,
    active_indices=SAPIENS70_NOFINGER30_SOURCE_INDICES,
    joint_names=SAPIENS70_NOFINGER30_NAMES,
    flip_permutation=(
        0,
        2,
        1,
        4,
        3,
        6,
        5,
        8,
        7,
        10,
        9,
        12,
        11,
        14,
        13,
        18,
        19,
        20,
        15,
        16,
        17,
        22,
        21,
        24,
        23,
        26,
        25,
        28,
        27,
        29,
    ),
    # Feet inherit ankle statistics; elbow surface landmarks inherit elbow
    # statistics; acromia inherit shoulder statistics. COCO17 has no neck, so
    # the neck profile is the mean of its left/right shoulder rows.
    coco17_aug_sources=(
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (11,),
        (12,),
        (13,),
        (14,),
        (15,),
        (16,),
        (15,),
        (15,),
        (15,),
        (16,),
        (16,),
        (16,),
        (10,),
        (9,),
        (7,),
        (8,),
        (7,),
        (8,),
        (5,),
        (6,),
        (5, 6),
    ),
    visibility_parents=(
        (5, 6),
        (0,),
        (0,),
        (1,),
        (2,),
        (),
        (),
        (5,),
        (6,),
        (),
        (),
        (9,),
        (10,),
        (11,),
        (12,),
        (13,),
        (13,),
        (13,),
        (14,),
        (14,),
        (14,),
        (8,),
        (7,),
        (7,),
        (8,),
        (7,),
        (8,),
        (5,),
        (6,),
        (5, 6),
    ),
    lower_body_indices=tuple(range(9, 21)),
    # Semantic endpoint metadata remains available to schema consumers. The
    # released endpoint modifier is a coordinate no-op, not active augmentation.
    hand_endpoint_pairs=((22, 21),),
    leg_endpoint_pairs=((13, 14), (15, 18), (16, 19), (17, 20)),
)

SOMA77_NATIVE = "soma77_native_v1"
SOMA29_NOFINGER = "soma29_nofinger_v1"
SOMA_OBSERVATION_SCHEMAS = (SOMA77_NATIVE, SOMA29_NOFINGER)


def _hand_names(side):
    return tuple(
        f"{side}Hand{finger}{suffix}"
        for finger, suffixes in (
            ("Thumb", (1, 2, 3, "End")),
            ("Index", (1, 2, 3, 4, "End")),
            ("Middle", (1, 2, 3, 4, "End")),
            ("Ring", (1, 2, 3, 4, "End")),
            ("Pinky", (1, 2, 3, 4, "End")),
        )
        for suffix in suffixes
    )


SOMA77_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    *_hand_names("Left"),
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    *_hand_names("Right"),
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)
# Native SOMA29 excludes finger articulation; this is not the body33 layout.
SOMA29_SOURCE_INDICES = (*range(15), *range(39, 43), *range(67, 77))
SOMA77_FINGER_INDICES = (*range(15, 39), *range(43, 67))

# Profile strengths, not geometric joint correspondences. Torso/head joints
# absent from COCO receive an explicitly chosen regional mean/profile.
_COCO17_SOURCES = (
    (11, 12),
    (5, 6, 11, 12),
    (5, 6, 11, 12),
    (5, 6),
    (5, 6),
    (5, 6),
    (0,),
    (0,),
    (0,),
    (1,),
    (2,),
    (5,),
    (5,),
    (7,),
    (9,),
    *((9,),) * 24,
    (6,),
    (6,),
    (8,),
    (10,),
    *((10,),) * 24,
    (11,),
    (13,),
    (15,),
    (15,),
    (15,),
    (12,),
    (14,),
    (16,),
    (16,),
    (16,),
)

# Existing compact30 rows for COCO nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees and ankles. In compact30, right wrist precedes left.
_COCO17_TO_SAPIENS30 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 21, 9, 10, 11, 12, 13, 14)


def _visibility_parents():
    """An observation hierarchy: torso, shoulder/hip anchors stay independent."""
    parents = [()] * 77
    # Hips/spine/chest/neck/head are independent detection anchors. Missing
    # Hips must not hide the body; clavicle absence does not hide the arm.
    for child in (7, 8, 9, 10):
        parents[child] = (6,)
    for arm, forearm, hand in ((12, 13, 14), (40, 41, 42)):
        parents[forearm] = (arm,)
        parents[hand] = (forearm,)
        for start, count in (
            (hand + 1, 4),
            (hand + 5, 5),
            (hand + 10, 5),
            (hand + 15, 5),
            (hand + 20, 5),
        ):
            parents[start] = (hand,)
            for child in range(start + 1, start + count):
                parents[child] = (child - 1,)
    for hip in (67, 72):
        for child in range(hip + 1, hip + 5):
            parents[child] = (child - 1,)
    return tuple(parents)


_VISIBILITY_PARENTS = _visibility_parents()
_NAME_TO_INDEX = {name: index for index, name in enumerate(SOMA77_JOINT_NAMES)}
_FLIP77 = tuple(
    (
        _NAME_TO_INDEX[name.replace("Left", "Right", 1)]
        if name.startswith("Left")
        else _NAME_TO_INDEX[name.replace("Right", "Left", 1)] if name.startswith("Right") else index
    )
    for index, name in enumerate(SOMA77_JOINT_NAMES)
)


def _schema_fields(name):
    if name not in SOMA_OBSERVATION_SCHEMAS:
        raise ValueError(f"Unknown native SOMA observation schema: {name!r}")
    selected = tuple(range(77)) if name == SOMA77_NATIVE else SOMA29_SOURCE_INDICES
    compact = {source: index for index, source in enumerate(selected)}
    return dict(
        name=name,
        source_num_joints=77,
        active_indices=selected,
        joint_names=tuple(SOMA77_JOINT_NAMES[index] for index in selected),
        flip_permutation=tuple(compact[_FLIP77[index]] for index in selected),
        coco17_aug_sources=tuple(_COCO17_SOURCES[index] for index in selected),
        # Every retained body's observation parent is also retained. Finger
        # removal therefore does not require collapsing/reparenting the tree.
        visibility_parents=tuple(
            tuple(compact[parent] for parent in _VISIBILITY_PARENTS[index]) for index in selected
        ),
        lower_body_indices=tuple(compact[index] for index in range(67, 77)),
        hand_endpoint_pairs=((compact[14], compact[42]),),
        leg_endpoint_pairs=((compact[69], compact[74]),),
    )


def soma_observation_profile(name):
    """Pure metadata, available to launchers without Torch, assets or reference."""
    return {
        **_schema_fields(name),
        "num_joints": 77 if name == SOMA77_NATIVE else 29,
        "coordinate_semantics": "native_public_skeleton_joint_centres",
        "source_includes_virtual_root": False,
        "augmentation_algorithm": "observation_augmentation_at_schema_joint_count",
        "noise_profile_policy": "coco17_to_native_regional_mapping_v1",
        "noise_profile_calibration": "design_initialization_not_measured_vitpose_residuals",
        "visibility_policy": "independent_torso_and_limb_anchors_with_local_descendants",
        "joint_count_random_draw_policy": "independent29_or77_not_full77_then_subset",
        "image_policy": "existing_sapiens_style_2d_noise_and_probabilities_at_native_joint_count",
        "detected_noise_strength_proxy": {
            "source_schema": "sapiens70_nofinger30_v1",
            "coco17_to_source_rows": _COCO17_TO_SAPIENS30,
            "rule": "arithmetic_mean_xy_strengths_via_native_coco17_associations",
            "calibration": "transferred_sapiens_strength_proxy_not_measured_vitpose_residuals",
            "coordinate_mapping": False,
            "changes_confidence_policy": False,
        },
    }


def map_sapiens30_detected_noise(config, schema_name):
    """Return native source strength tables, preserving all other noise fields.

    This is an explicit recipe-time proxy conversion, not runtime mutation or
    a mapping of keypoint coordinates. Input tables are the existing calibrated
    Sapiens30 E/H ``source_joint_scales``; each native row averages the selected
    COCO-associated table rows independently for X and Y. Global bias/temporal
    standard deviations, rho and seed are preserved exactly as provided.
    """
    associations = _schema_fields(schema_name)["coco17_aug_sources"]
    mapped = {}
    for source, table in config["source_joint_scales"].items():
        if len(table) != 30 or any(len(row) != 2 for row in table):
            raise ValueError(f"{source} must provide a Sapiens30 strength table with shape(30,2)")
        values = [[float(value) for value in row] for row in table]
        if not all(math.isfinite(value) for row in values for value in row):
            raise ValueError(f"{source} Sapiens30 strength table contains nonfinite values")
        mapped[source] = [
            [
                sum(values[_COCO17_TO_SAPIENS30[index]][axis] for index in indices) / len(indices)
                for axis in (0, 1)
            ]
            for indices in associations
        ]
    result = deepcopy(dict(config))
    result["source_joint_scales"] = mapped
    return result


OBSERVATION_SCHEMAS = {
    name: ObservationSchema(**_schema_fields(name)) for name in SOMA_OBSERVATION_SCHEMAS
}
OBSERVATION_SCHEMAS[SAPIENS70_NOFINGER30_V1.name] = SAPIENS70_NOFINGER30_V1


def get_observation_schema(schema: str | ObservationSchema) -> ObservationSchema:
    if isinstance(schema, ObservationSchema):
        return schema
    try:
        return OBSERVATION_SCHEMAS[schema]
    except KeyError as error:
        raise ValueError(f"Unknown observation schema {schema!r}") from error


def get_soma_observation_schema(name: str) -> ObservationSchema:
    if name not in SOMA_OBSERVATION_SCHEMAS:
        raise ValueError(f"Unknown native SOMA observation schema: {name!r}")
    return OBSERVATION_SCHEMAS[name]


def select_observation_joints(
    joints: torch.Tensor, schema: str | ObservationSchema
) -> torch.Tensor:
    """Select a schema's compact observation order from its source landmarks."""
    import torch

    schema = get_observation_schema(schema)
    if joints.shape[-2] != schema.source_num_joints:
        raise ValueError(
            f"{schema.name} expects {schema.source_num_joints} source joints, got {joints.shape[-2]}"
        )
    indices = torch.tensor(schema.active_indices, dtype=torch.long, device=joints.device)
    return joints.index_select(-2, indices)


def flip_observation_keypoints(
    keypoints: torch.Tensor,
    width: float | torch.Tensor,
    schema: str | ObservationSchema,
) -> torch.Tensor:
    """Flip compact ``[..., J, C]`` keypoints using ``width - x``."""
    import torch

    schema = get_observation_schema(schema)
    if keypoints.shape[-2] != schema.num_joints or keypoints.shape[-1] < 2:
        raise ValueError(
            f"{schema.name} expects keypoints ending in ({schema.num_joints}, C>=2), "
            f"got {tuple(keypoints.shape)}"
        )
    permutation = torch.tensor(schema.flip_permutation, dtype=torch.long, device=keypoints.device)
    flipped = keypoints.index_select(-2, permutation).clone()
    flipped[..., 0] = (
        torch.as_tensor(width, dtype=flipped.dtype, device=flipped.device) - flipped[..., 0]
    )
    return flipped


def select_soma_observations(public77, name):
    """Select native joints before drawing schema-sized augmentation noise."""
    return select_observation_joints(public77, get_soma_observation_schema(name))
