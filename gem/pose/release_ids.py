# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Pose release names and exact historical equivalents."""

BODY_RELEASE_ID = "gem-pose-body-s240000-v1"
LEGACY_BODY_RELEASE_ID = "gep2-sg6-a2-snow-cam1-cos1e7-s240000-v1"

_POSE_RELEASE_ALIASES = {
    LEGACY_BODY_RELEASE_ID: BODY_RELEASE_ID,
    f"{LEGACY_BODY_RELEASE_ID}:pose_tokens": f"{BODY_RELEASE_ID}:pose_tokens",
}


def canonical_pose_release_id(value: str | None) -> str | None:
    """Resolve the released Body model's known names without changing other IDs."""
    return _POSE_RELEASE_ALIASES.get(value, value)
