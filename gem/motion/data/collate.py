# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch adapters for public parameter/feature rows and private dataset plugins."""

import torch
from torch.utils.data import default_collate


def collate_image_rows(rows):
    """Add a T1 input axis while leaving original image supervision intact.

    Source datasets provide cached ``f_imgseq``, full-image ``bbx_xys`` and
    ``K_fullimg`` plus native SOMA and image-landmark labels. Native GT
    observations are computed by TrainingPreparation after device transfer.
    There are no invented world/contact targets or detector placeholders.
    """
    labels = default_collate(rows)
    return {
        "training_branch": "image",
        "length": torch.ones(len(rows), dtype=torch.long),
        "f_imgseq": labels["f_imgseq"][:, None],
        "bbx_xys": labels["bbx_xys"][:, None],
        "K_fullimg": labels["K_fullimg"][:, None],
        "image_labels": labels,
    }


def collate_evaluation_rows(rows):
    """Collate full-sequence evaluation tensors and retain metadata records.

    Evaluation loaders use one variable-length sequence per batch. Their
    frame mask describes benchmark GT availability, not training mask fields.
    """
    batch = {
        name: (
            [row[name] for row in rows]
            if name.startswith("meta")
            else default_collate([row[name] for row in rows])
        )
        for name in rows[0]
    }
    batch["B"] = len(rows)
    return batch


def _with_optional_root_targets(rows):
    """Supply masked-out root targets when only some sources have world labels."""
    if not any("motion_targets" in row for row in rows):
        return rows
    result = []
    for row in rows:
        if "motion_targets" in row:
            result.append(row)
            continue
        valid = row["mask"]["valid"]
        meta = row["meta"]
        if "source_time_seconds" in meta:
            length = row["length"]
            times = torch.empty(len(valid), dtype=torch.float64)
            times[:length] = torch.as_tensor(meta["source_time_seconds"], dtype=torch.float64)
            times[length:] = (
                times[length - 1]
                + torch.arange(1, len(valid) - length + 1, dtype=torch.float64)
                / meta["sampling_fps"]
            )
        else:
            times = torch.arange(len(valid), dtype=torch.float64) / 30.0
        zeros = row["soma_params_c"]["transl"].new_zeros(len(valid), 3)
        targets = {
            name: zeros.clone()
            for name in (
                "transl_w",
                "global_orient_w",
                "transl_c",
                "global_orient_c",
                "global_orient_gv",
            )
        }
        targets["frame_times"] = times
        masks = {
            **row["mask"],
            **{
                name: torch.zeros_like(valid)
                for name in ("root_position", "root_orientation", "root_interval")
            },
            "spv_root_motion": False,
        }
        result.append({**row, "mask": masks, "motion_targets": targets, "frame_times": times})
    return result


def collate_video_rows(rows):
    """Batch heterogeneous supervision while preserving per-sequence metadata.

    Optional sparse/root masks receive the original neutral defaults. Dataset
    rows are left untouched so cached records can be reused across epochs.
    """
    rows = _with_optional_root_targets(rows)
    optional_masks = {"label_valid", "kp2d", "imgfeat", "spv_reprojection", "bbox_condition"}
    if any(optional_masks.intersection(row["mask"]) for row in rows):
        has_bbox_condition = any("bbox_condition" in row["mask"] for row in rows)
        rows = [{**row, "mask": dict(row["mask"])} for row in rows]
        for row in rows:
            masks = row["mask"]
            valid = masks["valid"]
            masks.setdefault("label_valid", valid)
            masks.setdefault("kp2d", torch.zeros_like(valid))
            masks.setdefault("imgfeat", torch.ones_like(valid))
            masks.setdefault("spv_reprojection", not masks["spv_incam_only"])
            if has_bbox_condition:
                masks.setdefault("bbox_condition", torch.ones_like(valid))
    batch = {
        name: (
            [row[name] for row in rows]
            if name.startswith("meta")
            else default_collate([row[name] for row in rows])
        )
        for name in rows[0]
    }
    batch["B"] = len(rows)
    batch["training_branch"] = "video"
    return batch
