# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply contact/IK to saved predictions while retaining their raw parameters."""

import math

import torch

from .timing import prediction_fps

PARAMETER_FIELDS = ("pred_soma_params_incam", "pred_soma_params_global")


def select_motion_result(result, variant=None):
    """Return a raw/processed view without modifying the saved result container."""
    if variant is None:
        return result
    processed = bool(result.get("postprocessing"))
    if variant == "postprocessed":
        if not processed:
            raise ValueError("This result has no postprocessed prediction")
        return result
    if variant != "raw":
        raise ValueError("Prediction variant must be raw or postprocessed")
    if not processed:
        return result
    selected = {
        key: value
        for key, value in result.items()
        if key not in ("raw_predictions", "postprocessing")
    }
    return {**selected, **result["raw_predictions"], "postprocessing": False}


def postprocess_motion_result(result, *, soma_assets, device="cpu", processor=None):
    """Start from retained raw predictions, including when replaying this operation.

    Neural conditions and weights are unnecessary. The output's main parameter
    fields are corrected; ``raw_predictions`` preserves original parameters and
    coordinates. Reprocessing a saved result never compounds the correction.
    """
    raw = select_motion_result(result, "raw")
    length = torch.as_tensor(raw["length"]).reshape(-1)
    shape = raw["pred_soma_params_global"]["body_pose"].shape
    if shape[0] != 1 or length.numel() != 1 or int(length[0]) != shape[1]:
        raise ValueError("Postprocess one unpadded Motion sequence at a time")
    if not math.isclose(prediction_fps(raw), 30.0, rel_tol=0, abs_tol=1e-4):
        raise ValueError(
            "Contact/IK requires a continuous 30Hz sequence; omit --postprocess for other clocks"
        )
    if processor is None:
        from gem.motion.postprocess.api import SomaContactPostprocessor

        processor = SomaContactPostprocessor(soma_assets, device=device)
    corrected = processor(
        raw["pred_soma_params_incam"], raw["pred_soma_params_global"], raw["static_conf_logits"]
    )
    preserved = {key: raw[key] for key in PARAMETER_FIELDS}
    if "coordinates" in raw:
        preserved["coordinates"] = raw["coordinates"]
    return {
        **raw,
        **corrected,
        "raw_predictions": preserved,
        "coordinates": {**raw.get("coordinates", {}), "global": "world_meters_after_contact_ik"},
    }
