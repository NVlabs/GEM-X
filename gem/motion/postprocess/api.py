# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SOMA contact correction with the deployed geometry conventions."""

from contextlib import contextmanager

import torch

from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig

from .contact import SOMA_CONTACT_JOINT_IDS, correct_static_pose, correct_static_translation

POSTPROCESS_METHOD = "soma_contact_translation_ccd_ik"
POSTPROCESS_VERSION = "soma-contact-ccd-ieee-v1"


@contextmanager
def ieee_geometry():
    """Keep FK/IK matrix products in IEEE float32 and restore the caller policy."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class PostprocessBody:
    """Expose public77 rig metadata while retaining full-sequence shape handling."""

    def __init__(self, body):
        self.body = body
        self.soma = body.runtime
        self.parents = [
            int(parent) - 1 for parent in self.soma.output_joint_parent_ids.cpu().tolist()
        ][1:]

    def __call__(self, **parameters):
        return self.body(**parameters)


def contact_motion(joints, logits):
    """Foot speeds under the same predicted contact mask, not GT contact metrics."""
    feet = SOMA_CONTACT_JOINT_IDS[:4]
    speed = (joints[:, 1:, feet] - joints[:, :-1, feet]).norm(dim=-1) * 30
    mask = logits[:, :-1, :4] > 0
    selected = speed[mask]
    return {
        "predicted_contact_transition_count": int(mask.sum()),
        "predicted_contact_fraction": float(mask.float().mean()),
        "mean_contact_foot_speed_m_per_s": float(selected.mean()) if selected.numel() else None,
        "median_contact_foot_speed_m_per_s": float(selected.median()) if selected.numel() else None,
        "p95_contact_foot_speed_m_per_s": (
            float(torch.quantile(selected, 0.95)) if selected.numel() else None
        ),
        "all_foot_mean_speed_m_per_s": float(speed.mean()),
        "limitation": "Model-predicted contact mask, not GT contact or a benchmark metric",
    }


class SomaContactPostprocessor:
    """Correct world translation and shared body pose; preserve other parameters.

    Inputs are unpadded ``[B,T,...]`` 30Hz clips. This class does not infer a
    clock; the application boundary checks it before constructing geometry.
    Full sequences go to the adapter, preserving its original shape averaging.
    """

    def __init__(self, soma_assets, *, device="cpu", body_model=None):
        self.device = torch.device(device)
        self.body_model = (
            body_model
            if body_model is not None
            else SomaFixedGs1BodyAdapter(
                SomaRuntimeConfig(
                    data_root=soma_assets,
                    device=device,
                    lod="xlo",
                    mode="warp",
                    apply_correctives=False,
                    enable_procedural_transforms=True,
                    procedural_reference="runtime",
                )
            )
        )

    @torch.inference_mode()
    def __call__(self, params_incam, params_global, static_conf_logits):
        if params_global["body_pose"].shape[1] < 2:
            raise ValueError("Contact postprocessing requires at least two frames")
        body = PostprocessBody(self.body_model)
        camera = {key: value.detach().to(self.device) for key, value in params_incam.items()}
        world = {key: value.detach().to(self.device) for key, value in params_global.items()}
        logits = static_conf_logits.detach().to(self.device)
        with ieee_geometry(), torch.autocast(self.device.type, enabled=False):
            before = body(**world, return_joints_only=True)["joints"].clone()
            corrected = {**world, "transl": correct_static_translation(world, logits, body)}
            corrected["body_pose"] = correct_static_pose(corrected, logits, body)
            camera = {**camera, "body_pose": corrected["body_pose"]}
            after = body(**corrected, return_joints_only=True)["joints"]
        diagnostics = {
            "method": POSTPROCESS_METHOD,
            "version": POSTPROCESS_VERSION,
            "raw": contact_motion(before, logits),
            "postprocessed": contact_motion(after, logits),
            "max_global_translation_change_m": float(
                (corrected["transl"] - world["transl"]).norm(dim=-1).max()
            ),
            "geometry_precision": "IEEE float32 for FK/IK and world mesh; baseline model and overlay policy preserved",
            "contact_logit_order": [
                "left_ankle",
                "left_foot",
                "right_ankle",
                "right_foot",
                "left_wrist",
                "right_wrist",
            ],
        }
        return {
            "pred_soma_params_incam": {key: value.cpu() for key, value in camera.items()},
            "pred_soma_params_global": {key: value.cpu() for key, value in corrected.items()},
            "postprocessing": diagnostics,
        }
