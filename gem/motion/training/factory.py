# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the native trainer with the deployed fixed-GS1 SOMA geometry."""

import torch

from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
from gem.motion.codec.soma_native_fixed_gs1 import SomaNativeFixedGs1EnDecoder
from gem.motion.frontend.joint_sets import SOMA29_NOFINGER, get_soma_observation_schema
from gem.motion.model import MotionModel
from gem.motion.network import R3MatchedGEMNetwork
from gem.motion.pipeline import MotionPipeline
from gem.motion.training.losses import ImageT1Loss


def build_motion_model(
    *,
    soma_assets,
    motion_stats,
    landmark_projector,
    observation_schema=SOMA29_NOFINGER,
    device="cuda",
    network_options=None,
    image_loss_weights=None,
    **training_options,
):
    """Build real XLO/MID geometry without importing the reference framework.

    The adapter supports SOMA-X0.3.3 with explicit reference0.2.0, preserving
    the tested0.3.0 compatibility path. ``device='cuda'`` follows the process's
    current device when FK is first used, including Lightning DDP workers.
    Paths identify caller-owned assets and the training normalizer/projector.
    """
    schema = get_soma_observation_schema(observation_schema)
    options = {"obs_num_joints": schema.num_joints, **(network_options or {})}
    network = R3MatchedGEMNetwork(**options)
    xlo = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(data_root=soma_assets, lod="xlo", device=device, apply_correctives=False)
    )
    mid = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(data_root=soma_assets, lod="mid", device=device, apply_correctives=False)
    )
    codec = SomaNativeFixedGs1EnDecoder(stats_path=motion_stats, body_model=xlo)
    projector = torch.load(landmark_projector, map_location="cpu", weights_only=True)["M_soma"]
    image_loss = ImageT1Loss(mid_body=mid, m_soma=projector, weights=image_loss_weights)
    return MotionModel(
        MotionPipeline(network, codec),
        image_loss,
        observation_schema=observation_schema,
        **training_options,
    )
