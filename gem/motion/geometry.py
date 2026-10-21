# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked SOMA geometry from raw Motion camera or world parameters."""

from dataclasses import replace

import torch
from torch import nn

from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig


class MotionGeometry(nn.Module):
    """Evaluate cached axis-angle Motion parameters without modifying them.

    Pose/translation use ``[B,T,...]``; identity128 and packed GS1+local56 may
    use ``[B,T,D]`` or sequence-static ``[B,D]``. Each frame's shape is retained:
    the temporal network, not this visualization boundary, owns shape averaging.

    MID geometry, correctives and the native Motion procedural convention match
    the deployed renderer. ``apply_correctives=False`` selects the training FK
    surface. Both paths use SOMA's reference0.2.0 with its0.3.3 runtime; neither
    uses Pose's legacy twist-binding profile. Translation is applied once by
    SOMA. Camera/world selection is entirely determined by the supplied params.

    Iteration transfers only one chunk to the geometry device and returns CPU
    tensors. ``forward`` materializes the full sequence on CPU for convenience.
    """

    def __init__(
        self,
        soma_assets=None,
        *,
        device="cpu",
        chunk_size=32,
        lod="mid",
        apply_correctives=True,
        body_model=None,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("Motion geometry chunk_size must be positive")
        self.chunk_size = chunk_size
        self.apply_correctives = apply_correctives
        self.register_buffer("_device_anchor", torch.empty(0, device=device), persistent=False)
        self.body_model = (
            body_model
            if body_model is not None
            else SomaFixedGs1BodyAdapter(
                SomaRuntimeConfig(
                    data_root=soma_assets,
                    device=device,
                    lod=lod,
                    apply_correctives=apply_correctives,
                    enable_procedural_transforms=True,
                    procedural_reference="runtime",
                )
            )
        )

    @property
    def device(self):
        return self._device_anchor.device

    def _apply(self, fn, recurse=True):
        self._device_anchor = fn(self._device_anchor)
        self.body_model.to(device=self.device, dtype=torch.float32)
        self.body_model.runtime_config = replace(self.body_model.runtime_config, device=self.device)
        return self

    @property
    def faces(self):
        """Shared triangle indices on CPU; initializes the runtime if needed."""
        return self.body_model.runtime.faces.detach().cpu()

    def iter_chunks(self, parameters):
        """Yield ``(start, {vertices:CPU[B,C,V,3], joints:CPU[B,C,77,3]})``.

        Flattening each pose chunk to independent frames preserves the original
        rendering adapter's per-frame shape behavior, including avgbeta=False.
        """
        root = parameters["global_orient"]
        if root.ndim != 3 or root.shape[-1] != 3:
            raise ValueError("Motion geometry requires global_orient shaped (B,T,3)")
        batch, length = root.shape[:2]
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            count = stop - start
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                values = {}
                for name in ("body_pose", "global_orient", "transl"):
                    value = parameters[name][:, start:stop]
                    values[name] = value.reshape(batch * count, *value.shape[2:]).to(
                        device=self.device, dtype=torch.float32
                    )
                for name in ("identity_coeffs", "scale_params"):
                    value = parameters[name]
                    value = (
                        value[:, None].expand(-1, count, -1)
                        if value.ndim == 2
                        else value[:, start:stop]
                    )
                    values[name] = value.reshape(batch * count, -1).to(
                        device=self.device, dtype=torch.float32
                    )
                output = self.body_model(
                    **values,
                    pose2rot=True,
                    apply_correctives=self.apply_correctives,
                    repose_to_bind_pose=True,
                )
                chunk = {
                    name: output[name].reshape(batch, count, -1, 3).detach().cpu()
                    for name in ("vertices", "joints")
                }
                del output, values
            yield start, chunk

    def forward(self, parameters):
        chunks = [chunk for _, chunk in self.iter_chunks(parameters)]
        return {
            name: torch.cat([chunk[name] for chunk in chunks], dim=1)
            for name in ("vertices", "joints")
        } | {"faces": self.faces}
