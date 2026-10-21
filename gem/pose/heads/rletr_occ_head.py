# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLETR occupancy head components for direct 2D joints.

The module deliberately separates three concerns so the same coordinate and
presence readout can consume independently decoded or supplied joint tokens:

* :class:`RLEVisualMemory` projects a ViT feature map to positional memory.
* :class:`RLETransformerTokenProducer` produces one token per 2D joint.
* :class:`RLEOccReadout` maps joint tokens and visual memory to unbounded RLE
  coordinates, uncertainty, and a crop-presence evidence field.

The merged hand-head variant can project its 1024-D joint tokens to ``d_model``
and call ``RLEOccReadout`` directly, while reusing ``RLEVisualMemory`` for the
evidence keys.  There is no mmcv/mmpose dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinear(nn.Module):
    """RLE norm-linear readout: ``x W^T / ||x|| + b``."""

    def __init__(self, in_features: int, out_features: int, *, normalize: bool) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.matmul(self.linear.weight.t())
        if self.normalize:
            eps = torch.finfo(x.dtype).eps
            y = y / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return y + self.linear.bias


class RLEVisualMemory(nn.Module):
    """Project a fixed-resolution ViT feature map to decoder/evidence memory."""

    def __init__(
        self,
        in_channels: int = 1280,
        d_model: int = 256,
        feat_hw: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        if len(feat_hw) != 2 or min(feat_hw) <= 0:
            raise ValueError("feat_hw must contain two positive integers")
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.feat_hw = tuple(int(v) for v in feat_hw)
        self.in_proj = nn.Linear(self.in_channels, self.d_model)
        self.position = nn.Parameter(
            torch.zeros(1, self.feat_hw[0] * self.feat_hw[1], self.d_model)
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B,C,H,W], got {tuple(features.shape)}")
        _, channels, height, width = features.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} feature channels, got {channels}")
        if (height, width) != self.feat_hw:
            raise ValueError(f"expected feature size {self.feat_hw}, got {(height, width)}")
        content = features.flatten(2).transpose(1, 2)
        return self.in_proj(content) + self.position


class RLETransformerTokenProducer(nn.Module):
    """Produce per-joint query tokens from a ViT feature map."""

    def __init__(
        self,
        in_channels: int = 1280,
        num_joints: int = 25,
        d_model: int = 256,
        depth: int = 3,
        n_heads: int = 8,
        feat_hw: tuple[int, int] = (16, 16),
        dropout: float = 0.0,
        visual_memory: RLEVisualMemory | None = None,
    ) -> None:
        super().__init__()
        if num_joints <= 0:
            raise ValueError("num_joints must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_joints = int(num_joints)
        self.d_model = int(d_model)
        if visual_memory is not None and (
            visual_memory.in_channels != int(in_channels)
            or visual_memory.d_model != self.d_model
            or visual_memory.feat_hw != tuple(int(v) for v in feat_hw)
        ):
            raise ValueError(
                "visual_memory dimensions must match in_channels, d_model, " "and feat_hw"
            )
        self.memory = visual_memory or RLEVisualMemory(
            in_channels=in_channels,
            d_model=d_model,
            feat_hw=feat_hw,
        )
        self.query = nn.Embedding(self.num_joints, self.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(self.d_model),
        )

    def build_memory(self, features: torch.Tensor) -> torch.Tensor:
        """Expose memory projection for callers that provide their own tokens."""
        return self.memory(features)

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_memory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        memory = self.build_memory(features)
        queries = self.query.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        tokens = self.decoder(queries, memory)
        if return_memory:
            return tokens, memory
        return tokens


class RLEOccReadout(nn.Module):
    """Shared unbounded-coordinate, uncertainty, and presence readout."""

    def __init__(self, d_model: int = 256, num_joints: int = 25) -> None:
        super().__init__()
        if num_joints <= 0:
            raise ValueError("num_joints must be positive")
        self.d_model = int(d_model)
        self.num_joints = int(num_joints)
        self.coord = NormLinear(self.d_model, 2, normalize=True)
        self.sigma = NormLinear(self.d_model, 2, normalize=False)
        self.map_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.map_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        joint_tokens: torch.Tensor,
        visual_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if joint_tokens.ndim != 3:
            raise ValueError("joint_tokens must have shape [B,K,D]")
        if visual_memory.ndim != 3:
            raise ValueError("visual_memory must have shape [B,N,D]")
        batch_size, num_joints, token_dim = joint_tokens.shape
        if num_joints != self.num_joints or token_dim != self.d_model:
            raise ValueError(
                "joint token shape mismatch: expected "
                f"[B,{self.num_joints},{self.d_model}], got "
                f"{tuple(joint_tokens.shape)}"
            )
        if visual_memory.shape[0] != batch_size or visual_memory.shape[2] != self.d_model:
            raise ValueError(f"visual memory must have shape [B,N,{self.d_model}] with matching B")

        flat_tokens = joint_tokens.reshape(batch_size * num_joints, self.d_model)
        mu = self.coord(flat_tokens).reshape(batch_size, num_joints, 2)
        sigma = torch.sigmoid(self.sigma(flat_tokens).reshape(batch_size, num_joints, 2))

        scores = torch.einsum(
            "bkd,bnd->bkn",
            self.map_q(joint_tokens),
            self.map_k(visual_memory),
        ) / math.sqrt(self.d_model)
        evidence = F.softplus(scores)
        mass = evidence.sum(dim=-1)
        pres_logit = self.alpha * torch.log(mass + 1e-8) + self.beta
        return {
            "mu": mu,
            "sigma": sigma,
            "evidence": evidence,
            "mass": mass,
            "pres_logit": pres_logit,
            "presence": torch.sigmoid(pres_logit),
        }


class RLETransformerOccHead(nn.Module):
    """Clean ``rletr_occ_outcoord`` head without the SimCC auxiliary."""

    def __init__(
        self,
        in_channels: int = 1280,
        num_joints: int = 25,
        d_model: int = 256,
        depth: int = 3,
        n_heads: int = 8,
        feat_hw: tuple[int, int] = (16, 16),
        dropout: float = 0.0,
        visual_memory: RLEVisualMemory | None = None,
        readout: RLEOccReadout | None = None,
    ) -> None:
        super().__init__()
        if readout is not None and (
            readout.d_model != int(d_model) or readout.num_joints != int(num_joints)
        ):
            raise ValueError("readout dimensions must match d_model and num_joints")
        self.token_producer = RLETransformerTokenProducer(
            in_channels=in_channels,
            num_joints=num_joints,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            feat_hw=feat_hw,
            dropout=dropout,
            visual_memory=visual_memory,
        )
        self.readout = readout or RLEOccReadout(
            d_model=d_model,
            num_joints=num_joints,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, memory = self.token_producer(features, return_memory=True)
        return self.readout(tokens, memory)
