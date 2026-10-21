# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Losses for the clean RLETR occupancy out-of-crop coordinate head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flow_scale_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 2),
        nn.Tanh(),
    )


def _flow_translation_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 2),
    )


class RealNVP(nn.Module):
    """Six-coupling 2D RealNVP used by the original RLE implementation."""

    def __init__(self) -> None:
        super().__init__()
        masks = torch.tensor([[0.0, 1.0], [1.0, 0.0]] * 3)
        self.register_buffer("mask", masks)
        self.translation = nn.ModuleList([_flow_translation_net() for _ in range(len(masks))])
        self.scale = nn.ModuleList([_flow_scale_net() for _ in range(len(masks))])
        for module in list(self.translation) + list(self.scale):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)

    def log_prob(self, residual: torch.Tensor) -> torch.Tensor:
        log_det_jacobian = residual.new_zeros(residual.shape[0])
        latent = residual
        for index in reversed(range(len(self.translation))):
            masked = self.mask[index] * latent
            scale = self.scale[index](masked) * (1.0 - self.mask[index])
            translation = self.translation[index](masked) * (1.0 - self.mask[index])
            latent = (1.0 - self.mask[index]) * (latent - translation) * torch.exp(-scale) + masked
            log_det_jacobian -= scale.sum(dim=1)
        base_log_prob = -0.5 * latent.square().sum(dim=1) - math.log(2.0 * math.pi)
        return base_log_prob + log_det_jacobian


def marginal_crps_map(
    probability: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    normalized: bool = False,
) -> torch.Tensor:
    """Sum x/y marginal CRPS for maps ``[B,H,W]`` and grid targets ``[B,2]``."""
    _, height, width = probability.shape
    result = probability.new_zeros(probability.shape[0])
    for spatial_axis, coord_axis, size in ((2, 0, width), (1, 1, height)):
        if normalized and size == 1:
            continue
        marginal = probability.sum(dim=3 - spatial_axis)
        support = torch.arange(
            size,
            device=probability.device,
            dtype=probability.dtype,
        )
        expected_distance = (marginal * (support[None] - target_xy[:, coord_axis, None]).abs()).sum(
            dim=1
        )
        distance = (support[:, None] - support[None, :]).abs()
        expected_pairwise = torch.einsum(
            "bi,ij,bj->b",
            marginal,
            distance,
            marginal,
        )
        axis_crps = expected_distance - 0.5 * expected_pairwise
        if normalized:
            axis_crps = axis_crps / (size - 1)
        result = result + axis_crps
    return result


class RLEOccOutCoordLoss(nn.Module):
    """Raw clean-RLETR losses with a direct crop-coordinate target contract.

    Args:
        feat_hw: Spatial shape of the evidence memory, as ``(H, W)``.
        crop_bounds: Half-open direct-coordinate support ``[lower, upper)``.
            Callers should pass ``inside_crop_mask`` explicitly when their
            exact crop/pixel-center convention differs.
        crps_normalize: Divide each marginal CRPS by its grid span.

    ``forward`` returns three *unweighted* components:

    * ``rle_loss``: official RLE sum over axes/joints divided by batch size;
      every labeled joint is supervised, including crop-external joints.
    * ``presence_loss``: BCE mean over labeled joints.
    * ``evidence_crps_loss``: marginal-CRPS mean over labeled crop-present
      joints only.

    Global task coefficients live in ``HandPoseObjective``.
    """

    _amplitude = 1.0 / math.sqrt(2.0 * math.pi)

    def __init__(
        self,
        feat_hw: tuple[int, int] = (16, 16),
        crop_bounds: tuple[float, float] = (-0.5, 0.5),
        crps_normalize: bool = False,
        negative_presence_weight: float | None = None,
        sigma_min: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(feat_hw) != 2 or min(feat_hw) <= 0:
            raise ValueError("feat_hw must contain two positive integers")
        lower, upper = crop_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("crop_bounds must contain finite increasing values")
        if negative_presence_weight is not None and negative_presence_weight < 0:
            raise ValueError("negative_presence_weight must be non-negative")
        if sigma_min <= 0:
            raise ValueError("sigma_min must be positive")
        self.feat_hw = tuple(int(v) for v in feat_hw)
        self.crop_bounds = (float(lower), float(upper))
        self.crps_normalize = bool(crps_normalize)
        self.negative_presence_weight = negative_presence_weight
        self.sigma_min = float(sigma_min)
        self.flow = RealNVP()

    @staticmethod
    def _check_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target_coords: torch.Tensor,
        labeled_mask: torch.Tensor,
        inside_crop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        required = {"mu", "sigma", "evidence", "mass", "pres_logit"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"RLE output is missing keys: {sorted(missing)}")

        mu = output["mu"]
        sigma = output["sigma"]
        if mu.ndim != 3 or mu.shape[-1] != 2:
            raise ValueError("mu must have shape [B,K,2]")
        batch_size, num_joints, _ = mu.shape
        self._check_shape("sigma", sigma, mu.shape)
        self._check_shape(
            "target_coords",
            target_coords,
            (batch_size, num_joints, 2),
        )
        self._check_shape(
            "labeled_mask",
            labeled_mask,
            (batch_size, num_joints),
        )
        self._check_shape(
            "pres_logit",
            output["pres_logit"],
            (batch_size, num_joints),
        )
        self._check_shape(
            "mass",
            output["mass"],
            (batch_size, num_joints),
        )
        expected_tokens = self.feat_hw[0] * self.feat_hw[1]
        self._check_shape(
            "evidence",
            output["evidence"],
            (batch_size, num_joints, expected_tokens),
        )

        if target_coords.device != mu.device or labeled_mask.device != mu.device:
            raise ValueError("targets and masks must be on the same device as output")
        # Some production camera annotations can yield a non-finite projected
        # coordinate even when the source joint confidence is positive.  A
        # post-hoc zero weight is not sufficient here: RLE and CRPS both do
        # arithmetic with every target before applying their masks, so
        # ``0 * NaN`` would still poison the batch.  Exclude such targets and
        # replace them with an inert finite value before any arithmetic.
        target_finite = torch.isfinite(target_coords).all(dim=-1)
        labeled_mask = labeled_mask.to(dtype=torch.bool) & target_finite
        labeled = labeled_mask.to(dtype=mu.dtype)
        safe_target_coords = torch.where(
            target_finite.unsqueeze(-1),
            target_coords,
            torch.zeros_like(target_coords),
        )
        if inside_crop_mask is None:
            lower, upper = self.crop_bounds
            inside_crop_mask = (
                target_finite
                & (safe_target_coords[..., 0] >= lower)
                & (safe_target_coords[..., 0] < upper)
                & (safe_target_coords[..., 1] >= lower)
                & (safe_target_coords[..., 1] < upper)
            )
        else:
            self._check_shape(
                "inside_crop_mask",
                inside_crop_mask,
                (batch_size, num_joints),
            )
            if inside_crop_mask.device != mu.device:
                raise ValueError("inside_crop_mask must be on the same device as output")
            inside_crop_mask = inside_crop_mask.to(dtype=torch.bool) & target_finite
        inside = inside_crop_mask.to(dtype=mu.dtype)

        flow_dtype = next(self.flow.parameters()).dtype
        mu_work = mu.to(dtype=flow_dtype)
        sigma_work = sigma.to(dtype=flow_dtype).clamp_min(self.sigma_min)
        target_work = safe_target_coords.to(dtype=flow_dtype)
        coord_weight = labeled.to(dtype=flow_dtype).unsqueeze(-1)
        standardized = (mu_work - target_work) / sigma_work
        log_phi = self.flow.log_prob(standardized.reshape(-1, 2)).reshape(
            batch_size,
            num_joints,
            1,
        )
        normalizing_flow = (torch.log(sigma_work) - log_phi) * coord_weight
        log_q = (
            torch.log(sigma_work / self._amplitude)
            + (mu_work - target_work).abs() / (math.sqrt(2.0) * sigma_work + 1e-9)
        ) * coord_weight
        rle_loss = (normalizing_flow + log_q).sum() / max(batch_size, 1)

        presence_target = inside.to(dtype=output["pres_logit"].dtype)
        presence_bce = F.binary_cross_entropy_with_logits(
            output["pres_logit"],
            presence_target,
            reduction="none",
        )
        presence_weight = labeled.to(dtype=presence_bce.dtype)
        if self.negative_presence_weight is not None:
            balance = torch.where(
                presence_target > 0.5,
                torch.ones_like(presence_target),
                torch.full_like(
                    presence_target,
                    self.negative_presence_weight,
                ),
            )
            presence_weight = presence_weight * balance
        presence_loss = (presence_bce * presence_weight).sum() / presence_weight.sum().clamp_min(
            1.0
        )

        height, width = self.feat_hw
        evidence = output["evidence"].reshape(
            batch_size * num_joints,
            height,
            width,
        )
        probability = evidence / (output["mass"].reshape(batch_size * num_joints, 1, 1) + 1e-8)
        target_grid = torch.stack(
            [
                (safe_target_coords[..., 0] + 0.5) * (width - 1.0),
                (safe_target_coords[..., 1] + 0.5) * (height - 1.0),
            ],
            dim=-1,
        ).reshape(batch_size * num_joints, 2)
        crps = marginal_crps_map(
            probability,
            target_grid.to(dtype=probability.dtype),
            normalized=self.crps_normalize,
        ).reshape(batch_size, num_joints)
        crps_weight = labeled.to(dtype=crps.dtype) * inside.to(dtype=crps.dtype)
        evidence_crps_loss = (crps * crps_weight).sum() / crps_weight.sum().clamp_min(1.0)

        return {
            "rle_loss": rle_loss,
            "presence_loss": presence_loss,
            "evidence_crps_loss": evidence_crps_loss,
        }
