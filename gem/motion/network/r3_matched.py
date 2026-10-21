# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native 655D GEM regression network with the deployed parameter layout."""

import math

import torch
from torch import nn

from gem.motion.frontend.observation import encode_observations

from .attention import EncoderRoPEBlock
from .mlp import Mlp


def _zero_projection(layer):
    for parameter in layer.parameters():
        parameter.detach().zero_()
    return layer


class PositionalEncoding(nn.Module):
    def __init__(self, width, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, width)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies)
        self.pe = nn.Parameter(pe[:, None], requires_grad=False)


class TimestepEmbedder(nn.Module):
    def __init__(self, width, sequence_pos_encoder):
        super().__init__()
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class TemporalRegressionCore(nn.Module):
    """The trained GEM core, specialized to direct identity/log-scale output.

    The fixed timestep embedding and zero-motion projection remain learned
    parts of regression checkpoints. No diffusion sampler is involved.
    """

    def __init__(
        self,
        output_dim=655,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.1,
        avgbeta=True,
    ):
        super().__init__()
        self.max_len = max_len
        self.output_dim = output_dim
        self.avgbeta = avgbeta
        self.blocks = nn.ModuleList(
            [EncoderRoPEBlock(latent_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.sequence_pos_encoder = PositionalEncoding(latent_dim)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.final_layer = Mlp(latent_dim, out_features=output_dim)
        self.pred_cam_head = Mlp(latent_dim, out_features=pred_cam_dim)
        self.register_buffer(
            "pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), persistent=False
        )
        self.register_buffer(
            "pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), persistent=False
        )
        self.static_conf_head = Mlp(latent_dim, out_features=static_conf_dim)
        self.add_cond_linear = nn.Linear(output_dim + latent_dim, latent_dim)
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            _zero_projection(nn.Linear(latent_dim, latent_dim)),
        )
        if cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                _zero_projection(nn.Linear(latent_dim, latent_dim)),
            )
        if imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(imgseq_dim),
                _zero_projection(nn.Linear(imgseq_dim, latent_dim)),
            )

    def forward(self, condition, length, positions=None):
        batch_size, frames, _ = condition.shape
        timesteps = torch.full((batch_size,), 999, device=condition.device, dtype=torch.long)
        x = condition + self.embed_timestep(timesteps)
        x = self.add_cond_linear(
            torch.cat((x, x.new_zeros(batch_size, frames, self.output_dim)), dim=-1)
        )
        padding = torch.arange(frames, device=length.device)[None] >= length[:, None]
        attention_mask = None
        if frames > self.max_len:
            attention_mask = torch.ones(frames, frames, device=x.device, dtype=torch.bool)
            for index in range(frames):
                start = min(frames - self.max_len, max(0, index - self.max_len // 2))
                stop = max(self.max_len, min(frames, index + self.max_len // 2))
                attention_mask[index, start:stop] = False
        for block in self.blocks:
            x = block(
                x, attn_mask=attention_mask, tgt_key_padding_mask=padding, positions=positions
            )

        sample = self.final_layer(x)
        if self.avgbeta:
            valid = ~padding[..., None]
            identity = (sample[..., 456:584] * valid).sum(1) / length[:, None]
            log_scale = (sample[..., 584:640] * valid).sum(1) / length[:, None]
            sample = torch.cat(
                (
                    sample[..., :456],
                    identity[:, None].expand(-1, frames, -1),
                    log_scale[:, None].expand(-1, frames, -1),
                    sample[..., 640:],
                ),
                dim=-1,
            )
        camera = self.pred_cam_head(x) * self.pred_cam_std + self.pred_cam_mean
        torch.clamp_min_(camera[..., 0], 0.25)
        return {
            "pred_x": sample,
            "pred_cam": camera,
            "static_conf_logits": self.static_conf_head(x),
            "pred_context": x,
        }


class R3MatchedGEMNetwork(nn.Module):
    """Observation frontend plus temporal core; preserves existing weight keys.

    Historical checkpoints use compact30; native models use SOMA29 or SOMA77.
    Each width requires its own trained observation embedding weights.
    """

    def __init__(
        self,
        output_dim=655,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.1,
        avgbeta=True,
        obs_num_joints=30,
        sequence_static_slices=((456, 584), (584, 640)),
    ):
        super().__init__()
        if output_dim != 655 or obs_num_joints not in (29, 30, 77):
            raise ValueError(
                "This GEM-Motion core requires 655D output and 29, 30, or 77 observations"
            )
        if tuple(map(tuple, sequence_static_slices)) != ((456, 584), (584, 640)):
            raise ValueError("Static fields must be identity456:584 and log-local584:640")
        if pred_cam_dim != 3 or static_conf_dim != 6:
            raise ValueError("The body motion core requires camera3 and contact6 heads")
        self.output_dim = output_dim
        self.obs_num_joints = obs_num_joints
        self.latent_dim = latent_dim
        self.sequence_static_slices = (slice(456, 584), slice(584, 640))
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(obs_num_joints, 32))
        self.embed_noisyobs = Mlp(
            obs_num_joints * 32,
            hidden_features=latent_dim * 2,
            out_features=latent_dim,
            drop=dropout,
        )
        self.denoiser = TemporalRegressionCore(
            output_dim=output_dim,
            max_len=max_len,
            cliffcam_dim=cliffcam_dim,
            cam_angvel_dim=cam_angvel_dim,
            imgseq_dim=imgseq_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pred_cam_dim=pred_cam_dim,
            static_conf_dim=static_conf_dim,
            dropout=dropout,
            avgbeta=avgbeta,
        )

    def forward(
        self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None, positions=None
    ):
        if obs.shape[-2:] != (self.obs_num_joints, 3):
            raise ValueError(f"Expected observations shaped (B, L, {self.obs_num_joints}, 3)")
        encoded = encode_observations(obs, self.learned_pos_linear, self.learned_pos_params)
        condition = self.embed_noisyobs(encoded) + self.denoiser.cliffcam_embedder(f_cliffcam)
        if hasattr(self.denoiser, "cam_angvel_embedder"):
            condition = condition + self.denoiser.cam_angvel_embedder(f_cam_angvel)
        if f_imgseq is not None and hasattr(self.denoiser, "imgseq_embedder"):
            condition = condition + self.denoiser.imgseq_embedder(f_imgseq)
        return self.denoiser(condition, length, positions)
