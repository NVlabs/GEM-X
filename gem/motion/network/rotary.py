# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotary positions in reference-frame units, including fractional times."""

import torch
from torch import nn


def get_encoding(d_model, max_seq_len):
    positions = torch.arange(max_seq_len).float()
    frequencies = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    return torch.einsum("i,j->ij", positions, frequencies).repeat_interleave(2, dim=-1)


def apply_rotary_emb(encoding, x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    first, second = pairs.unbind(-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * encoding.cos() + rotated * encoding.sin()


class ROPE(nn.Module):
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer("encoding", get_encoding(d_model, max_seq_len), persistent=False)
        frequencies = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def rotate_queries_or_keys(self, x, positions=None):
        length = x.shape[-2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            if positions is not None:
                positions = positions.to(device=x.device, dtype=torch.float32)
                encoding = (positions[..., None] * self.inv_freq.float()).repeat_interleave(2, -1)
                if positions.ndim == 2:
                    encoding = encoding[:, None]
            elif length > self.max_seq_len:
                encoding = get_encoding(self.d_model, length).to(x)
            else:
                encoding = self.encoding[:length]
            return apply_rotary_emb(encoding, x)
