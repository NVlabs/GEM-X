# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated temporal self-attention used by GEM-Motion."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mlp import Mlp
from .rotary import ROPE


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = ROPE(self.head_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None, positions=None):
        batch_size, length, _ = x.shape
        query = self.query(x).reshape(batch_size, length, self.num_heads, -1).transpose(1, 2)
        key = self.key(x).reshape(batch_size, length, self.num_heads, -1).transpose(1, 2)
        value = self.value(x).reshape(batch_size, length, self.num_heads, -1).transpose(1, 2)
        # Rotary trigonometry uses FP32; SDPA needs a common Q/K/V dtype.
        query = self.rope.rotate_queries_or_keys(query, positions).to(value.dtype)
        key = self.rope.rotate_queries_or_keys(key, positions).to(value.dtype)
        blocked = None
        if attn_mask is not None:
            blocked = attn_mask.reshape(-1, 1, length, length)
        if key_padding_mask is not None:
            padding = key_padding_mask[:, None, None]
            blocked = padding if blocked is None else blocked | padding
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None if blocked is None else ~blocked,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return self.proj(output.transpose(1, 2).reshape(batch_size, length, -1))


class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None, positions=None):
        x = x + self.gate_msa * self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            positions=positions,
        )
        return x + self.gate_mlp * self.mlp(self.norm2(x))
