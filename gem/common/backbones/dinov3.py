# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 feature maps using the Transformers implementation.

Converted weights preserve the legacy backbone's evaluation contract. Training
uses Transformers semantics: one positional augmentation per forward and
Bernoulli stochastic depth. Legacy training instead resamples positions in each
block and uses fixed-size batch subsets for stochastic depth. Consequently this
adapter is not an exact continuation of a historical backbone training run.
"""

import math
from collections.abc import Mapping
from os import PathLike

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    ALL_ATTENTION_FUNCTIONS,
    DINOv3ViTAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class _BF16DetectorRope(nn.Module):
    """Preserve the detector's BF16 coordinate, division and trig rounding.

    This inference compatibility mode deliberately differs from HF's FP32
    frequency multiplication. Periods must remain periods until the division.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        head_dim = config.hidden_size // config.num_attention_heads
        periods = config.rope_theta ** (
            2 * torch.arange(head_dim // 4, dtype=torch.bfloat16) / (head_dim // 2)
        )
        self.register_buffer("periods", periods)

    def forward(self, pixel_values):
        height, width = (size // self.patch_size for size in pixel_values.shape[-2:])
        kwargs = {"device": pixel_values.device, "dtype": torch.bfloat16}
        y = torch.arange(0.5, height, **kwargs) / height
        x = torch.arange(0.5, width, **kwargs) / width
        coords = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods.to(torch.bfloat16)[None, None]
        angles = angles.flatten(1, 2).tile(2)
        return angles.cos(), angles.sin()


class _BF16DetectorAttention(DINOv3ViTAttention):
    """HF attention with the detector's Q/K BF16 rotary roundtrip.

    Reuse HF's projection modules and attention backend. Unlike HF's default,
    the prefix tokens also undergo the BF16 roundtrip; values remain FP32.
    """

    def __init__(self, attention):
        nn.Module.__init__(self)
        for name in (
            "config",
            "embed_dim",
            "num_heads",
            "head_dim",
            "is_causal",
            "scaling",
            "dropout",
        ):
            setattr(self, name, getattr(attention, name))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(attention, name))

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        batch, patches, _ = hidden_states.shape
        # A packed projection is numerically significant here: separate GEMMs
        # can fall on opposite sides of a subsequent BF16 rounding boundary.
        # Keep HF's parameter layout, but preserve the detector's GEMM shape.
        weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
        bias = torch.cat((self.q_proj.bias, torch.zeros_like(self.q_proj.bias), self.v_proj.bias))
        qkv = F.linear(hidden_states, weight, bias)
        query, key, value = qkv.reshape(batch, patches, 3, self.num_heads, self.head_dim).unbind(2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        query, key = apply_rotary_pos_emb(
            query.to(torch.bfloat16), key.to(torch.bfloat16), *position_embeddings
        )
        attention = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention(
            self,
            query.to(value.dtype),
            key.to(value.dtype),
            value,
            attention_mask,
            dropout=self.dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        return self.o_proj(output.reshape(batch, patches, -1).contiguous()), weights


def dinov3_vith16plus_config(drop_path_rate: float = 0.0) -> DINOv3ViTConfig:
    """Build the ViT-H/16+ architecture without downloading a model config."""
    return DINOv3ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=5120,
        use_gated_mlp=True,
        hidden_act="silu",
        num_register_tokens=4,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layerscale_value=1e-5,
        layer_norm_eps=1e-5,
        pos_embed_rescale=2.0,
        drop_path_rate=drop_path_rate,
        _attn_implementation="sdpa",
    )


def convert_dinov3_native_state_dict(
    state_dict: Mapping[str, Tensor], *, prefix: str = "", rotary_precision: str = "hf"
) -> dict[str, Tensor]:
    """Convert a native encoder to Transformers keys, without changing inputs.

    ``prefix`` selects the encoder inside a larger checkpoint, for example
    ``backbone.encoder.`` in a SAM-3D-Body or legacy GEM-Pose checkpoint.
    The target uses the ViT-H/16+ conventions: a masked key bias, shared final
    LayerNorm. Saved RoPE periods are retained: pretrained checkpoints can
    contain rounded periods which differ from freshly constructed frequencies.
    Both separate ``w1/w2/w3`` and fused ``w12/w3`` SwiGLU layouts are accepted.
    Unknown keys remain errors instead of silently leaving random parameters.
    ``rotary_precision='bf16'`` keeps native periods for the image detector;
    it must match the option used to construct the target backbone.
    """
    source = {
        key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
    }
    if not source:
        raise ValueError(f"No DINOv3 parameters found with prefix {prefix!r}")

    converted = {}
    embedding_keys = {
        "cls_token": "embeddings.cls_token",
        "storage_tokens": "embeddings.register_tokens",
        "mask_token": "embeddings.mask_token",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
    }
    block_keys = {
        "attn.proj": "attention.o_proj",
        "ls1.gamma": "layer_scale1.lambda1",
        "ls2.gamma": "layer_scale2.lambda1",
        "mlp.w1": "mlp.gate_proj",
        "mlp.w2": "mlp.up_proj",
        "mlp.w3": "mlp.down_proj",
        "mlp.fc1": "mlp.up_proj",
        "mlp.fc2": "mlp.down_proj",
    }
    for key, value in source.items():
        if key.endswith(".attn.qkv.bias_mask"):
            continue
        if key == "rope_embed.periods":
            if rotary_precision == "bf16":
                converted["rope_embeddings.periods"] = value.to(torch.bfloat16)
            else:
                converted["rope_embeddings.inv_freq"] = value.reciprocal()
        elif key in embedding_keys:
            converted[embedding_keys[key]] = value.unsqueeze(1) if key == "mask_token" else value
        elif key in ("norm.weight", "norm.bias"):
            converted[key] = value
        elif key.startswith("blocks."):
            _, block_index, suffix = key.split(".", 2)
            target = f"layer.{block_index}."
            if suffix in ("attn.qkv.weight", "attn.qkv.bias"):
                parameter = suffix.rsplit(".", 1)[1]
                if parameter == "bias":
                    bias_mask = source.get(f"blocks.{block_index}.attn.qkv.bias_mask")
                    if bias_mask is not None:
                        value = value * bias_mask.to(value.dtype)
                query, key_value, value_value = value.chunk(3, dim=0)
                converted[f"{target}attention.q_proj.{parameter}"] = query
                converted[f"{target}attention.v_proj.{parameter}"] = value_value
                if parameter == "weight":
                    converted[f"{target}attention.k_proj.weight"] = key_value
                elif torch.count_nonzero(key_value).item():
                    raise ValueError(
                        f"{key} contains an unmasked key bias; the target has key_bias=False"
                    )
            elif suffix in ("mlp.w12.weight", "mlp.w12.bias"):
                parameter = suffix.rsplit(".", 1)[1]
                gate, up = value.chunk(2, dim=0)
                converted[f"{target}mlp.gate_proj.{parameter}"] = gate
                converted[f"{target}mlp.up_proj.{parameter}"] = up
            elif suffix.startswith(("norm1.", "norm2.")):
                converted[target + suffix] = value
            else:
                component, parameter = suffix.rsplit(".", 1)
                if suffix in block_keys:
                    converted[target + block_keys[suffix]] = value
                elif component in block_keys:
                    converted[f"{target}{block_keys[component]}.{parameter}"] = value
                else:
                    raise ValueError(f"Unsupported DINOv3 parameter: {key}")
        else:
            raise ValueError(f"Unsupported DINOv3 parameter: {key}")
    return converted


class DinoV3Backbone(nn.Module):
    """Return normalized patch features with shape ``(B, C, H/16, W/16)``.

    Inputs are already cropped and normalized RGB tensors. The architecture
    preset and checkpoints require image dimensions divisible by patch size.
    ``config`` also permits small models for local numerical checks.

    Training is supported with Transformers' augmentation/stochastic-depth
    semantics described in the module docstring. The default stochastic-depth
    rate is zero; a new training recipe must choose its rate explicitly.
    ``rotary_precision='bf16'`` is the native77 detector compatibility mode:
    BF16 rotary coordinates, Q/K roundtrip, and a packed QKV projection. It
    requires masked key bias and disabled positional augmentation. The default
    preserves the image-pose backbone's standard HF arithmetic.
    """

    def __init__(
        self,
        name: str = "dinov3_vith16plus",
        drop_path_rate: float = 0.0,
        *,
        config: DINOv3ViTConfig | None = None,
        gradient_checkpointing: bool = False,
        rotary_precision: str = "hf",
    ):
        super().__init__()
        if name != "dinov3_vith16plus":
            raise ValueError(f"Unsupported DINOv3 architecture: {name!r}")
        if config is None:
            config = dinov3_vith16plus_config(drop_path_rate)
        if rotary_precision not in ("hf", "bf16"):
            raise ValueError("rotary_precision must be 'hf' or 'bf16'")
        self.rotary_precision = rotary_precision
        self.encoder = DINOv3ViTModel(config)
        # Legacy checkpoints may carry rounded RoPE periods. Keep their inverse
        # frequencies in the state dict instead of regenerating ideal periods.
        if rotary_precision == "bf16":
            if config.key_bias or not config.query_bias or not config.value_bias:
                raise ValueError("Detector BF16 rotary mode requires Q/V bias and no key bias")
            if any(
                getattr(config, name) is not None
                for name in ("pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale")
            ):
                raise ValueError(
                    "Detector BF16 rotary mode requires positional augmentation disabled"
                )
            self.encoder.rope_embeddings = _BF16DetectorRope(config)
            for layer in self.encoder.layer:
                layer.attention = _BF16DetectorAttention(layer.attention)
        else:
            rope = self.encoder.rope_embeddings
            rope.register_buffer("inv_freq", rope.inv_freq, persistent=True)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.name = name
        self.patch_size = config.patch_size
        self.embed_dim = self.embed_dims = config.hidden_size
        self.num_register_tokens = config.num_register_tokens

    def forward(self, images: Tensor) -> Tensor:
        batch_size, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Image dimensions must be divisible by patch size {self.patch_size}")
        output = self.encoder(pixel_values=images, return_dict=True).last_hidden_state
        patches = output[:, 1 + self.num_register_tokens :]
        return (
            patches.transpose(1, 2)
            .reshape(
                batch_size, self.embed_dim, height // self.patch_size, width // self.patch_size
            )
            .contiguous()
        )

    def load_pretrained(
        self, checkpoint_path: str | PathLike[str], *, prefix: str = "backbone.encoder."
    ) -> None:
        """Initialize image features, allowing SAM-3D-Body's omitted mask token."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        converted = convert_dinov3_native_state_dict(
            state_dict, prefix=prefix, rotary_precision=self.rotary_precision
        )
        if "embeddings.mask_token" not in converted:
            # SAM-3D-Body omits this SSL-only token. Our RGB forward never
            # supplies bool_masked_pos; retain HF's zero initialization without
            # relaxing the required image-feature weights or consuming RNG.
            converted["embeddings.mask_token"] = torch.zeros_like(
                self.encoder.embeddings.mask_token
            )
        self.encoder.load_state_dict(converted, strict=True)
