# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM-3D-Body image initialization can omit only the unused mask token."""

import pytest
import torch
from transformers import DINOv3ViTConfig

from gem.common.backbones.dinov3 import DinoV3Backbone


def backbone():
    return DinoV3Backbone(
        config=DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            num_register_tokens=2,
            use_gated_mlp=True,
            hidden_act="silu",
            key_bias=False,
            layerscale_value=0.5,
            layer_norm_eps=1e-5,
            pos_embed_rescale=None,
        )
    )


def native_checkpoint(model, *, include_mask):
    """Export a tiny complete native layout, as used by the SAM3DB initializer."""
    encoder = model.encoder
    state = {
        "cls_token": encoder.embeddings.cls_token.detach().clone(),
        "storage_tokens": encoder.embeddings.register_tokens.detach().clone(),
        "patch_embed.proj.weight": encoder.embeddings.patch_embeddings.weight.detach().clone(),
        "patch_embed.proj.bias": encoder.embeddings.patch_embeddings.bias.detach().clone(),
        "rope_embed.periods": encoder.rope_embeddings.inv_freq.detach().reciprocal(),
        "norm.weight": encoder.norm.weight.detach().clone(),
        "norm.bias": encoder.norm.bias.detach().clone(),
    }
    if include_mask:
        state["mask_token"] = encoder.embeddings.mask_token.detach().squeeze(1).clone()
    for index, layer in enumerate(encoder.layer):
        block = {
            "attn.qkv.weight": torch.cat(
                [
                    layer.attention.q_proj.weight,
                    layer.attention.k_proj.weight,
                    layer.attention.v_proj.weight,
                ]
            ),
            "attn.qkv.bias": torch.cat(
                [
                    layer.attention.q_proj.bias,
                    torch.zeros_like(layer.attention.q_proj.bias),
                    layer.attention.v_proj.bias,
                ]
            ),
            "attn.proj.weight": layer.attention.o_proj.weight,
            "attn.proj.bias": layer.attention.o_proj.bias,
            "ls1.gamma": layer.layer_scale1.lambda1,
            "ls2.gamma": layer.layer_scale2.lambda1,
        }
        for name in ("norm1", "norm2"):
            block[name + ".weight"] = getattr(layer, name).weight
            block[name + ".bias"] = getattr(layer, name).bias
        for old, new in (("w1", "gate_proj"), ("w2", "up_proj"), ("w3", "down_proj")):
            block[f"mlp.{old}.weight"] = getattr(layer.mlp, new).weight
            block[f"mlp.{old}.bias"] = getattr(layer.mlp, new).bias
        state.update(
            {f"blocks.{index}.{name}": value.detach().clone() for name, value in block.items()}
        )
    return {"state_dict": {"backbone.encoder." + name: value for name, value in state.items()}}


@pytest.mark.parametrize("include_mask", [False, True])
def test_pretrained_mask_policy_preserves_features_and_trainable_gradients(tmp_path, include_mask):
    torch.manual_seed(13)
    reference = backbone().eval()
    with torch.no_grad():
        reference.encoder.embeddings.mask_token.fill_(0.75)
    checkpoint = native_checkpoint(reference, include_mask=include_mask)
    path = tmp_path / "sam3db_tiny.pt"
    torch.save(checkpoint, path)
    actual = backbone().eval()
    with torch.no_grad():
        actual.encoder.embeddings.mask_token.fill_(3)
    rng_before = torch.get_rng_state().clone()
    actual.load_pretrained(path)
    assert torch.equal(torch.get_rng_state(), rng_before)
    expected_token = 0.75 if include_mask else 0.0
    torch.testing.assert_close(
        actual.encoder.embeddings.mask_token,
        torch.full_like(actual.encoder.embeddings.mask_token, expected_token),
        rtol=0,
        atol=0,
    )
    pixels = torch.randn(2, 3, 32, 48)
    with torch.no_grad():
        expected = reference(pixels)
        loaded = actual(pixels)
    torch.testing.assert_close(loaded, expected, rtol=0, atol=0)
    actual.train()
    inputs = pixels.clone().requires_grad_()
    coefficients = torch.randn_like(loaded)
    (actual(inputs) * coefficients).mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradient = actual.encoder.embeddings.patch_embeddings.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert actual.encoder.embeddings.mask_token.grad is None


@pytest.mark.parametrize("failure", ["missing_patch", "missing_qkv", "unexpected", "wrong_shape"])
def test_pretrained_mask_exception_does_not_relax_other_weights(tmp_path, failure):
    model = backbone()
    checkpoint = native_checkpoint(model, include_mask=False)
    state = checkpoint["state_dict"]
    expected_error = RuntimeError
    if failure == "missing_patch":
        del state["backbone.encoder.patch_embed.proj.weight"]
        message = "patch_embeddings.weight"
    elif failure == "missing_qkv":
        del state["backbone.encoder.blocks.0.attn.qkv.weight"]
        message = "attention.q_proj.weight"
    elif failure == "unexpected":
        state["backbone.encoder.unexpected_trainable_weight"] = torch.ones(2)
        expected_error, message = ValueError, "Unsupported DINOv3 parameter"
    else:
        state["backbone.encoder.patch_embed.proj.weight"] = torch.zeros(1, 3, 16, 16)
        message = "size mismatch"
    path = tmp_path / "incomplete.pt"
    torch.save(checkpoint, path)
    with pytest.raises(expected_error, match=message):
        model.load_pretrained(path)
