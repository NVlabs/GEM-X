# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# ruff: noqa: I001

import copy as deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=None,
    ):
        super().__init__()
        layers = []
        in_ch = in_channels
        for out_ch, k in zip(num_deconv_filters, num_deconv_kernels):
            padding = 1 if k == 4 else (1 if k == 3 else 0)
            output_padding = 0 if k == 4 else (1 if k == 3 else 0)
            layers += [
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    k,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)
        final_k = extra.get("final_conv_kernel", 1) if extra else 1
        pad = 1 if final_k == 3 else 0
        self.final_layer = nn.Conv2d(in_ch, out_channels, final_k, padding=pad)

    def forward(self, x):
        if isinstance(x, list | tuple):
            x = x[-1]
        return self.final_layer(self.deconv_layers(x))


def keypoints_from_heatmaps(heatmaps, center, scale, use_udp=True):
    """Standard argmax + affine coordinate transform."""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(-1)  # (N, K)
    px = (idx % W).astype(np.float32)
    py = (idx // W).astype(np.float32)
    if use_udp:
        for n in range(N):
            for k in range(K):
                hm = heatmaps[n, k]
                x, y = int(px[n, k]), int(py[n, k])
                if 1 < x < W - 1:
                    px[n, k] += np.sign(hm[y, x + 1] - hm[y, x - 1]) * 0.25
                if 1 < y < H - 1:
                    py[n, k] += np.sign(hm[y + 1, x] - hm[y - 1, x]) * 0.25
    preds = np.stack([px, py], axis=-1)  # (N, K, 2)
    preds[..., 0] = preds[..., 0] / W * (scale[:, [0]] * 200) + center[:, [0]] - scale[:, [0]] * 100
    preds[..., 1] = preds[..., 1] / H * (scale[:, [1]] * 200) + center[:, [1]] - scale[:, [1]] * 100
    maxvals = flat.max(-1, keepdims=True)
    return preds, maxvals


_VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_batch(video_or_np, bbx_xys, img_ds=1.0, path_type="np"):
    """Crop+resize+normalize RGB video frames to (256, 256) input tensor."""
    import cv2

    if isinstance(video_or_np, np.ndarray):
        frames = video_or_np
    else:
        frames = [cv2.imread(p)[..., ::-1] for p in video_or_np]
    T = len(frames)
    out = np.zeros((T, 3, 256, 256), dtype=np.float32)
    for i, (frame, bxy) in enumerate(zip(frames, bbx_xys)):
        cx, cy, s = float(bxy[0]), float(bxy[1]), float(bxy[2])
        hs = s / 2
        # Affine warp: correctly handles out-of-bounds bbox via zero-padding
        src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
        dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LINEAR)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - _VITPOSE_MEAN) / _VITPOSE_STD
        out[i] = crop.transpose(2, 0, 1)
    return torch.from_numpy(out), bbx_xys


_MODELS = {
    "Dinov3_ViTPose_huge_metrosim_256x192": dict(
        backbone=dict(
            type="ViTDinoV3",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=20,
            ffn_ratio=6,
            n_storage_tokens=4,
            layerscale_init=1e-5,
            mask_k_bias=True,
            ffn_layer="swiglu",
        ),
        keypoint_head=dict(
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=77,
        ),
    ),
}


def _build_model_local(model_name, checkpoint=None):
    if model_name not in _MODELS:
        raise ValueError("not a correct config")
    model = _MODELS[model_name]
    head_cfg = model["keypoint_head"]
    head = TopdownHeatmapSimpleHead(
        in_channels=head_cfg["in_channels"],
        out_channels=head_cfg["out_channels"],
        num_deconv_filters=head_cfg["num_deconv_filters"],
        num_deconv_kernels=head_cfg["num_deconv_kernels"],
        num_deconv_layers=head_cfg["num_deconv_layers"],
        extra=head_cfg["extra"],
    )

    backbone_cfg = model["backbone"]
    if backbone_cfg["type"] == "ViTDinoV3":
        kwargs = deepcopy.copy(backbone_cfg)
        kwargs.pop("type")
        _proto = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vits16",
            source="github",
            pretrained=False,
            skip_validation=True,
        )
        DinoVisionTransformer = type(_proto)
        del _proto

        class _ViTDinoV3Backbone(DinoVisionTransformer):
            def forward(self, x):
                return self.get_intermediate_layers(
                    x,
                    n=1,
                    reshape=True,
                    return_class_token=False,
                    return_extra_tokens=False,
                    norm=True,
                )[0]

        backbone = _ViTDinoV3Backbone(**kwargs)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_cfg['type']}")

    class VitPoseModel(nn.Module):
        def __init__(self, backbone_, keypoint_head_):
            super().__init__()
            self.backbone = backbone_
            self.keypoint_head = keypoint_head_

        def forward(self, x):
            return self.keypoint_head(self.backbone(x))

    pose = VitPoseModel(backbone, head)
    if checkpoint is not None:
        check = torch.load(checkpoint, map_location="cpu")
        pose.load_state_dict(check["state_dict"])
    return pose


def flip_heatmap_soma77(output_flipped):
    assert output_flipped.ndim == 4
    batch_size, num_joints, _, _ = output_flipped.shape
    assert num_joints == 77, f"Expected 77 joints, got {num_joints}"
    x = output_flipped.reshape(batch_size, -1, 1, output_flipped.shape[2], output_flipped.shape[3])
    y = x.clone()
    pairs = [
        (9, 10),
        (11, 39),
        (12, 40),
        (13, 41),
        (14, 42),
        (15, 43),
        (16, 44),
        (17, 45),
        (18, 46),
        (19, 47),
        (20, 48),
        (21, 49),
        (22, 50),
        (23, 51),
        (24, 52),
        (25, 53),
        (26, 54),
        (27, 55),
        (28, 56),
        (29, 57),
        (30, 58),
        (31, 59),
        (32, 60),
        (33, 61),
        (34, 62),
        (35, 63),
        (36, 64),
        (37, 65),
        (38, 66),
        (67, 72),
        (68, 73),
        (69, 74),
        (70, 75),
        (71, 76),
    ]
    for left, right in pairs:
        y[:, left, ...] = x[:, right, ...]
        y[:, right, ...] = x[:, left, ...]
    return y.reshape_as(output_flipped).flip(3)


class VitPoseExtractor:
    def __init__(self, device="cuda:0", pose_type="soma", tqdm_leave=True):
        from gem.utils.hf_utils import download_vitpose_checkpoint

        ckpt_path = download_vitpose_checkpoint()
        self.pose = _build_model_local("Dinov3_ViTPose_huge_metrosim_256x192", ckpt_path)

        self.pose.to(device).eval()
        self.device = device
        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_or_np, bbx_xys, img_ds=1.0, batch_size=16, path_type="np"):
        if isinstance(video_or_np, str | list | np.ndarray):
            imgs, bbx_xys = get_batch(video_or_np, bbx_xys, img_ds=img_ds, path_type=path_type)
        else:
            imgs = video_or_np

        total_frames = imgs.shape[0]
        results = []
        for j in tqdm(range(0, total_frames, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].to(self.device)
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(
                    torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)
                ).chunk(2)
                heatmap_flipped = flip_heatmap_soma77(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
            else:
                heatmap = self.pose(imgs_batch.clone())

            bbx_xys_batch = bbx_xys[j : j + batch_size]
            heatmap_np = heatmap.clone().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (
                torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200
            ).numpy()
            preds, maxvals = keypoints_from_heatmaps(
                heatmaps=heatmap_np, center=center, scale=scale, use_udp=True
            )
            kp2d = torch.from_numpy(np.concatenate((preds, maxvals), axis=-1))
            results.append(kp2d.detach().cpu())

        return torch.cat(results, dim=0).clone()
