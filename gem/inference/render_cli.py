# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight entry point for rendering saved Motion results."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render saved GEM Pose geometry or Motion parameters"
    )
    parser.add_argument("--task", choices=("body", "hand"), required=True)
    parser.add_argument("--input", required=True, help="Saved Pose or Motion prediction .pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--soma-assets", help="Required for Motion; saved Pose geometry needs no assets"
    )
    parser.add_argument("--device", default="cpu", help="SOMA geometry device")
    parser.add_argument(
        "--variant",
        choices=("raw", "postprocessed", "smoothed"),
        help="Prediction to render; defaults to the active result",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("overlay", "world"),
        help="Pose defaults to overlay; Motion to overlay and world",
    )
    parser.add_argument(
        "--source-video",
        "--source",
        help="Override source image/video path, or supply an aligned video for bare caches",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Override width; defaults to source width for overlay, 1280 for world",
    )
    parser.add_argument("--backend", choices=("software", "egl"), default="software")
    args = parser.parse_args(argv)

    import torch

    saved = torch.load(args.input, map_location="cpu", weights_only=False)
    if saved.get("format") == "gem_pose_predictions_v1":
        from .render_pose import render_pose

        if saved["task"] != args.task:
            parser.error("--task differs from the saved Pose task")
        if args.views is not None and args.views != ["overlay"]:
            parser.error("Framewise Pose results provide camera overlays only")
        result = render_pose(
            saved,
            args.output_dir,
            source=args.source_video,
            width=args.width,
            backend=args.backend,
            variant=args.variant,
            progress=lambda message: print(message, flush=True),
        )
    else:
        from .render import render_motion

        if args.task != "body" or args.soma_assets is None:
            parser.error("Motion rendering requires --task body and --soma-assets")
        result = render_motion(
            saved,
            args.output_dir,
            soma_assets=args.soma_assets,
            device=args.device,
            views=args.views or ["overlay", "world"],
            source_video=args.source_video,
            width=args.width,
            backend=args.backend,
            variant=args.variant,
            progress=lambda message: print(message, flush=True),
        )
    for kind in ("images", "videos"):
        for view, path in result.get(kind, {}).items():
            print(f"Saved {view} {kind[:-1]} to {path}")


if __name__ == "__main__":
    main()
