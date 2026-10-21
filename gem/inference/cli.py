# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight CLI; parsing/help does not import any model or body runtime."""

import argparse


def add_motion_output_options(parser):
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply 30Hz contact/IK and retain raw predictions",
    )
    parser.add_argument("--render-dir", help="Optionally render predictions into this directory")
    parser.add_argument(
        "--soma-assets", help="SOMA0.3.3 assets, required for postprocessing or rendering"
    )
    parser.add_argument(
        "--render-variant",
        choices=("raw", "postprocessed"),
        help="Prediction to render; defaults to the active result",
    )
    parser.add_argument(
        "--render-views", nargs="+", choices=("overlay", "world"), default=["overlay", "world"]
    )
    parser.add_argument(
        "--render-width", type=int, help="Override source-size overlay / 1280px world output"
    )
    parser.add_argument("--render-backend", choices=("software", "egl"), default="software")
    parser.add_argument("--source-video", help="Source path override for rendering cached results")


def build_parser():
    parser = argparse.ArgumentParser(description="GEM framewise Pose and temporal Motion inference")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("pose-image", "Body/hand Pose from RGB with supplied boxes or automatic detection"),
        ("pose-video", "Framewise Pose with supplied tracks or automatic detection"),
    ):
        sub = commands.add_parser(command, help=description, description=description)
        sub.add_argument("--task", choices=("body", "hand"), required=True)
        sub.add_argument("--input", required=True, help="RGB image or video path")
        boxes = sub.add_mutually_exclusive_group()
        boxes.add_argument(
            "--boxes", help="JSON/YAML annotations, or saved per-hand bbox directory"
        )
        boxes.add_argument("--hand-detector", help="Local RT-DETR hand ONNX (hand task only)")
        boxes.add_argument(
            "--body-detector",
            help="Local YOLOX person ONNX; Body without --boxes defaults to weights/yolox_humanart.onnx",
        )
        sub.add_argument("--detector-device", default="cpu")
        sub.add_argument("--detector-batch-size", type=int, default=8)
        sub.add_argument(
            "--hand-view",
            choices=("ego", "exo"),
            default="exo",
            help="Hand processing profile: exo (default) or explicit wearer-view ego",
        )
        sub.add_argument(
            "--hand-tracker",
            choices=("native", "none"),
            help="Automatic video boxes: defaults to native for ego, none for exo",
        )
        sub.add_argument(
            "--hands",
            nargs="+",
            choices=("left", "right"),
            default=["left", "right"],
            help="Sides for automatic detection or legacy per-hand box files",
        )
        sub.add_argument(
            "--smooth-hand",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Smooth hand parameters and rebuild geometry; defaults on for ego video",
        )
        sub.add_argument(
            "--smooth-hand-boxes",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Stabilize tracked detector crops; defaults on for ego hand video",
        )
        sub.add_argument(
            "--square-pixels",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Correct pixel aspect for Body/Hand Pose; defaults on for ego hand except perspective crops",
        )
        cameras = sub.add_mutually_exclusive_group()
        cameras.add_argument(
            "--camera",
            help="Calibrated ego-hand source camera JSON/YAML; distorted cameras are undistorted before detection",
        )
        cameras.add_argument(
            "--perspective-camera",
            "--fisheye-camera",
            dest="perspective_camera",
            help="Shortcut for --camera FILE --perspective-crop",
        )
        cameras.add_argument(
            "--intrinsics", help="Optional pinhole intrinsics JSON/YAML overriding all frames"
        )
        sub.add_argument(
            "--perspective-crop",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Detect on the source image and rectify each hand crop; requires --camera",
        )
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument(
            "--release",
            help="body, hand, or PoseReleaseConfig JSON/YAML (defaults by task)",
        )
        sub.add_argument(
            "--soma-assets", required=True, help="Local SOMA0.3.3 assets for the selected task"
        )
        sub.add_argument("--hand-mirror", choices=("full_image", "legacy_crop"))
        sub.add_argument("--device", default="cpu")
        sub.add_argument(
            "--output", required=True, help="Output .pt file with physical geometry and caller IDs"
        )
    sub = commands.add_parser(
        "motion-video", help="Temporal Motion from video, with automatic tracking and features"
    )
    sub.add_argument("--task", choices=("body",), required=True)
    sub.add_argument("--input", required=True, help="Input video")
    sub.add_argument("--config", required=True, help="Body video release JSON/YAML")
    sub.add_argument("--start", type=float, default=0.0, help="Clip start in seconds")
    sub.add_argument("--duration", type=float, help="Clip duration, at most 60 seconds")
    sub.add_argument("--device", default="cpu")
    sub.add_argument("--output", required=True, help="Output .pt with conditions and raw Motion")
    add_motion_output_options(sub)
    sub = commands.add_parser(
        "motion-cache", help="Motion from cached conditions, or contact/IK on saved predictions"
    )
    sub.add_argument("--task", choices=("body",), required=True)
    sub.add_argument("--input", required=True, help=".pt conditions or saved Motion predictions")
    sub.add_argument(
        "--config",
        help="JSON/YAML release; required when running the Motion network",
    )
    sub.add_argument(
        "--checkpoint", help="Motion weights; omit for postprocessing saved predictions"
    )
    sub.add_argument(
        "--stats", help="Training statistics NPZ, required only for legacy checkpoints"
    )
    sub.add_argument("--device", default="cpu")
    sub.add_argument(
        "--output", required=True, help="Output .pt with camera/global SOMA parameters"
    )
    add_motion_output_options(sub)
    return parser


def resolve_pose_options(args, parser):
    """Resolve application profiles without importing image/model runtimes."""
    ego_hand = args.task == "hand" and args.hand_view == "ego"
    hand_video = args.task == "hand" and args.command == "pose-video"
    if args.task != "hand" and args.hand_view == "ego":
        parser.error("--hand-view ego requires --task hand")
    if args.hand_tracker == "native" and not ego_hand:
        parser.error("--hand-tracker native requires --task hand --hand-view ego")
    if args.smooth_hand and not hand_video:
        parser.error("--smooth-hand requires pose-video --task hand")
    camera_path = args.camera or args.perspective_camera
    if camera_path is not None and not ego_hand:
        parser.error("--camera / --perspective-camera require --task hand --hand-view ego")
    if args.perspective_camera is not None:
        if args.perspective_crop is False:
            parser.error("--perspective-camera cannot be combined with --no-perspective-crop")
        args.perspective_crop = True
    if args.perspective_crop is None:
        args.perspective_crop = False
    if args.perspective_crop and camera_path is None:
        parser.error("--perspective-crop requires --camera")
    if args.perspective_crop and args.square_pixels:
        parser.error("Perspective crops use the source image; omit --square-pixels")
    if args.hand_tracker is None:
        args.hand_tracker = "native" if ego_hand and hand_video else "none"
    if args.smooth_hand is None:
        args.smooth_hand = ego_hand and hand_video
    tracked_detector = (
        ego_hand and hand_video and args.hand_tracker == "native" and args.hand_detector is not None
    )
    if args.smooth_hand_boxes is None:
        args.smooth_hand_boxes = tracked_detector
    elif args.smooth_hand_boxes and not tracked_detector:
        parser.error("--smooth-hand-boxes requires ego video with the native RT-DETR tracker")
    if args.square_pixels is None:
        args.square_pixels = ego_hand and not args.perspective_crop


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "hand_detector", None) is not None and args.task != "hand":
        parser.error("--hand-detector requires --task hand")
    if args.command in ("pose-image", "pose-video"):
        if args.body_detector is not None and args.task != "body":
            parser.error("--body-detector requires --task body")
        if args.task == "body" and args.boxes is None:
            args.body_detector = args.body_detector or "weights/yolox_humanart.onnx"
        if args.task == "hand" and args.boxes is None and args.hand_detector is None:
            parser.error("--task hand requires --boxes or --hand-detector")
        resolve_pose_options(args, parser)
    if (
        getattr(args, "render_dir", None) is not None or getattr(args, "postprocess", False)
    ) and args.soma_assets is None:
        parser.error("--render-dir / --postprocess require --soma-assets")
    from .inputs import frame_annotations, load_mapping
    from .serialization import save_predictions

    if args.command == "motion-video":
        from .body_video import load_body_video_config, run_body_video

        result = run_body_video(
            args.input,
            load_body_video_config(args.config),
            device=args.device,
            start=args.start,
            duration=args.duration,
            progress=lambda message: print(message, flush=True),
        )
    elif args.command == "motion-cache":
        import torch

        from .motion import load_predictor, run_motion

        cached = torch.load(args.input, map_location="cpu", weights_only=False)
        if args.checkpoint is not None or args.config is not None:
            if args.checkpoint is None or args.config is None:
                parser.error("Network inference requires both --config and --checkpoint")
            config = load_mapping(args.config)
            predictor = load_predictor(
                args.checkpoint, config, stats_path=args.stats, device=args.device
            )
            result = run_motion(predictor, cached.get("conditions", cached), device=args.device)
            for key in ("video", "tracking", "camera"):
                if key in cached:
                    result[key] = cached[key]
            result["release"] = config
        elif args.postprocess and "pred_soma_params_global" in cached:
            result = cached
        else:
            parser.error(
                "Supply --config and --checkpoint, or use --postprocess with saved predictions"
            )
    else:
        from pathlib import Path

        from .pose import load_predictor, run_pose

        intrinsics = (
            None if args.intrinsics is None else load_mapping(args.intrinsics)["intrinsics"]
        )
        perspective_camera = None
        undistort_camera = None
        source_camera = None
        camera_path = args.camera or args.perspective_camera
        if camera_path is not None:
            from gem.common.camera_models import camera_from_dict

            calibration = load_mapping(camera_path)
            source_camera = camera_from_dict(calibration)
            intrinsics = source_camera.K.tolist()
            if args.perspective_crop:
                perspective_camera = calibration
            elif source_camera.model_name != "PINHOLE":
                if args.hand_detector is None:
                    parser.error(
                        "Distorted --camera requires --hand-detector for full-image undistortion; "
                        "use --perspective-crop with supplied source-image boxes"
                    )
                undistort_camera = calibration
            else:
                from .inputs import input_frames

                with input_frames(args.input, args.command == "pose-video") as frames:
                    if frames[0].shape[:2] != (source_camera.height, source_camera.width):
                        parser.error(
                            "Camera calibration dimensions must match the decoded source raster"
                        )
        provider = None
        if args.body_detector is not None:
            from .body import detect_body_annotations

            annotations, provider = detect_body_annotations(
                args.input,
                args.body_detector,
                video=args.command == "pose-video",
                intrinsics=intrinsics,
                device=args.detector_device,
            )
        elif args.hand_detector is not None:
            from .hand import detect_hand_annotations

            annotations, provider = detect_hand_annotations(
                args.input,
                args.hand_detector,
                video=args.command == "pose-video",
                intrinsics=intrinsics,
                device=args.detector_device,
                tracker=args.hand_tracker,
                batch_size=args.detector_batch_size,
                hands=tuple(dict.fromkeys(args.hands)),
                square_pixels=args.square_pixels,
                smooth_boxes=args.smooth_hand_boxes,
                undistort_camera=undistort_camera,
            )
        elif Path(args.boxes).is_dir():
            if args.task != "hand":
                parser.error("Per-hand bbox directories require --task hand")
            from .hand import saved_hand_annotations

            annotations, provider = saved_hand_annotations(
                args.input,
                args.boxes,
                video=args.command == "pose-video",
                intrinsics=intrinsics,
                hands=tuple(dict.fromkeys(args.hands)),
            )
        else:
            annotations = frame_annotations(
                load_mapping(args.boxes),
                task=args.task,
                video=args.command == "pose-video",
                intrinsics=intrinsics,
            )
        predictor, release = load_predictor(
            args.checkpoint,
            task=args.task,
            release=args.release,
            soma_assets=args.soma_assets,
            device=args.device,
            hand_mirror=args.hand_mirror,
        )
        result = run_pose(
            predictor,
            args.input,
            annotations,
            task=args.task,
            video=args.command == "pose-video",
            release=release,
            square_pixels=args.square_pixels,
            perspective_camera=perspective_camera,
            undistort_camera=undistort_camera,
        )
        if args.task == "hand":
            result["hand_view"] = args.hand_view
        if provider is not None:
            result["box_provider"] = provider
        if args.smooth_hand:
            from gem.pose.postprocess import smooth_hand_predictions

            result = smooth_hand_predictions(result, geometry=predictor.geometry)
        if undistort_camera is not None:
            from .camera import restore_source_camera_predictions

            result = restore_source_camera_predictions(result, source_camera)
    if getattr(args, "postprocess", False):
        from .postprocess import postprocess_motion_result

        print("Applying SOMA contact translation and CCD IK", flush=True)
        result = postprocess_motion_result(result, soma_assets=args.soma_assets, device=args.device)
    save_predictions(result, args.output)
    print(f"Saved {args.command} predictions to {args.output}")
    if getattr(args, "render_dir", None) is not None:
        from .render import render_motion

        render_motion(
            result,
            args.render_dir,
            soma_assets=args.soma_assets,
            device=args.device,
            views=args.render_views,
            source_video=args.source_video,
            width=args.render_width,
            backend=args.render_backend,
            variant=args.render_variant,
            progress=lambda message: print(message, flush=True),
        )


if __name__ == "__main__":
    main()
