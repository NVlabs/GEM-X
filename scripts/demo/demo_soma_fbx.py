# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import hydra
import numpy as np
import torch
import warp as wp

wp.init()
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.net_utils import detach_to_cpu
from gem.utils.pylogger import Log
from gem.utils.video_io_utils import get_video_lwh, get_video_reader
from scripts.demo.demo_soma import (
    _build_cfg,
    _copy_video_if_needed,
    _get_body_params,
    load_data_dict,
    resolve_ckpt_path,
    run_preprocess,
)
from scripts.demo.retarget_utils import build_soma_skeleton_from_model, export_soma_bvh


class ProgressReporter:
    def __init__(self, callback=None):
        self.callback = callback

    def update(self, stage, message, percent=None):
        Log.info(f"[{stage}] {message}")
        if self.callback is not None:
            self.callback(stage=stage, message=message, percent=percent)


def _open_cv2_writer(path, width, height, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEM SOMA demo with In-cam render and BVH export."
    )
    parser.add_argument("--video", type=str, default="data/8turn_1_downsize_trim.mp4")
    parser.add_argument("--output_root", type=str, default="outputs/demo_soma")
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render_mhr", action="store_true")
    parser.add_argument("--sam3d_ckpt_path", type=str, default=None)
    parser.add_argument("--sam3d_mhr_path", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="inputs/pretrained/gem_soma.ckpt")
    parser.add_argument("--exp", type=str, default="gem_soma_regression")
    parser.add_argument(
        "--detector_name",
        type=str,
        default="vitdet",
        help="Human detector: 'vitdet' (Detectron2) or 'sam3'. Set empty to skip detection.",
    )
    parser.add_argument(
        "--bvh_path",
        type=str,
        default=None,
        help="Optional BVH output path. Defaults to <output_dir>/<video_name>_animated_skeleton.bvh",
    )
    parser.add_argument(
        "--render_incam",
        action="store_true",
        help="Render in-cam preview video.",
    )
    parser.add_argument(
        "--render_scale",
        type=float,
        default=0.5,
        help="Scale factor for in-cam render resolution. Default: 0.5",
    )
    parser.add_argument(
        "--composite_background",
        action="store_true",
        help="Composite rendered mesh over the input video background.",
    )
    return parser.parse_args()


def _default_bvh_path(cfg, args):
    output_dir = Path(cfg.output_dir)
    if getattr(args, "bvh_path", None):
        return Path(args.bvh_path)
    return output_dir / f"{cfg.video_name}_animated_skeleton.bvh"


def _to_cpu_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.as_tensor(x).float()


@torch.no_grad()
def export_bvh(cfg, fps, bvh_path):
    pred = torch.load(cfg.paths.hpe_results)
    body_params_global = _get_body_params(pred, "body_params_global")
    identity_coeffs = _to_cpu_float_tensor(body_params_global["identity_coeffs"])
    scale_params = _to_cpu_float_tensor(body_params_global["scale_params"])
    skeleton = build_soma_skeleton_from_model(identity_coeffs, scale_params)
    bvh_path.parent.mkdir(parents=True, exist_ok=True)
    export_soma_bvh(body_params_global, skeleton, fps, str(bvh_path))
    return bvh_path


@torch.no_grad()
def render_incam_with_progress(
    cfg,
    fps,
    progress_callback=None,
    percent_start=80,
    percent_end=92,
    render_scale=0.5,
    composite_background=False,
):
    import open3d as o3d

    from gem.utils.soma_utils.soma_layer import SomaLayer
    from gem.utils.vis.o3d_render import Settings, create_meshes

    pred = torch.load(cfg.paths.hpe_results)
    body_params_incam = _get_body_params(pred, "body_params_incam")

    device = "cuda:0"
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    )
    pred_c_verts = soma(**{k: v.cuda() for k, v in body_params_incam.items()})["vertices"]
    faces = soma.faces.long().cuda()

    video_path = cfg.video_path
    _, src_width, src_height = get_video_lwh(video_path)
    render_scale = float(render_scale)
    render_width = max(1, int(src_width * render_scale))
    render_height = max(1, int(src_height * render_scale))
    K = pred["K_fullimg"][0]
    K = K.clone()
    K[0, 0] *= render_width / src_width
    K[1, 1] *= render_height / src_height
    K[0, 2] *= render_width / src_width
    K[1, 2] *= render_height / src_height

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.4, 0.8, 0.4], device=device)

    renderer = o3d.visualization.rendering.OffscreenRenderer(render_width, render_height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0] if not composite_background else [0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.01, 100.0, float(render_width), float(render_height)
    )
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    total_frames = int(pred_c_verts.shape[0])
    update_every = max(total_frames // 50, 1)
    reader = get_video_reader(video_path) if composite_background else None
    writer = _open_cv2_writer(cfg.paths.incam_video, render_width, render_height, fps)

    for i in range(total_frames):
        img_raw = None
        if reader is not None:
            try:
                img_raw = next(reader)
            except StopIteration:
                break
        mesh = create_meshes(pred_c_verts[i], faces, color)
        mesh_name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())
        if composite_background and img_raw is not None:
            img_raw = cv2.resize(img_raw, (render_width, render_height), interpolation=cv2.INTER_AREA)
            depth = np.asarray(renderer.render_to_depth_image())
            mask = (depth < 1.0).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.0)
            alpha = mask[..., np.newaxis]
            frame = rendered.astype(np.float32) * alpha + img_raw.astype(np.float32) * (1.0 - alpha)
            frame = frame.clip(0, 255).astype(np.uint8)
        else:
            frame = rendered.astype(np.uint8)
        writer.write(frame[..., ::-1])

        if progress_callback is not None and (
            i == 0 or (i + 1) % update_every == 0 or (i + 1) == total_frames
        ):
            ratio = float(i + 1) / float(total_frames)
            percent = percent_start + (percent_end - percent_start) * ratio
            progress_callback(
                stage="RENDER",
                message=f"Rendering in-cam video ({i + 1}/{total_frames})",
                percent=int(percent),
            )

    writer.release()
    if reader is not None:
        reader.close()


def run_demo_pipeline(args, progress_callback=None):
    reporter = ProgressReporter(progress_callback)

    reporter.update("SETUP", "Building config", 5)
    cfg = _build_cfg(args)

    reporter.update("SETUP", "Preparing input video", 10)
    _copy_video_if_needed(cfg)

    reporter.update("PREPROCESS", "Running preprocessing", 25)
    run_preprocess(cfg)

    reporter.update("INFERENCE", "Loading model inputs", 40)
    fps = int(cv2.VideoCapture(cfg.video_path).get(cv2.CAP_PROP_FPS) + 0.5) or 30
    data = load_data_dict(cfg)

    if not Path(cfg.paths.hpe_results).exists():
        reporter.update("INFERENCE", "Running 3D body pose estimation", 60)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        ckpt_path = resolve_ckpt_path(cfg)
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        torch.save(detach_to_cpu(pred), cfg.paths.hpe_results)
    else:
        reporter.update("INFERENCE", "Using cached 3D body pose result", 60)

    video_path = None
    if getattr(args, "render_incam", False):
        reporter.update("RENDER", "Rendering in-cam video", 80)
        render_incam_with_progress(
            cfg,
            fps=fps,
            progress_callback=progress_callback,
            percent_start=80,
            percent_end=92,
            render_scale=getattr(args, "render_scale", 0.5),
            composite_background=getattr(args, "composite_background", False),
        )
        video_path = str(Path(cfg.paths.incam_video))
    else:
        reporter.update("RENDER", "Skipping in-cam render", 80)

    reporter.update("EXPORT", "Exporting BVH", 92)
    bvh_path = export_bvh(cfg, fps, _default_bvh_path(cfg, args))

    _, width, height = get_video_lwh(cfg.video_path)
    result = {
        "video_path": video_path,
        "bvh_path": str(bvh_path),
        "output_dir": str(Path(cfg.output_dir)),
        "fps": fps,
        "frame_size": [width, height],
    }
    reporter.update("DONE", "Pipeline completed", 100)
    return result


def run_demo_from_kwargs(**kwargs):
    args = SimpleNamespace(
        video=kwargs.get("video", "data/8turn_1_downsize_trim.mp4"),
        output_root=kwargs.get("output_root", "outputs/demo_soma"),
        static_cam=kwargs.get("static_cam", False),
        verbose=kwargs.get("verbose", False),
        render_mhr=kwargs.get("render_mhr", False),
        sam3d_ckpt_path=kwargs.get("sam3d_ckpt_path"),
        sam3d_mhr_path=kwargs.get("sam3d_mhr_path"),
        ckpt=kwargs.get("ckpt", "inputs/pretrained/gem_soma.ckpt"),
        exp=kwargs.get("exp", "gem_soma_regression"),
        detector_name=kwargs.get("detector_name", "vitdet"),
        bvh_path=kwargs.get("bvh_path"),
        render_incam=kwargs.get("render_incam", False),
        render_scale=kwargs.get("render_scale", 0.5),
        composite_background=kwargs.get("composite_background", False),
    )
    return run_demo_pipeline(args, progress_callback=kwargs.get("progress_callback"))


def main():
    args = _parse_args()
    result = run_demo_pipeline(args)
    Log.info(
        f"[Done] Saved In-cam render to {result['video_path']} and BVH to {result['bvh_path']}"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    main()
