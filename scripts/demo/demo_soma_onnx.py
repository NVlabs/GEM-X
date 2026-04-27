# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accelerated GEM demo using ONNX / TensorRT inference.

This is a fast variant of ``demo_soma.py`` that replaces PyTorch inference
with ONNX Runtime (+ optional TensorRT EP) for the three heavy models:

  1. **VitPose** — 2D keypoint detection (77 SOMA joints)
  2. **SAM-3D-Body** — pose-token extraction (1024-dim)
  3. **GEM Denoiser** — motion prediction (regression mode)

The rendering / post-processing steps remain in PyTorch since they are I/O-
bound and not the bottleneck.

Workflow
--------
Just run the demo — ONNX models are downloaded automatically from
HuggingFace (``nvidia/GEM-X``) on first use::

       python scripts/demo/demo_soma_onnx.py --video inputs/demo.mp4

If an ONNX model is missing and auto-download fails, the script falls back
to PyTorch.
"""

# ruff: noqa: E402, I001
import argparse
import os
import sys
import time
from pathlib import Path

_timings: dict[str, float] = {}

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

# Preload cuDNN 9 from pip nvidia-cudnn package so ONNX Runtime CUDA EP works.
try:
    import nvidia.cudnn as _cudnn
    import ctypes as _ctypes

    if _cudnn.__file__ is not None:
        _cudnn_lib = str(Path(_cudnn.__file__).parent / "lib" / "libcudnn.so.9")
        _ctypes.cdll.LoadLibrary(_cudnn_lib)
except (ImportError, OSError):
    pass

import cv2
import numpy as np
import torch

# Wrap torch.load to handle files saved with numpy 2.x (numpy._core) on numpy 1.x.
_original_torch_load = torch.load
_need_numpy_shim = not hasattr(np, "_core")


def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    if _need_numpy_shim:
        # Temporarily inject numpy._core aliases so torch's unpickler resolves them.
        import numpy.core as _np_core

        _added = {}
        for mod in ("numpy._core", "numpy._core.multiarray"):
            if mod not in sys.modules:
                target = (
                    _np_core if mod == "numpy._core" else getattr(_np_core, "multiarray", _np_core)
                )
                sys.modules[mod] = target
                _added[mod] = True
        try:
            return _original_torch_load(*args, **kwargs)
        finally:
            for mod in _added:
                sys.modules.pop(mod, None)
    return _original_torch_load(*args, **kwargs)


torch.load = _compat_torch_load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.cam_utils import compute_transl_full_cam, estimate_K, get_a_pred_cam
from gem.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    get_bbx_xys_from_xyxy,
    normalize_T_w2c,
)
from gem.utils.pylogger import Log


# ---------------------------------------------------------------------------
#  Lazy imports for heavy pipeline dependencies
# ---------------------------------------------------------------------------
# These modules pull in pytorch_lightning → torchmetrics → torchvision →
# matplotlib, which hangs on macOS (font-cache build).  demo_webcam.py only
# needs OnnxRunner / load_vitpose / load_denoiser from this file and must not
# trigger the full import chain.  We defer these imports to first use.


def _ensure_pipeline_deps():
    """Import heavy deps on first use."""
    global smooth_bbx_xyxy, detach_to_cpu, to_cuda, SomaLayer
    global get_video_lwh, get_video_reader, get_writer
    global merge_videos_grid_2x2, merge_videos_horizontal, read_video_np, save_video
    global draw_bbx_xyxy_on_image_batch
    global Settings, create_meshes, get_ground
    global get_global_cameras_static_v2, get_ground_params_from_points
    if "SomaLayer" in globals():
        return
    from gem.utils.kp2d_utils import smooth_bbx_xyxy as _smooth_bbx_xyxy
    from gem.utils.net_utils import detach_to_cpu as _detach_to_cpu, to_cuda as _to_cuda
    from gem.utils.soma_utils.soma_layer import SomaLayer as _SomaLayer
    from gem.utils.video_io_utils import (
        get_video_lwh as _get_video_lwh,
        get_video_reader as _get_video_reader,
        get_writer as _get_writer,
        merge_videos_grid_2x2 as _merge_videos_grid_2x2,
        merge_videos_horizontal as _merge_videos_horizontal,
        read_video_np as _read_video_np,
        save_video as _save_video,
    )
    from gem.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch as _draw_bbx
    from gem.utils.vis.o3d_render import (
        Settings as _Settings,
        create_meshes as _create_meshes,
        get_ground as _get_ground,
    )
    from gem.utils.vis.renderer import (
        get_global_cameras_static_v2 as _get_global_cameras_static_v2,
        get_ground_params_from_points as _get_ground_params_from_points,
    )

    smooth_bbx_xyxy = _smooth_bbx_xyxy
    detach_to_cpu = _detach_to_cpu
    to_cuda = _to_cuda
    SomaLayer = _SomaLayer
    get_video_lwh = _get_video_lwh
    get_video_reader = _get_video_reader
    get_writer = _get_writer
    merge_videos_grid_2x2 = _merge_videos_grid_2x2
    merge_videos_horizontal = _merge_videos_horizontal
    read_video_np = _read_video_np
    save_video = _save_video
    draw_bbx_xyxy_on_image_batch = _draw_bbx
    Settings = _Settings
    create_meshes = _create_meshes
    get_ground = _get_ground
    get_global_cameras_static_v2 = _get_global_cameras_static_v2
    get_ground_params_from_points = _get_ground_params_from_points


CRF = 23

# ──────────────────────────────────────────────────────────────────────
#  ONNX / TRT runners
# ──────────────────────────────────────────────────────────────────────


class OnnxRunner:
    """Generic ONNX Runtime inference session with GPU IOBinding support."""

    def __init__(self, onnx_path: str, device: str = "cuda"):
        import onnxruntime as ort

        providers = []
        if device == "cuda":
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        if sys.platform == "darwin":
            # ONNX Runtime's built-in CoreML EP (auto-converts ONNX → CoreML
            # at load time).  Distinct from the native CoreMLRunner which loads
            # pre-converted .mlpackage files for better ANE utilization.
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Resolve symlinks so external data files are found relative to the real path.
        real_path = str(Path(onnx_path).resolve())
        self.sess = ort.InferenceSession(real_path, sess_options=so, providers=providers)
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]
        active_ep = self.sess.get_providers()
        self._use_cuda = "CUDAExecutionProvider" in active_ep
        Log.info(f"[ONNX] Loaded {onnx_path} (EP={active_ep}, inputs={self.input_names})")

    def __call__(self, **kwargs) -> dict:
        if self._use_cuda:
            return self._run_cuda(**kwargs)
        return self._run_cpu(**kwargs)

    def _run_cpu(self, **kwargs) -> dict:
        feed = {}
        for name in self.input_names:
            val = kwargs[name]
            feed[name] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
        ort_out = self.sess.run(self.output_names, feed)
        return {name: torch.from_numpy(arr) for name, arr in zip(self.output_names, ort_out)}

    def _run_cuda(self, **kwargs) -> dict:

        io = self.sess.io_binding()
        # Keep references to all input tensors so their GPU memory stays valid
        # until after run_with_iobinding completes.
        bound_inputs = []
        for name in self.input_names:
            val = kwargs[name]
            if isinstance(val, torch.Tensor):
                t = val.contiguous().cuda()
            else:
                t = torch.from_numpy(np.ascontiguousarray(val)).cuda()
            bound_inputs.append(t)
            io.bind_input(
                name,
                "cuda",
                0,
                _torch_to_np_dtype(t.dtype),
                list(t.shape),
                t.data_ptr(),
            )

        for name in self.output_names:
            io.bind_output(name, "cuda", 0)

        self.sess.run_with_iobinding(io)
        del bound_inputs  # safe to release now
        ort_outputs = io.get_outputs()
        result = {}
        for name, ort_val in zip(self.output_names, ort_outputs):
            arr = ort_val.numpy()  # OrtValue → numpy (on CPU after device copy)
            result[name] = torch.from_numpy(arr)
        return result


def _torch_to_np_dtype(dtype):
    """Map torch dtype to numpy dtype for IOBinding."""
    _map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float64: np.float64,
        torch.int8: np.int8,
    }
    return _map.get(dtype, np.float32)


class CoreMLRunner:
    """CoreML (.mlpackage) inference runner for Apple Silicon.

    Provides the same ``__call__(**kwargs) -> dict`` interface as OnnxRunner
    and TRTRunner so it can be used as a drop-in replacement.
    """

    def __init__(self, mlpackage_path: str):
        import coremltools as ct

        self.model = ct.models.MLModel(mlpackage_path)
        spec = self.model.get_spec()
        self.input_names = [inp.name for inp in spec.description.input]
        self.output_names = [out.name for out in spec.description.output]
        Log.info(f"[CoreML] Loaded {mlpackage_path} (inputs={self.input_names})")

    def __call__(self, **kwargs) -> dict:
        feed = {}
        for name in self.input_names:
            val = kwargs[name]
            if isinstance(val, torch.Tensor):
                feed[name] = val.cpu().numpy()
            else:
                feed[name] = np.ascontiguousarray(val)

        prediction = self.model.predict(feed)
        return {name: torch.from_numpy(np.array(prediction[name])) for name in self.output_names}


class TRTRunner:
    """TensorRT engine runner (for models with fixed input/output names)."""

    def __init__(
        self, engine_path: str, input_names: list, output_shapes: dict, device: str = "cuda"
    ):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.input_names = input_names
        self.output_shapes = output_shapes
        Log.info(f"[TRT] Loaded {engine_path}")

    @torch.inference_mode()
    def __call__(self, **kwargs) -> dict:
        for name in self.input_names:
            tensor = kwargs[name].contiguous().to(self.device)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        outputs = {}
        for name, shape_fn in self.output_shapes.items():
            shape = tuple(self.context.get_tensor_shape(name))
            if shape[0] <= 0:
                shape = shape_fn(kwargs)
            out = torch.empty(shape, device=self.device, dtype=torch.float32)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        return outputs


# ──────────────────────────────────────────────────────────────────────
#  Model loaders with auto-fallback
# ──────────────────────────────────────────────────────────────────────

_PREFER_QUANTIZED = sys.platform == "darwin" or not torch.cuda.is_available()
_IS_MACOS = sys.platform == "darwin"


def _find_coreml(onnx_path: str) -> str | None:
    """Return the CoreML .mlpackage path corresponding to an ONNX model, or None."""
    if not _IS_MACOS:
        return None
    p = Path(onnx_path)
    # Convention: inputs/onnx/vitpose.onnx → inputs/coreml/vitpose.mlpackage
    candidate = p.parent.parent / "coreml" / f"{p.stem}.mlpackage"
    if candidate.exists():
        return str(candidate)
    return None


def _find_best_onnx(base_path: str) -> str:
    """On CPU/CoreML, prefer quantized ONNX variants if available.

    Checks for ``<stem>_fp16.onnx`` and ``<stem>_int8.onnx`` alongside
    the base model.  On CUDA systems, always returns the base path.
    """
    if not _PREFER_QUANTIZED:
        return base_path
    p = Path(base_path)
    for suffix in ("_fp16", "_int8"):
        candidate = p.with_name(f"{p.stem}{suffix}.onnx")
        if candidate.exists():
            Log.info(f"[ONNX] Using quantized model: {candidate.name}")
            return str(candidate)
    return base_path


def load_vitpose(onnx_path="inputs/onnx/vitpose.onnx", trt_path="outputs/trt/vitpose.engine"):
    """Load VitPose: prefer CoreML > TRT > quantized ONNX > ONNX > PyTorch."""
    # CoreML (macOS only)
    coreml_path = _find_coreml(onnx_path)
    if coreml_path:
        try:
            return CoreMLRunner(coreml_path), "coreml"
        except Exception as e:
            Log.warn(f"[CoreML] VitPose load failed: {e}")

    if os.path.exists(trt_path):
        try:
            return (
                TRTRunner(
                    trt_path,
                    input_names=["imgs"],
                    output_shapes={"heatmaps": lambda kw: (kw["imgs"].shape[0], 77, 16, 12)},
                ),
                "trt",
            )
        except Exception as e:
            Log.warn(f"[TRT] VitPose load failed: {e}")

    # Auto-download ONNX from HuggingFace if not present
    if not os.path.exists(onnx_path):
        try:
            from gem.utils.hf_utils import download_onnx_model

            Log.info("[VitPose] Downloading ONNX model from HuggingFace...")
            download_onnx_model("vitpose")
        except Exception as e:
            Log.warn(f"[VitPose] ONNX download failed: {e}")

    best_onnx = _find_best_onnx(onnx_path)
    for candidate in dict.fromkeys([best_onnx, onnx_path]):
        if os.path.exists(candidate):
            try:
                return OnnxRunner(candidate), "onnx"
            except Exception as e:
                Log.warn(f"[ONNX] VitPose load failed ({candidate}): {e}")

    Log.info("[VitPose] Falling back to PyTorch")
    from gem.utils.vitpose_extractor import VitPoseExtractor

    return VitPoseExtractor(device="cuda:0", pose_type="soma"), "pytorch"


class _OnnxBackbone(torch.nn.Module):
    """Drop-in replacement for the DINOv2 backbone that runs ONNX under the hood.

    The ONNX model expects ``(B, 3, H, W)`` **uint8-range** crops.  It applies
    ImageNet normalization internally (baked into the ONNX graph), so we must
    *undo* the normalization that ``data_preprocess`` already applied.
    """

    def __init__(self, onnx_runner, image_mean, image_std):
        super().__init__()
        self._runner = onnx_runner
        # Store the mean/std used by data_preprocess so we can invert it.
        self.register_buffer("_mean", image_mean.view(1, 3, 1, 1))
        self.register_buffer("_std", image_std.view(1, 3, 1, 1))

    def forward(self, x, **kwargs):
        # ``x`` arrives as ``(x_raw / 255 - mean) / std`` from data_preprocess.
        # Invert to get uint8-range [0, 255] that the ONNX graph expects.
        imgs_uint8 = (x.float() * self._std + self._mean) * 255.0
        out = self._runner(imgs=imgs_uint8)
        emb = out["image_embeddings"]
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        return emb.to(x.device, dtype=x.dtype)


def load_sam3db(
    backbone_onnx_path="inputs/onnx/sam3db_backbone.onnx",
    backbone_trt_path="outputs/trt/sam3db_backbone.engine",
    cfg=None,
):
    """Load SAM-3D-Body with hybrid ONNX backbone + PyTorch decoder.

    The backbone (DINOv2 ViT-Huge, ~90% of compute) runs via ONNX/TRT while
    the lightweight decoder stays in PyTorch.  Falls back to full PyTorch if
    no ONNX/TRT backbone is found.
    """
    from gem.utils.sam3db_extractor import SAM3DBExtractor

    ckpt = cfg.get("sam3d_ckpt_path", None) if cfg else None
    mhr = cfg.get("sam3d_mhr_path", None) if cfg else None
    extractor = SAM3DBExtractor(checkpoint_path=ckpt, mhr_path=mhr, device="cuda:0")

    # Try to replace the backbone with ONNX/TRT
    backbone_runner = None
    backend = "pytorch"
    if os.path.exists(backbone_trt_path):
        try:
            backbone_runner = TRTRunner(
                backbone_trt_path,
                input_names=["imgs"],
                output_shapes={
                    "image_embeddings": lambda kw: (kw["imgs"].shape[0], 1280, 16, 12),
                },
            )
            backend = "trt"
        except Exception as e:
            Log.warn(f"[TRT] SAM3DB backbone load failed: {e}")

    # Auto-download ONNX from HuggingFace if not present
    if backbone_runner is None and not os.path.exists(backbone_onnx_path):
        try:
            from gem.utils.hf_utils import download_onnx_model

            Log.info("[SAM-3D-Body] Downloading ONNX backbone from HuggingFace...")
            download_onnx_model("sam3db_backbone")
        except Exception as e:
            Log.warn(f"[SAM-3D-Body] ONNX download failed: {e}")

    if backbone_runner is None and os.path.exists(backbone_onnx_path):
        try:
            backbone_runner = OnnxRunner(backbone_onnx_path)
            backend = "onnx"
        except Exception as e:
            Log.warn(f"[ONNX] SAM3DB backbone load failed: {e}")

    if backbone_runner is not None:
        model = extractor.estimator.model
        image_mean = model.image_mean.clone()
        image_std = model.image_std.clone()
        # Replace backbone with ONNX-backed module and free GPU memory
        old_backbone = model.backbone
        model.backbone = _OnnxBackbone(backbone_runner, image_mean, image_std).to("cuda:0")
        del old_backbone
        torch.cuda.empty_cache()
        Log.info(f"[SAM-3D-Body] Using hybrid {backend.upper()} backbone + PyTorch decoder")
    else:
        Log.info("[SAM-3D-Body] Using full PyTorch backend")

    return extractor, backend


def load_denoiser(
    onnx_path="inputs/onnx/gem_denoiser.onnx",
    trt_path="outputs/trt/gem_denoiser.engine",
    no_imgfeat=False,
):
    """Load GEM Denoiser: prefer CoreML > TRT > quantized ONNX > ONNX > None.

    When *no_imgfeat* is True, look for the patched no-imgfeat model first.
    """
    # CoreML (macOS only) — check no-imgfeat variant first
    source_onnx = onnx_path
    if no_imgfeat:
        source_onnx = onnx_path.replace("gem_denoiser.onnx", "gem_denoiser_no_imgfeat.onnx")
    coreml_path = _find_coreml(source_onnx)
    if coreml_path:
        try:
            return CoreMLRunner(coreml_path), "coreml"
        except Exception as e:
            Log.warn(f"[CoreML] Denoiser load failed: {e}")

    # When no_imgfeat, prefer the patched ONNX model that bakes in the absent bias.
    if no_imgfeat:
        no_imgfeat_onnx = onnx_path.replace("gem_denoiser.onnx", "gem_denoiser_no_imgfeat.onnx")
        # Auto-download from HuggingFace if not present
        if not os.path.exists(no_imgfeat_onnx):
            try:
                from gem.utils.hf_utils import download_onnx_model

                Log.info("[Denoiser] Downloading no-imgfeat ONNX model from HuggingFace...")
                download_onnx_model("gem_denoiser_no_imgfeat")
            except Exception as e:
                Log.warn(f"[Denoiser] ONNX download failed: {e}")
        best_nif = _find_best_onnx(no_imgfeat_onnx)
        for candidate in dict.fromkeys([best_nif, no_imgfeat_onnx]):
            if os.path.exists(candidate):
                try:
                    return OnnxRunner(candidate), "onnx"
                except Exception as e:
                    Log.warn(f"[ONNX] No-imgfeat denoiser load failed ({candidate}): {e}")

    if os.path.exists(trt_path):
        try:
            return (
                TRTRunner(
                    trt_path,
                    input_names=["obs", "bbx_xys", "K_fullimg", "f_imgseq", "f_cam_angvel"],
                    output_shapes={
                        "pred_x": lambda kw: (kw["obs"].shape[0], kw["obs"].shape[1], 585),
                        "pred_cam": lambda kw: (kw["obs"].shape[0], kw["obs"].shape[1], 3),
                    },
                ),
                "trt",
            )
        except Exception as e:
            Log.warn(f"[TRT] Denoiser load failed: {e}")

    # Auto-download standard denoiser ONNX from HuggingFace if not present
    if not os.path.exists(onnx_path):
        try:
            from gem.utils.hf_utils import download_onnx_model

            Log.info("[Denoiser] Downloading ONNX model from HuggingFace...")
            download_onnx_model("gem_denoiser")
        except Exception as e:
            Log.warn(f"[Denoiser] ONNX download failed: {e}")

    best_onnx = _find_best_onnx(onnx_path)
    candidates = dict.fromkeys([best_onnx, onnx_path])  # deduplicate, preserve order
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                return OnnxRunner(candidate), "onnx"
            except Exception as e:
                Log.warn(f"[ONNX] Denoiser load failed ({candidate}): {e}")

    return None, "pytorch"


# ──────────────────────────────────────────────────────────────────────
#  VitPose inference wrapper
# ──────────────────────────────────────────────────────────────────────

_VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def vitpose_preprocess(frames, bbx_xys):
    """Crop+resize+normalize for VitPose: (T, 3, 256, 256)."""
    T = len(frames)
    out = np.zeros((T, 3, 256, 256), dtype=np.float32)
    for i, (frame, bxy) in enumerate(zip(frames, bbx_xys)):
        cx, cy, s = float(bxy[0]), float(bxy[1]), float(bxy[2])
        hs = s / 2
        src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
        dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LINEAR)
        crop = crop[..., ::-1].astype(np.float32) / 255.0  # BGR→RGB
        crop = (crop - _VITPOSE_MEAN) / _VITPOSE_STD
        out[i] = crop.transpose(2, 0, 1)
    return torch.from_numpy(out), bbx_xys


def vitpose_postprocess(heatmaps_np, bbx_xys):
    """Argmax + affine coordinate transform on VitPose heatmaps."""
    from gem.utils.vitpose_extractor import keypoints_from_heatmaps

    center = bbx_xys[:, :2].numpy()
    scale = (torch.cat((bbx_xys[:, [2]] * 24 / 32, bbx_xys[:, [2]]), dim=1) / 200).numpy()
    preds, maxvals = keypoints_from_heatmaps(
        heatmaps=heatmaps_np, center=center, scale=scale, use_udp=True
    )
    return torch.from_numpy(np.concatenate((preds, maxvals), axis=-1))


def run_vitpose_onnx(runner, backend, frames, bbx_xys, batch_size=16):
    """Run VitPose inference using ONNX/TRT or fallback to PyTorch."""
    if backend == "pytorch":
        return runner.extract(frames, bbx_xys, img_ds=1.0, path_type="np")

    imgs, bbx_xys_t = vitpose_preprocess(frames, bbx_xys)
    total = imgs.shape[0]
    results = []

    for j in range(0, total, batch_size):
        batch = imgs[j : j + batch_size, :, :, 32:224]  # Crop to 256x192
        if backend == "trt":
            batch = batch.cuda()
        out = runner(imgs=batch)
        hm = out["heatmaps"]
        if isinstance(hm, torch.Tensor):
            hm = hm.cpu().numpy()

        # Flip test
        batch_flip = torch.flip(
            batch if isinstance(batch, torch.Tensor) else torch.from_numpy(batch), dims=[3]
        )
        if backend == "trt":
            batch_flip = batch_flip.cuda()
        out_flip = runner(imgs=batch_flip)
        hm_flip = out_flip["heatmaps"]
        if isinstance(hm_flip, torch.Tensor):
            hm_flip = hm_flip.cpu().numpy()

        from gem.utils.vitpose_extractor import flip_heatmap_soma77

        hm_flip = flip_heatmap_soma77(torch.from_numpy(hm_flip)).numpy()
        hm = (hm + hm_flip) * 0.5

        bbx_batch = bbx_xys_t[j : j + batch_size]
        kp2d = vitpose_postprocess(hm, bbx_batch)
        results.append(kp2d)

    return torch.cat(results, dim=0)


# ──────────────────────────────────────────────────────────────────────
#  SAM-3D-Body inference wrapper
# ──────────────────────────────────────────────────────────────────────


def run_sam3db_onnx(runner, backend, video_path, bbx_xys, batch_size=16, W=640, H=480):
    """Run SAM-3D-Body using ONNX/TRT or fallback to PyTorch."""
    if backend == "pytorch":
        sam3d_results = runner.extract_video_features(video_path, bbx_xys)
        return sam3d_results["pose_tokens"]

    # For ONNX/TRT: need to manually preprocess
    from gem.utils.video_io_utils import read_video_np

    frames = read_video_np(video_path)
    bbx_xys_np = bbx_xys.float().cpu().numpy()

    K_fullimg = estimate_K(W, H)
    tokens = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        batch_bbx = bbx_xys_np[i : i + batch_size]
        B = len(batch_frames)

        # Affine-transform crops (same as prepare_batch)
        crops = np.zeros((B, 3, 256, 256), dtype=np.float32)
        cond_info = np.zeros((B, 3), dtype=np.float32)
        for j, (frame, bxy) in enumerate(zip(batch_frames, batch_bbx)):
            cx, cy, s = float(bxy[0]), float(bxy[1]), float(bxy[2])
            hs = s / 2
            src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
            dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LINEAR)
            crops[j] = crop.transpose(2, 0, 1).astype(np.float32)  # Keep as uint8-range
            # CLIFF condition: (cx - img_cx) / fl, (cy - img_cy) / fl, bbox_size / fl
            fl = K_fullimg[0, 0].item()
            icx = K_fullimg[0, 2].item()
            icy = K_fullimg[1, 2].item()
            cond_info[j] = [(cx - icx) / fl, (cy - icy) / fl, s / fl]

        crops_t = torch.from_numpy(crops)
        cond_t = torch.from_numpy(cond_info)
        if backend == "trt":
            crops_t = crops_t.cuda()
            cond_t = cond_t.cuda()

        out = runner(imgs=crops_t, condition_info=cond_t)
        pt = out["pose_tokens"]
        if isinstance(pt, torch.Tensor):
            pt = pt.cpu()
        tokens.append(pt)

    return torch.cat(tokens, dim=0)


# ──────────────────────────────────────────────────────────────────────
#  GEM denoiser inference wrapper
# ──────────────────────────────────────────────────────────────────────


def run_denoiser_onnx(runner, backend, batch):
    """Run GEM denoiser using ONNX/TRT. Returns pred_x, pred_cam tensors."""
    out = runner(
        obs=batch["obs"],
        bbx_xys=batch["bbx_xys"],
        K_fullimg=batch["K_fullimg"],
        f_imgseq=batch["f_imgseq"],
        f_cam_angvel=batch["f_cam_angvel"],
    )
    pred_x = out["pred_x"]
    pred_cam = out["pred_cam"]
    if isinstance(pred_x, torch.Tensor):
        pred_x = pred_x.cuda()
        pred_cam = pred_cam.cuda()
    return pred_x, pred_cam


# ──────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(description="GEM accelerated demo (ONNX/TRT)")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/demo_soma_onnx")
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--exp", type=str, default="gem_soma_regression")
    parser.add_argument(
        "--force_pytorch",
        action="store_true",
        help="Force PyTorch inference even if ONNX/TRT available",
    )
    parser.add_argument("--retarget", action="store_true")
    parser.add_argument(
        "--no-imgfeat",
        action="store_true",
        help="Skip SAM3DB and disable image feature conditioning (2D kp only)",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM sampling (50 steps) instead of regression. "
        "Slower but much better quality, especially for --no-imgfeat mode.",
    )
    return parser.parse_args()


def _build_cfg(args):
    from hydra import compose, initialize_config_dir

    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    cfg_dir = str(PROJECT_ROOT / "configs")

    overrides = [
        f"exp={args.exp}",
        f"video_name={video_path.stem}",
        f"video_path={video_path}",
        f"output_root={args.output_root}",
        f"static_cam={str(args.static_cam).lower()}",
        f"verbose={str(args.verbose).lower()}",
        "render_mhr=false",
        "use_wandb=false",
        "task=test",
    ]
    if args.ckpt is not None:
        overrides.append(f"ckpt_path={args.ckpt}")

    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="demo_soma", overrides=overrides)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@torch.no_grad()
def run_preprocess_fast(cfg, force_pytorch=False, no_imgfeat=False):
    """Preprocessing with ONNX/TRT acceleration."""
    _ensure_pipeline_deps()
    Log.info("[Preprocess] Start (fast mode)")
    video_path = cfg.video_path
    paths = cfg.paths
    L, W, H = get_video_lwh(video_path)

    # --- Human detection ---
    if not Path(paths.bbx).exists():
        Log.info("[Preprocess] Running human detection with YOLOX + ByteTrack...")
        t0 = time.time()
        frames = read_video_np(video_path)

        from gem.utils.yolox_detector import YOLOXDetector, detect_and_track

        yolox = YOLOXDetector(device="cuda")
        bbx_xyxy_np, _ = detect_and_track(frames, yolox)
        bbx_xyxy = torch.from_numpy(bbx_xyxy_np).float()
        bbx_xyxy = smooth_bbx_xyxy(bbx_xyxy, window=3)

        bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
        bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        _timings["Detection (YOLOX+ByteTrack)"] = time.time() - t0
        Log.info(f"[Preprocess] Detection done. Saved {L} bboxes.")

    bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    # --- VitPose 2D keypoints (ONNX/TRT) ---
    if not Path(paths.vitpose).exists():
        vitpose_runner, vitpose_backend = load_vitpose() if not force_pytorch else (None, "pytorch")
        if vitpose_backend == "pytorch":
            from gem.utils.vitpose_extractor import VitPoseExtractor

            vp = VitPoseExtractor(device="cuda:0", pose_type="soma")
            frames = read_video_np(video_path)
            vitpose = vp.extract(frames, bbx_xys, img_ds=1.0, path_type="np")
        else:
            frames = read_video_np(video_path)
            t0 = time.time()
            vitpose = run_vitpose_onnx(vitpose_runner, vitpose_backend, frames, bbx_xys)
            _timings["VitPose 2D keypoints"] = time.time() - t0
            Log.info(
                f"[VitPose] {vitpose_backend.upper()} inference: {_timings['VitPose 2D keypoints']:.2f}s for {L} frames"
            )
        torch.save(vitpose, paths.vitpose)
    else:
        vitpose = torch.load(paths.vitpose)

    # --- Camera tracking ---
    if cfg.static_cam or not Path(paths.slam).exists():
        eye = torch.eye(4).unsqueeze(0).repeat(L, 1, 1).numpy()
        torch.save(eye, paths.slam)

    # --- SAM-3D-Body features (hybrid ONNX backbone + PyTorch decoder) ---
    if no_imgfeat:
        Log.info("[Preprocess] Skipping SAM-3D-Body (--no-imgfeat)")
    elif not Path(paths.vit_features).exists():
        sam3db_extractor, sam3db_backend = (
            load_sam3db(cfg=cfg) if not force_pytorch else (None, "pytorch")
        )
        if sam3db_extractor is None:
            from gem.utils.sam3db_extractor import SAM3DBExtractor

            sam3db_extractor = SAM3DBExtractor(
                checkpoint_path=cfg.get("sam3d_ckpt_path", None),
                mhr_path=cfg.get("sam3d_mhr_path", None),
                device="cuda:0",
            )
        t0 = time.time()
        sam3d_results = sam3db_extractor.extract_video_features(video_path, bbx_xys)
        pose_tokens = sam3d_results["pose_tokens"]
        transls = sam3d_results["transls"]
        _timings["SAM-3D-Body features"] = time.time() - t0
        Log.info(
            f"[SAM-3D-Body] {sam3db_backend.upper()} backbone: {_timings['SAM-3D-Body features']:.2f}s for {L} frames"
        )

        K_fullimg = estimate_K(W, H).repeat(pose_tokens.shape[0], 1, 1)
        pred_cam = (
            get_a_pred_cam(transls, bbx_xys, K_fullimg)
            if transls.abs().sum() > 0
            else torch.zeros(L, 3)
        )
        vit_features = {"pose_tokens": pose_tokens, "pred_cam": pred_cam}
        torch.save(vit_features, paths.vit_features)

    Log.info("[Preprocess] Done (fast mode)")


def load_data_dict(cfg, no_imgfeat=False):
    """Load preprocessed data into the format expected by GEM.predict()."""
    _ensure_pipeline_deps()
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    vitpose = torch.load(paths.vitpose)
    if isinstance(vitpose, tuple):
        vitpose = vitpose[0]
    bbx_xys = torch.load(paths.bbx)["bbx_xys"].clone()

    if cfg.static_cam:
        T_w2c = torch.eye(4).unsqueeze(0).repeat(length, 1, 1)
    else:
        traj = torch.as_tensor(torch.load(paths.slam)).float()
        T_w2c = normalize_T_w2c(traj)

    R_w2c = T_w2c[:, :3, :3]
    t_w2c = T_w2c[:, :3, 3]

    if no_imgfeat:
        f_imgseq = torch.zeros(length, 1024)
        noisy_pred_cam = None
        has_img_mask = torch.zeros(length).bool()
    else:
        vit_features = torch.load(paths.vit_features)
        f_imgseq = vit_features["pose_tokens"]
        noisy_pred_cam = vit_features.get("pred_cam", None)
        has_img_mask = torch.ones(length).bool()

    return {
        "meta": [{"vid": Path(cfg.video_path).stem}],
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "cam_tvel": compute_cam_tvel(t_w2c),
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "f_imgseq": f_imgseq,
        "noisy_pred_cam": noisy_pred_cam,
        "has_text": torch.tensor([False]),
        "mask": {
            "valid": torch.ones(length).bool(),
            "has_img_mask": has_img_mask,
            "has_2d_mask": torch.ones(length).bool(),
            "has_cam_mask": torch.ones(length).bool(),
            "has_audio_mask": torch.zeros(length).bool(),
            "has_music_mask": torch.zeros(length).bool(),
        },
    }


@torch.no_grad()
def run_inference_fast(cfg, data, force_pytorch=False, no_imgfeat=False, use_ddim=False):
    """Run GEM inference with ONNX/TRT denoiser if available.

    When *use_ddim* is True, always use PyTorch with 50-step DDIM sampling
    instead of single-step regression.  This produces significantly better
    results when image features are absent (--no-imgfeat).
    """
    _ensure_pipeline_deps()
    import hydra as _hydra

    if use_ddim:
        force_pytorch = True  # DDIM requires PyTorch (ONNX model bakes in regression timestep)

    denoiser_runner, denoiser_backend = (
        (None, "pytorch") if force_pytorch else load_denoiser(no_imgfeat=no_imgfeat)
    )

    if denoiser_backend != "pytorch":
        # ONNX/TRT path: compose inputs and run denoiser directly
        Log.info(f"[Denoiser] Using {denoiser_backend.upper()} backend")
        t0 = time.time()

        # Pass raw kp2d — the ONNX wrapper normalizes internally via _normalize_kp2d.
        batch = {
            "obs": data["kp2d"].unsqueeze(0),
            "bbx_xys": data["bbx_xys"].unsqueeze(0),
            "K_fullimg": data["K_fullimg"].unsqueeze(0),
            "f_imgseq": data["f_imgseq"].unsqueeze(0),
            "f_cam_angvel": data["cam_angvel"].unsqueeze(0),
        }

        pred_x, pred_cam = run_denoiser_onnx(denoiser_runner, denoiser_backend, batch)
        _timings["GEM denoiser"] = time.time() - t0
        Log.info(
            f"[Denoiser] {denoiser_backend.upper()} inference: {_timings['GEM denoiser']:.2f}s"
        )

        # Decode with EnDecoder (still PyTorch — lightweight)

        model = _hydra.utils.instantiate(cfg.model, _recursive_=False)
        ckpt_path = cfg.ckpt_path
        if ckpt_path is None:
            from gem.utils.hf_utils import download_checkpoint

            ckpt_path = download_checkpoint()
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()

        endecoder = model.endecoder
        if endecoder.obs_indices_dict is None:
            endecoder.build_obs_indices_dict()

        decode_dict = endecoder.decode(pred_x)
        if "scale_params" in decode_dict:
            sp = decode_dict["scale_params"].clone()
            sp[..., 0] = sp[..., 0].clamp(0.7, 1.0)
            decode_dict["scale_params"] = sp

        # Build body params incam
        pred_body_params_incam = {
            "body_pose": decode_dict["body_pose"][0],
            "global_orient": decode_dict["global_orient"][0],
            "transl": compute_transl_full_cam(
                pred_cam, data["bbx_xys"].unsqueeze(0).cuda(), data["K_fullimg"].unsqueeze(0).cuda()
            )[0],
        }
        if "identity_coeffs" in decode_dict:
            pred_body_params_incam["identity_coeffs"] = decode_dict["identity_coeffs"][0]
        if "scale_params" in decode_dict:
            pred_body_params_incam["scale_params"] = decode_dict["scale_params"][0]

        # Build body params global (simplified - use pipeline's get_body_params_w_Rt_v2)
        from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2

        if "global_orient_gv" in decode_dict and "local_transl_vel" in decode_dict:
            body_params_global = get_body_params_w_Rt_v2(
                global_orient_gv=decode_dict["global_orient_gv"],
                local_transl_vel=decode_dict["local_transl_vel"],
                global_orient_c=decode_dict["global_orient"],
                cam_angvel=data["cam_angvel"].unsqueeze(0).cuda(),
            )
            body_params_global = {
                "body_pose": decode_dict["body_pose"][0],
                "global_orient": body_params_global["global_orient"][0],
                "transl": body_params_global["transl"][0],
            }
            if "identity_coeffs" in decode_dict:
                body_params_global["identity_coeffs"] = decode_dict["identity_coeffs"][0]
            if "scale_params" in decode_dict:
                body_params_global["scale_params"] = decode_dict["scale_params"][0]
        else:
            body_params_global = pred_body_params_incam

        pred = {
            "body_params_incam": detach_to_cpu(pred_body_params_incam),
            "body_params_global": detach_to_cpu(body_params_global),
            "K_fullimg": data["K_fullimg"],
        }
        return pred

    else:
        # Full PyTorch path (fallback)
        mode_str = "DDIM (50 steps)" if use_ddim else "regression"
        Log.info(f"[Denoiser] Using PyTorch backend ({mode_str})")
        model = _hydra.utils.instantiate(cfg.model, _recursive_=False)
        ckpt_path = cfg.ckpt_path
        if ckpt_path is None:
            from gem.utils.hf_utils import download_checkpoint

            ckpt_path = download_checkpoint()
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()

        if use_ddim:
            # Override regression_only to enable DDIM sampling.
            # The model's test_gen_only_diffusion is already set up with
            # 50-step DDIM schedule from the ddim.yaml config.
            model.pipeline.regression_only = False
            Log.info("[Denoiser] Overriding regression_only=False for DDIM sampling")

        pred = model.predict(data, static_cam=cfg.static_cam, postproc=True)
        return detach_to_cpu(pred)


def _open_cv2_writer(path, width, height, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))


def render_incam(cfg, fps=30):
    """Render in-camera overlay (same as demo_soma.py)."""
    _ensure_pipeline_deps()
    import open3d as o3d

    pred = torch.load(cfg.paths.hpe_results)
    body_params_incam = pred["body_params_incam"]

    device = "cuda:0"
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        pred_c_verts = soma(**to_cuda(body_params_incam))["vertices"]
    faces = soma.faces.long().cuda()

    _, width, height = get_video_lwh(cfg.video_path)
    K = pred["K_fullimg"][0]

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.4, 0.8, 0.4], device=device)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.01, 100.0, float(width), float(height)
    )
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    reader = get_video_reader(cfg.video_path)
    writer = _open_cv2_writer(cfg.paths.incam_video, width, height, fps)
    from tqdm import tqdm

    for i, img_raw in tqdm(enumerate(reader), total=pred_c_verts.shape[0], desc="Render Incam"):
        if i >= pred_c_verts.shape[0]:
            break
        mesh = create_meshes(pred_c_verts[i], faces, color)
        name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())
        depth = np.asarray(renderer.render_to_depth_image())
        mask = (depth < 1.0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.0)
        alpha = mask[..., np.newaxis]
        composite = rendered.astype(np.float32) * alpha + img_raw.astype(np.float32) * (1.0 - alpha)
        writer.write(composite.clip(0, 255).astype(np.uint8)[..., ::-1])
    writer.release()
    reader.close()


def render_global_o3d(cfg, fps=30):
    """Render global view (same as demo_soma.py)."""
    _ensure_pipeline_deps()
    import open3d as o3d
    from gem.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay

    pred = torch.load(cfg.paths.hpe_results)
    body_params_global = pred["body_params_global"]

    device = "cuda:0"
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device=device,
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        soma_out = soma(**to_cuda(body_params_global))
    verts = soma_out["vertices"]
    joints = soma_out["joints"]
    faces_soma = soma.faces.long().cuda()

    # Move to start point face z
    y_min = verts[:, :, 1].min()
    verts[:, :, 1] -= y_min
    joints[:, :, 1] -= y_min
    T_ay2ayfz = compute_T_ayfz2ay(joints[[0]], inverse=True)
    verts = apply_T_on_points(verts, T_ay2ayfz)
    joints = apply_T_on_points(joints, T_ay2ayfz)

    _, width, height = get_video_lwh(cfg.video_path)
    from gem.utils.cam_utils import create_camera_sensor

    _, _, K = create_camera_sensor(width, height, 24)

    scale, cx, cz = get_ground_params_from_points(joints[:, 0], verts)
    ground = get_ground(max(scale, 3) * 1.5, cx, cz)
    position, target, up = get_global_cameras_static_v2(
        verts.cpu().clone(), beta=4.5, cam_height_degree=30, target_center_height=1.0
    )

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]
    color = torch.tensor([0.4, 0.8, 0.4], device=device)
    writer = _open_cv2_writer(cfg.paths.global_video, width, height, fps)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS, np.array([0.577, -0.577, -0.577])
    )
    renderer.scene.camera.set_projection(
        K.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )
    renderer.scene.camera.look_at(target.cpu().numpy(), position.cpu().numpy(), up.cpu().numpy())

    gv, gf, gc = ground
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.shader = Settings.LIT
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_mat)

    from tqdm import tqdm

    for t in tqdm(range(verts.shape[0]), desc="Render Global"):
        mesh = create_meshes(verts[t], faces_soma, color)
        if t > 0:
            renderer.scene.remove_geometry(f"mesh_{t - 1}")
        renderer.scene.add_geometry(f"mesh_{t}", mesh, lit_mat)
        writer.write(np.array(renderer.render_to_image())[..., ::-1])
    writer.release()


def _print_benchmark(num_frames: int):
    """Print a benchmark summary table with time and FPS per stage."""
    # Ordered list of stages to display (skip missing ones).
    stage_order = [
        "Detection (YOLOX+ByteTrack)",
        "VitPose 2D keypoints",
        "SAM-3D-Body features",
        "GEM denoiser",
        "Render (in-camera)",
        "Render (global)",
        "Retargeting (G1)",
    ]
    header = f"Benchmark Summary ({num_frames} frames)"
    width = 64
    print(f"\n{'=' * width}")
    print(f"{header:^{width}}")
    print(f"{'=' * width}")
    print(f"{'Stage':<36} {'Time (s)':>10} {'FPS':>10}")
    print(f"{'-' * width}")
    for stage in stage_order:
        if stage in _timings:
            t = _timings[stage]
            fps_val = num_frames / t if t > 0 else float("inf")
            print(f"{stage:<36} {t:>10.1f} {fps_val:>10.1f}")
    print(f"{'-' * width}")
    if "Total" in _timings:
        t = _timings["Total"]
        fps_val = num_frames / t if t > 0 else float("inf")
        print(f"{'Total':<36} {t:>10.1f} {fps_val:>10.1f}")
    print(f"{'=' * width}\n")


def main():
    _ensure_pipeline_deps()
    total_t0 = time.time()
    args = _parse_args()
    cfg = _build_cfg(args)
    no_imgfeat = args.no_imgfeat

    # Preprocess (detection + vitpose + SAM-3D-Body features)
    t0 = time.time()
    run_preprocess_fast(cfg, force_pytorch=args.force_pytorch, no_imgfeat=no_imgfeat)
    Log.info(f"[Timing] Preprocess: {time.time() - t0:.1f}s")

    # Load data
    data = load_data_dict(cfg, no_imgfeat=no_imgfeat)
    fps = int(cv2.VideoCapture(cfg.video_path).get(cv2.CAP_PROP_FPS) + 0.5) or 30

    # GEM inference (denoiser)
    if not Path(cfg.paths.hpe_results).exists():
        t0 = time.time()
        pred = run_inference_fast(
            cfg,
            data,
            force_pytorch=args.force_pytorch,
            no_imgfeat=no_imgfeat,
            use_ddim=args.ddim,
        )
        _timings.setdefault("GEM denoiser", time.time() - t0)
        Log.info(f"[Timing] GEM inference: {_timings['GEM denoiser']:.1f}s")
        torch.save(pred, cfg.paths.hpe_results)

    # Render
    t0 = time.time()
    render_incam(cfg, fps=fps)
    _timings["Render (in-camera)"] = time.time() - t0

    t0 = time.time()
    render_global_o3d(cfg, fps=fps)
    _timings["Render (global)"] = time.time() - t0

    Log.info(
        f"[Timing] Rendering: {_timings['Render (in-camera)'] + _timings['Render (global)']:.1f}s"
    )

    # Retarget + merge videos
    if args.retarget:
        from gem.utils.kp2d_utils import render_2d_keypoints
        from scripts.demo.retarget_utils import render_g1_robot, run_retarget

        t0 = time.time()
        kp2d_video = str(Path(cfg.output_dir) / "0_kp2d77_overlay.mp4")
        render_2d_keypoints(
            video_path=cfg.video_path,
            vitpose_path=cfg.paths.vitpose,
            bbx_path=cfg.paths.bbx,
            output_path=kp2d_video,
            fps=fps,
        )
        pred = torch.load(cfg.paths.hpe_results)
        csv_buffer = run_retarget(pred["body_params_global"], fps, cfg.paths.retarget_csv)
        render_g1_robot(cfg, csv_buffer, fps=fps)
        _timings["Retargeting (G1)"] = time.time() - t0
        merge_videos_grid_2x2(
            [kp2d_video, cfg.paths.incam_video, cfg.paths.global_video, cfg.paths.retarget_video],
            cfg.paths.incam_global_horiz_video,
        )
    else:
        merge_videos_horizontal(
            [cfg.paths.incam_video, cfg.paths.global_video],
            cfg.paths.incam_global_horiz_video,
        )

    _timings["Total"] = time.time() - total_t0
    num_frames = get_video_lwh(cfg.video_path)[0]
    Log.info(f"[Done] Outputs in {cfg.output_dir}")
    _print_benchmark(num_frames)


if __name__ == "__main__":
    main()
