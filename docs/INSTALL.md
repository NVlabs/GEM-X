# Installation

[Overview](../README.md) · [Pretrained models](PRETRAINED_MODELS.md) ·
[Inference guide](DEMO.md) · [Training](TRAINING.md)

Install GEM from the repository root, then prepare the
[SOMA assets](#soma-code-and-assets) and [checkpoints](#checkpoints) for your task.

GEM requires **Python 3.10 or newer** and **PyTorch 2.10 or newer within the 2.x
series**. For CUDA, install a PyTorch build compatible with your driver in the
environment before the GEM installation step below. Linux ARM64 CUDA users should
first prepare the
[ARM64 runtime dependencies](#arm64-cuda-runtime).

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, with uv:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Optional dependencies

Choose the extras needed for your workflow; they can be combined, for example
`python -m pip install -e '.[video,hand,render]'` on Linux x86-64.

```bash
python -m pip install -e '.[video]'   # Body Pose/Motion ONNXRuntime person detector
python -m pip install -e '.[hand]'    # ONNXRuntime hand detector
python -m pip install -e '.[render]'  # Open3D mesh/video rendering
python -m pip install -e '.[train]'   # Lightning and Hydra training integration
python -m pip install -e '.[eval]'    # Native Hand evaluation and caller-defined evaluators
python -m pip install -e '.[dev]'     # Tests, formatting and wheel builds
```

Automatic Body Pose (`pose-image` or `pose-video` with `--task body` and no
`--boxes`) and Body Motion share a local person detector checkpoint; follow the
[YOLOX download and extraction steps](PRETRAINED_MODELS.md#person-detector-for-body-motion).
The `video` extra installs CPU ONNXRuntime. Body Pose defaults to
`weights/yolox_humanart.onnx`; use `--body-detector PATH` to choose another location.
Its `--detector-device` defaults to `cpu`; Motion uses `detector.device` in its
configuration. Pose and Motion execution follow `--device` independently.
Supplied Pose boxes bypass detection and do not require the `video` extra.

Automatic `motion-video --task body` requires the `ffmpeg` executable on PATH.
On Debian/Ubuntu, install FFmpeg with:

```bash
sudo apt-get install ffmpeg
```

Automatic hand boxes require a local `hand_detector.onnx` from
[nvidia/RT-DETR-Hand-Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1), obtained under its applicable terms and supplied
through `--hand-detector`. The `hand` extra installs CPU ONNXRuntime.
`--detector-device` defaults to `cpu`, independently of Pose's `--device`.
Saved hand box directories and unified annotations do not require this extra.
See [Hand input providers](DEMO.md#hand-image-or-framewise-video).

A CUDA detector requires an ONNXRuntime installation with
`CUDAExecutionProvider`. GEM uses the requested provider and requires you to
supply detector weights locally.

## Contact/IK dependencies

Contact/IK postprocessing requires the base package and matching SOMA 0.3.3 body
assets. For a saved Motion prediction, run:

```bash
gem-infer motion-cache --task body --input motion.pt --postprocess \
  --soma-assets /path/to/soma-assets --output processed.pt
```

This path needs the core rig, identity and procedural assets, without model
checkpoints or a release configuration. Corrective weights are required only when
rendering meshes.

Supplying both `--config` and `--checkpoint` requests a new Motion forward before
postprocessing. See [Contact/IK input restrictions](DEMO.md#contactik-postprocessing).

## Rendering dependencies

Mesh rendering is supported on **Linux x86-64 with Open3D 0.19.0**, installed by
the `render` extra. Other Open3D versions, including 0.20, are unsupported and
rejected by the mesh renderer. Base Pose/Motion inference does not require Open3D.

Calibrated Hand source-camera skeleton overlays use OpenCV from the base package
and do not create an Open3D context. See
[Pose overlays](DEMO.md#render-saved-pose) for the applicable camera paths.

Video export requires the system `ffmpeg` executable. On headless Debian/Ubuntu,
the default `software` mesh backend uses Mesa EGL with Open3D 0.19:

```bash
sudo apt-get install ffmpeg libegl-mesa0 libgl1-mesa-dri
```

The default `software` backend selects Mesa's GLVND vendor file
`/usr/share/glvnd/egl_vendor.d/50_mesa.json` and `EGL_PLATFORM=surfaceless` for
the current process. `--backend egl` uses caller-selected EGL. Choose the backend
before importing Open3D or creating a rendering context.

The software mesh renderer runs on the CPU, including when `--device cuda` is used.
For saved Motion results, `--device` controls SOMA mesh reconstruction separately.
The assets required for rendering depend on the saved result:

| Saved result | Required assets |
| --- | --- |
| Body or Hand Pose | Source media; geometry is already saved in the prediction |
| Body Motion | Source media and matching SOMA 0.3.3 MID body assets, including corrective weights |

Neither rendering path requires model checkpoints. See
[Pose overlays](DEMO.md#render-saved-pose) and
[Motion camera/world views](DEMO.md#render-saved-motion) for commands.

For Linux ARM64 CUDA inference, install GEM without the `render` extra. Transfer
the saved predictions and source media to a Linux x86-64/Open3D 0.19 environment
for mesh rendering, along with SOMA assets for Motion. Inference and contact/IK
do not need to run again.

## SOMA code and assets

The base GEM installation includes `py-soma-x==0.3.3`. The geometry data is a
separate download from [nvidia/SOMA-X](https://huggingface.co/nvidia/SOMA-X/tree/v0.3.3).
Use the **`v0.3.3` asset revision** with this runtime. Run the commands below from
the GEM repository root, with the same environment used for GEM:

```bash
python -m pip install "py-soma-x==0.3.3" huggingface_hub
python -c "import soma; print(soma.__version__); print(soma.__file__)"
```

First download the shared body/hand rig and shape data:

```bash
hf download nvidia/SOMA-X \
  SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json \
  --revision v0.3.3 --local-dir assets/soma
```

For **Body Pose or Body Motion mesh rendering**, add the corrective weights:

```bash
hf download nvidia/SOMA-X correctives_model.pt \
  --revision v0.3.3 --local-dir assets/soma
```

For **Hand Pose**, add the hand mapping and MHR identity assets. Keep the `MHR/`
subdirectory created by the download command:

```bash
hf download nvidia/SOMA-X SOMAHand.npz \
  MHR/mhr_model_lod1.pt MHR/base_body_lod1.obj MHR/SOMA_wrap_lod1.obj \
  --revision v0.3.3 --local-dir assets/soma
```

The required files depend on the task:

| Task | Files in the SOMA asset directory |
| --- | --- |
| Body Pose | Shared rig/shape files plus `correctives_model.pt` |
| Body Motion Contact/IK | Shared rig/shape files |
| Body Motion mesh rendering | Shared rig/shape files plus `correctives_model.pt` |
| Hand Pose | Shared rig/shape files, `SOMAHand.npz`, and the three `MHR/` files above |

After downloading all three groups, the directory is:

```text
assets/soma/
├── SOMA_neutral.npz
├── SOMA_template_rig.usda
├── SOMA_procedural_transforms.json
├── correctives_model.pt
├── SOMAHand.npz
└── MHR/
    ├── mhr_model_lod1.pt
    ├── base_body_lod1.obj
    └── SOMA_wrap_lod1.obj
```

Pass **`assets/soma` itself** to `--soma-assets`; the rig files should be directly
inside that directory. For example:

```bash
export GEM_SOMA_ASSETS="$PWD/assets/soma"
gem-infer pose-image --task body --input image.jpg \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets "$GEM_SOMA_ASSETS" --device cuda --output body.pt
```

The tagged assets supply the reference data used by Body Pose and the Motion
geometry profiles. Hand uses the downloaded MHR model directly and does not
require the SAM-3D-Body Python package. Parameter-only Motion inference needs
no SOMA assets; Contact/IK and mesh reconstruction use the files listed above.
For the upstream asset API and optional workflows, see
[SOMA-X's installation guide](https://github.com/NVlabs/SOMA-X/blob/main/docs/installation.md).

## Checkpoints

Supply checkpoints locally using [Pretrained models](PRETRAINED_MODELS.md),
which lists the weights and matching components for each task.

Body and Hand use the `body` and `hand` architecture presets respectively.
Motion requires a matching network/condition configuration and normalization
statistics. Full GEM training checkpoints include these statistics; historical
checkpoints require their training normalization sidecar.

## Entry points

```bash
gem-infer --help
gem-infer motion-video --help
gem-infer motion-cache --help
gem-render --help
gem-train --help
gem-train pose --help
gem-train motion --help
gem-eval --help
python -m gem.inference --help
```

The source wrappers `scripts/predict.py` and `scripts/train_motion.py` call the
same entry points. Training requires the `train` extra. Pose supports
`--task body` and `--task hand`; `motion-video` and `motion-cache` require
`--task body`.

## Docker

The Dockerfile defaults to `python:3.12-slim` with the `train,video` extras.
`BASE_IMAGE` selects a different Python/PyTorch runtime; `GEM_EXTRAS` selects a
comma-separated set of package extras. An empty value installs base inference.
FFmpeg is included for video processing. Selecting `render` also installs the
Mesa EGL libraries required by the supported Open3D 0.19 renderer.

```bash
docker build -t gem:local .
docker build --build-arg GEM_EXTRAS=video,hand,render -t gem:render .
docker build --build-arg GEM_EXTRAS= -t gem:base .
```

For GPU execution, select a CUDA-enabled PyTorch base with `BASE_IMAGE` and use
the NVIDIA Container Toolkit with `--gpus` at run time. Installation preserves a
compatible PyTorch build already present in the base. Mount local checkpoints,
SOMA assets and input/output directories when running the container.

The Dockerfile sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` so PyTorch's explicit
matrix precision settings, including the IEEE geometry/IK scopes, take effect.
When using conda or an NGC container outside this Dockerfile, set the same value
before starting Python:

```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
```

### ARM64 CUDA runtime

For Linux ARM64/Python 3.12 CUDA inference with
`nvcr.io/nvidia/pytorch:26.03-py3`, prepare its native dependencies before
installing GEM or building the GEM image:

1. Install an OpenUSD `usd-core` wheel matching the runtime's Python version and
   architecture, or make it available through the runtime's pip index. SOMA 0.3.3
   requires the installed distribution; exposing a `pxr` directory through
   `PYTHONPATH` alone does not satisfy that dependency.
2. Build `cholespy==2.2.0` with `nanobind<3`, then install the wheel in the base.

For the cholespy build, use a pip version supporting `--build-constraint` and
create `build-constraints.txt` containing:

```text
nanobind<3
```

Then build and install the resulting wheel normally:

```bash
python -m pip wheel --build-constraint build-constraints.txt \
  --wheel-dir wheels cholespy==2.2.0
python -m pip install --find-links wheels cholespy==2.2.0
```

The constraint applies to build tooling. With OpenUSD and cholespy installed in
the prepared base, the Dockerfile resolves SOMA's dependencies through ordinary
pip installation.

After preparing that ARM64 inference runtime, select it explicitly without the
`render` extra:

```bash
docker build \
  --build-arg BASE_IMAGE=your-arm64-pytorch-with-usd:26.03 \
  --build-arg GEM_EXTRAS=train,video,hand \
  -t gem:arm64-inference .
```

Use the [Linux x86-64/Open3D 0.19 environment](#rendering-dependencies) to render
saved meshes. Native ARM64 mesh rendering is unsupported.

## Retargeting

Robot retargeting is outside this package. Pose and Motion estimation do not
require a retargeter installation.
