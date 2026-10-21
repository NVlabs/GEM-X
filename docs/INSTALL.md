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
python -m pip install -e '.[video]'   # ONNXRuntime person detector
python -m pip install -e '.[hand]'    # ONNXRuntime hand detector
python -m pip install -e '.[render]'  # Open3D mesh/video rendering
python -m pip install -e '.[train]'   # Lightning and Hydra training integration
python -m pip install -e '.[eval]'    # Native Hand evaluation and caller-defined evaluators
python -m pip install -e '.[dev]'     # Tests, formatting and wheel builds
```

Automatic `motion-video --task body` requires the `ffmpeg` executable on PATH and
a local person detector checkpoint. The `video` extra installs CPU ONNXRuntime;
Pose and Motion execution follow `--device` independently of the detector.
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

The base installation includes `py-soma-x==0.3.3`. Verify the installed runtime:

```bash
python -c "import soma; print(soma.__version__); print(soma.__file__)"
```

Geometry assets are supplied separately. Obtain the SOMA 0.3.3 assets through
[SOMA-X's documented asset setup](https://github.com/NVlabs/SOMA-X) and pass their
local directory as `--soma-assets` or the corresponding Python API argument.

- **Body Pose:** native body rig, procedural/reference data and corrective weights.
  Body Pose uses GEM's Pose twist-binding profile, including the `t_pose_world`
  reference in the matching assets.
- **Hand Pose:** `SOMAHand.npz` and its MHR identity model and mesh assets. These
  assets do not require the SAM-3D-Body Python package.
- **Body Motion:** the Motion geometry profile for contact/IK and mesh
  reconstruction, with MID corrective assets for rendering.

Use an asset set matched to the model and SOMA 0.3.3 runtime. An older body asset
directory may lack the reference data needed by the Body Pose profile. GEM selects
the Body Pose and Motion geometry conventions separately.

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
