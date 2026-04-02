# macOS Installation (Apple Silicon)

Run the GEM demo scripts on a MacBook with Apple Silicon.

> **Platform note:** Training and the full offline pipeline (`demo_soma.py`)
> work best with an NVIDIA GPU — see [INSTALL.md](INSTALL.md). The ONNX
> accelerated demo (`demo_soma_onnx.py`) runs well on Apple Silicon using
> ONNX Runtime with CoreML.

## Quick Setup (recommended)

Run the one-step setup script — it handles everything (environment, dependencies,
models):

```bash
git clone --recursive https://github.com/NVlabs/GEM-X.git && cd GEM-X
bash scripts/setup_mac.sh
```

Then run the demo:

```bash
source .venv/bin/activate
python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4
```

The rest of this document explains each step in detail if you prefer manual setup
or need to troubleshoot.

---

## Prerequisites

- macOS 13 (Ventura) or later
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- ~5 GB disk space for models and assets

## Step 1 — Clone with submodules

```bash
git clone --recursive https://github.com/NVlabs/GEM-X.git
cd GEM-X
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

## Step 2 — Create virtual environment

```bash
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
```

## Step 3 — Install PyTorch (Apple Silicon)

```bash
# PyTorch with MPS (Metal Performance Shaders) backend — no CUDA needed
uv pip install torch torchvision
```

Verify MPS is available:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

## Step 4 — Install SOMA body model

```bash
uv pip install -e third_party/soma
cd third_party/soma && git lfs pull && cd ../..
```

## Step 5 — Install GEM and dependencies

```bash
bash scripts/install_env.sh
```

This script detects macOS and automatically skips `detectron2` (which requires
CUDA), installing ONNX Runtime instead.

Or install manually:

```bash
uv pip install -e .
uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'
uv pip install onnxruntime
```

> **Note:** ONNX Runtime on macOS automatically includes the **CoreML Execution
> Provider**, which routes supported operations to the Apple Neural Engine (ANE)
> and GPU.

## Step 6 — Download ONNX models

ONNX models are automatically downloaded from
[HuggingFace](https://huggingface.co/nvidia/GEM-X) on first run. To download
them ahead of time:

```bash
python -c "from gem.utils.hf_utils import download_all_onnx; download_all_onnx()"
```

## Step 7 — SOMA assets for 3D rendering

The rendering pipeline requires SOMA body model assets:

```bash
# Create symlink (assets ship with the SOMA submodule after git lfs pull)
ln -sf third_party/soma/assets inputs/soma_assets
```

## Run the demo

```bash
# ONNX accelerated demo (recommended on macOS)
python scripts/demo/demo_soma_onnx.py \
  --video path/to/video.mp4

# Standard demo (uses PyTorch — slower on macOS)
python scripts/demo/demo_soma.py \
  --video path/to/video.mp4 \
  --ckpt inputs/pretrained/gem_soma.ckpt
```

See [DEMO.md](DEMO.md) for full argument reference and output descriptions.

## Troubleshooting

| Issue | Solution |
|---|---|
| `MPS: False` in PyTorch | Ensure macOS 13+ and `torch>=2.0` |
| `import detectron2` errors | Not needed for `demo_soma_onnx.py`. Run `bash scripts/install_env.sh` which skips detectron2 on macOS |
| `No ONNX/TRT denoiser found` | Download ONNX models (Step 6) |
| YOLOX download fails | YOLOX auto-downloads on first run. Check internet connection |
| Very slow ONNX inference | Check `[ONNX] Loaded ... (EP=...)` log — should show `CoreMLExecutionProvider`. If not, reinstall `onnxruntime` |

## Model backend priority

The ONNX demo automatically selects the fastest available backend:

```
FP16 ONNX  →  INT8 ONNX  →  Full ONNX  →  PyTorch
    ↑                            ↑
quantize_onnx.py --fp16    download_all_onnx()
```

On macOS, VitPose is automatically converted to FP16 on first run (~2-3 min,
one-time). Both VitPose and the denoiser use ONNX Runtime with CoreML EP for
hardware acceleration.
