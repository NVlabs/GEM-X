<h1 align="center">GEM: A Generalist Model for Human Motion</h1>

<p align="center">
  <em>Monocular whole-body 3D human pose estimation using the SOMA body model</em>
</p>

<p align="center">
  <a href="https://research.nvidia.com/labs/dair/gem/"><img src="https://img.shields.io/badge/Project-Page-0099cc"></a>
  <a href="https://arxiv.org/abs/2505.01425"><img src="https://img.shields.io/badge/arXiv-2505.01425-b31b1b.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.10%2B-ee4c2c"></a>
  <a href="https://developer.nvidia.com/cuda/toolkit"><img src="https://img.shields.io/badge/CUDA-12.6%2B-76b900"></a>
</p>

---

<p align="center">
  <img src="assets/demo_2d_keypoints.gif" width="30%" alt="2D Keypoint Overlay">
  <img src="assets/demo_incam_mesh.gif" width="30%" alt="In-Camera Mesh">
</p>
<p align="center">
  <img src="assets/demo_global_mesh.gif" width="30%" alt="Global Mesh">
  <img src="assets/demo_g1_retarget.gif" width="30%" alt="G1 Robot Retarget">
</p>
<p align="center">
  <sub>2D Keypoint Overlay &nbsp;|&nbsp; In-Camera Mesh &nbsp;|&nbsp; Global Mesh &nbsp;|&nbsp; Retargeted G1 Motion</sub>
</p>


## Overview

GEM is a video-based 3D human pose estimation model developed by NVIDIA. It recovers full-body 77-joint motion — including body, hands, and face — from monocular video using the [**SOMA**](https://github.com/NVlabs/SOMA-X) parametric body model. The pipeline handles dynamic cameras and recovers global motion trajectories. GEM includes a bundled 2D pose estimation model that detects 77 SOMA keypoints, making the system fully self-contained. Licensed under Apache 2.0 for commercial use.

## Key Features

- **77-joint SOMA body model** — full body, hands, and face articulation
- **Bundled 2D keypoint detector** — 2D pose estimator trained for SOMA 77-joint skeleton
- **Camera-space motion recovery** — camera-space human motion estimation from dynamic monocular video
- **World-space motion recovery** — world-space human motion estimation from dynamic monocular video
- **Humanoid robot retargeting** — retarget recovered motion to Unitree G1 robot via [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
- **Apache 2.0 licensed** — commercially usable, trained on NVIDIA-owned data only

## Research Version: Multi-Modal Conditioning

Looking for **multi-modal motion generation** (text, audio, music conditioning)? Check out [**GEM-SMPL**](https://github.com/NVlabs/GENMO), our research model using the SMPL body model that supports both motion estimation and generation from diverse input modalities. Presented at **ICCV 2025 (Highlight)**.

## Quick Start

```bash
# 1. Clone
git clone --recursive https://github.com/NVlabs/GEM-X.git && cd GEM-X

# 2. Setup environment
pip install uv && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/soma && cd third_party/soma && git lfs pull && cd ../..
bash scripts/install_env.sh

# 3. Run demo
python scripts/demo/demo_soma.py --video path/to/video.mp4 --ckpt inputs/pretrained/gem_soma.ckpt

# 4. (Optional) Retarget to G1 robot
uv pip install -e third_party/soma-retargeter
python scripts/demo/demo_soma.py --video path/to/video.mp4 --retarget
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Documentation

| Document | Description |
|---|---|
| [Installation](docs/INSTALL.md) | Prerequisites, step-by-step setup, Docker, troubleshooting |
| [Demo](docs/DEMO.md) | Full 3D pipeline, 2D keypoint-only demo, output formats |
| [Training & Evaluation](docs/TRAINING.md) | Dataset preparation, training commands, config system |
| [Model Overview](docs/MODEL_OVERVIEW.md) | Architecture, SOMA body model, bundled 2D pose model |
| [Related Projects](docs/RELATED_PROJECTS.md) | GENMO, SOMA, ecosystem cross-references |

## Pretrained Models

| Model | Body Model | Joints | Download |
|---|---|---|---|
| GEM (SOMA) | SOMA | 77 (body + hands + face) | [gem_soma.ckpt](https://huggingface.co/nvidia/GEM-X) |

Place checkpoints under `inputs/pretrained/` or pass the path via `--ckpt`. The demo scripts will automatically download the checkpoint from HuggingFace if `--ckpt` is not provided.

## Related Humanoid Work at NVIDIA
GEM is part of a larger effort to enable humanoid motion data for robotics, physical AI, and other applications.

Check out these related works:
* [GENMO](https://github.com/NVlabs/GENMO)
* [SOMA Body Model](https://github.com/NVlabs/SOMA-X)
* [BONES-SEED Dataset]()
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
* [SOMA Retargeter]()
* [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
* [Kimodo](https://github.com/nv-tlabs/kimodo)


## Citation

```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: A GENeralist Model for Human MOtion},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```


## License

This project is released under [Apache 2.0](LICENSE). This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for specifics. 

## GOVERNING TERMS: 

Use of the source code is governed by the [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md), Version 2.0. Use of the associated model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). 
