# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

HF_REPO_ID = "nvidia/GEM-X"

# GEM-SOMA checkpoint
DEFAULT_CKPT_FILENAME = "gem_soma.ckpt"
DEFAULT_CKPT_DIR = "inputs/pretrained"

# ViTPose checkpoint
VITPOSE_CKPT_FILENAME = "vitpose.pth"
VITPOSE_CKPT_DIR = "inputs/checkpoints/vitpose"

# SAM-3D-Body checkpoint
SAM3D_CKPT_FILENAME = "sam3d_body.ckpt"
SAM3D_CONFIG_FILENAME = "model_config.yaml"
SAM3D_CKPT_DIR = "inputs/checkpoints/sam-3d-body-dinov3"

# MHR model
MHR_MODEL_FILENAME = "mhr_model.pt"
MHR_MODEL_DIR = "inputs/mhr_data"

# SOMA scale data
SOMA_SCALE_MEAN_FILENAME = "scale_mean.pth"
SOMA_SCALE_COMPS_FILENAME = "scale_comps.pth"
SOMA_DATA_DIR = "inputs/soma_data"


def _download_hf_file(repo_id, filename, local_dir):
    """Download a single file from HuggingFace Hub if not already present."""
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def download_checkpoint(
    repo_id=HF_REPO_ID, filename=DEFAULT_CKPT_FILENAME, local_dir=DEFAULT_CKPT_DIR
):
    """Download GEM-SOMA checkpoint from HuggingFace Hub if not already cached."""
    return _download_hf_file(repo_id, filename, local_dir)


def download_vitpose_checkpoint(
    repo_id=HF_REPO_ID, filename=VITPOSE_CKPT_FILENAME, local_dir=VITPOSE_CKPT_DIR
):
    """Download ViTPose checkpoint from HuggingFace Hub if not already cached."""
    return _download_hf_file(repo_id, filename, local_dir)


def download_sam3d_checkpoint(repo_id=HF_REPO_ID, local_dir=SAM3D_CKPT_DIR):
    """Download SAM-3D-Body checkpoint and config from HuggingFace Hub.

    Returns the checkpoint path. The config (model_config.yaml) is downloaded
    alongside it so load_sam_3d_body() can find it in the same directory.
    """
    _download_hf_file(repo_id, SAM3D_CONFIG_FILENAME, local_dir)
    return _download_hf_file(repo_id, SAM3D_CKPT_FILENAME, local_dir)


def download_mhr_model(repo_id=HF_REPO_ID, filename=MHR_MODEL_FILENAME, local_dir=MHR_MODEL_DIR):
    """Download MHR model from HuggingFace Hub if not already cached."""
    return _download_hf_file(repo_id, filename, local_dir)


def download_soma_data(repo_id=HF_REPO_ID, local_dir=SOMA_DATA_DIR):
    """Download SOMA scale data (scale_mean.pth, scale_comps.pth) from HuggingFace Hub.

    Returns the directory path containing the downloaded files.
    """
    _download_hf_file(repo_id, SOMA_SCALE_MEAN_FILENAME, local_dir)
    _download_hf_file(repo_id, SOMA_SCALE_COMPS_FILENAME, local_dir)
    return local_dir


# ONNX models for fast demo
ONNX_DIR = "inputs/onnx"
ONNX_MODELS = {
    "vitpose": ("vitpose.onnx", "vitpose.onnx.data"),
    "gem_denoiser": ("gem_denoiser.onnx", "gem_denoiser.onnx.data"),
    "gem_denoiser_no_imgfeat": (
        "gem_denoiser_no_imgfeat.onnx",
        "gem_denoiser_no_imgfeat.onnx.data",
    ),
    "sam3db_backbone": ("sam3db_backbone.onnx",),
}


def download_onnx_model(name, repo_id=HF_REPO_ID, local_dir=ONNX_DIR):
    """Download an ONNX model (and its .data file) from HuggingFace Hub.

    Args:
        name: Model name key, one of: "vitpose", "gem_denoiser",
              "gem_denoiser_no_imgfeat", "sam3db_backbone".

    Returns:
        Path to the main .onnx file.
    """
    if name not in ONNX_MODELS:
        raise ValueError(f"Unknown ONNX model '{name}'. Available: {list(ONNX_MODELS)}")
    files = ONNX_MODELS[name]
    # Files live under onnx/ in the HF repo; download into the parent of
    # local_dir so that onnx/<file> maps to <local_dir>/<file>.
    parent = str(Path(local_dir).parent)
    for f in files:
        _download_hf_file(repo_id, f"onnx/{f}", parent)
    return str(Path(local_dir) / files[0])


def download_all_onnx(repo_id=HF_REPO_ID, local_dir=ONNX_DIR):
    """Download all ONNX models for the fast demo pipeline."""
    for name in ONNX_MODELS:
        download_onnx_model(name, repo_id, local_dir)
    return local_dir
